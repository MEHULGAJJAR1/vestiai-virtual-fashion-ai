/* =====================================================================================
 * VestiAI — client-side garment alignment + warping
 *
 * A faithful JavaScript port of backend/cv/garment_align.py + occlusion.py so the live
 * experience is computed in the browser at full frame rate. Keeping both implementations
 * numerically equivalent means "Fast mode" looks the same whether it runs locally or on the
 * server (the /api/live/frame endpoint), and the Python versions stay the reference for tests.
 * ===================================================================================== */
(function () {
  'use strict';
  const { util } = window.VestiAI;
  const L = window.VestiAI.LANDMARKS;

  /* Category fit profiles — MUST match backend/cv/garment_align.PROFILES */
  const PROFILES = {
    't-shirt': { widthFactor: 1.30, lengthFactor: 1.30, shoulderOffset: 0.05, hemAnchor: 'hip', hemFactor: 1.0 },
    shirt: { widthFactor: 1.26, lengthFactor: 1.40, shoulderOffset: 0.06, hemAnchor: 'hip', hemFactor: 1.0 },
    jacket: { widthFactor: 1.42, lengthFactor: 1.60, shoulderOffset: 0.04, hemAnchor: 'mid_thigh', hemFactor: 1.45 },
    kurta: { widthFactor: 1.32, lengthFactor: 2.05, shoulderOffset: 0.05, hemAnchor: 'mid_thigh', hemFactor: 1.45 },
    dress: { widthFactor: 1.34, lengthFactor: 2.60, shoulderOffset: 0.05, hemAnchor: 'knee', hemFactor: 2.05 },
    top: { widthFactor: 1.22, lengthFactor: 1.15, shoulderOffset: 0.07, hemAnchor: 'hip', hemFactor: 1.0 },
    sweater: { widthFactor: 1.34, lengthFactor: 1.35, shoulderOffset: 0.04, hemAnchor: 'hip', hemFactor: 1.0 },
    traditional: { widthFactor: 1.45, lengthFactor: 2.30, shoulderOffset: 0.04, hemAnchor: 'mid_thigh', hemFactor: 1.45 },
    formal: { widthFactor: 1.28, lengthFactor: 1.42, shoulderOffset: 0.06, hemAnchor: 'hip', hemFactor: 1.0 },
    bottom: { widthFactor: 1.10, lengthFactor: 2.10, shoulderOffset: 0.00, hemAnchor: 'ankle', hemFactor: 2.75 },
    shoes: { widthFactor: 0.45, lengthFactor: 0.20, shoulderOffset: 0.00, hemAnchor: 'ankle', hemFactor: 2.75 },
    accessory: { widthFactor: 0.60, lengthFactor: 0.35, shoulderOffset: 0.02, hemAnchor: 'shoulder', hemFactor: 0.0 },
    unknown: { widthFactor: 1.28, lengthFactor: 1.40, shoulderOffset: 0.05, hemAnchor: 'hip', hemFactor: 1.0 },
  };

  const align = {
    PROFILES,

    /**
     * Per-category fit multipliers edited on the Settings page. They are a *client-side*
     * presentation tweak (how much fabric the warp covers) and are persisted in
     * localStorage, so a user can dial in a better fit for their own body proportions
     * without retraining or editing Python.
     */
    overrides: util.store.get('fitOverrides', {}),

    setOverrides(next) {
      this.overrides = next || {};
      util.store.set('fitOverrides', this.overrides);
    },

    getOverrides() { return this.overrides; },

    profile(category, overrides) {
      const base = PROFILES[category] || PROFILES.unknown;
      const table = overrides || this.overrides || {};
      const extra = table[category] || null;
      return extra ? Object.assign({}, base, extra) : base;
    },

    /** Extract garment anchors from an alpha canvas (mirrors extract_garment_anchors). */
    garmentAnchors(alphaCanvas) {
      const w = alphaCanvas.width; const h = alphaCanvas.height;
      const ctx = alphaCanvas.getContext('2d', { willReadFrequently: true });
      const data = ctx.getImageData(0, 0, w, h).data;
      let x0 = w; let y0 = h; let x1 = -1; let y1 = -1;
      const columnProfile = new Int32Array(w);
      const rowProfile = new Int32Array(h);
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
          const a = data[(y * w + x) * 4 + 3];
          if (a > 96) {
            if (x < x0) x0 = x; if (x > x1) x1 = x;
            if (y < y0) y0 = y; if (y > y1) y1 = y;
            columnProfile[x] += 1; rowProfile[y] += 1;
          }
        }
      }
      if (x1 < 0) return null;
      const boxH = y1 - y0 + 1;
      const boxW = x1 - x0 + 1;

      /* widest rows of the upper 35% — robust shoulder estimate */
      const searchEnd = Math.min(y1 + 1, Math.floor(y0 + Math.max(4, boxH * 0.35)));
      let span = boxW; let leftEdge = x0; let rightEdge = x1; let best = -1;
      for (let y = y0; y < searchEnd; y += 1) {
        let lo = -1; let hi = -1;
        for (let x = x0; x <= x1; x += 1) {
          if (data[(y * w + x) * 4 + 3] > 96) { if (lo < 0) lo = x; hi = x; }
        }
        if (lo >= 0 && (hi - lo) > best) { best = hi - lo; span = hi - lo; leftEdge = lo; rightEdge = hi; }
      }
      const hemBandStart = Math.max(y0, y1 - Math.max(2, Math.floor(boxH * 0.03)));
      let hemLeft = x0; let hemRight = x1; let foundHem = false;
      for (let x = x0; x <= x1; x += 1) {
        for (let y = hemBandStart; y <= y1; y += 1) {
          if (data[(y * w + x) * 4 + 3] > 96) { if (!foundHem) { hemLeft = x; foundHem = true; } hemRight = x; break; }
        }
      }
      /* keep anchors inside the silhouette bbox */
      const clampX = (v) => util.clamp(v, x0, x1);
      return {
        topLeft: [clampX(leftEdge), y0],
        topRight: [clampX(rightEdge), y0],
        bottomLeft: [clampX(hemLeft), y1 + 1],
        bottomRight: [clampX(hemRight), y1 + 1],
        neck: [(x0 + x1) / 2, y0 + 0.04 * boxH],
        center: [(x0 + x1) / 2, (y0 + y1) / 2],
        width: span,
        height: boxH,
        bbox: [x0, y0, x1 + 1, y1 + 1],
      };
    },

    /** Body anchors from smoothed landmarks (mirrors extract_pose_anchors). */
    poseAnchors(landmarks, profile) {
      if (!landmarks) return null;
      const ls = landmarks[L.LEFT_SHOULDER]; const rs = landmarks[L.RIGHT_SHOULDER];
      if (!ls || !rs || ls.visibility < 0.25 || rs.visibility < 0.25) return null;
      const shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y);
      if (shoulderWidth < 8) return null;

      const cx = (ls.x + rs.x) / 2;
      const shoulderY = (ls.y + rs.y) / 2;
      let lean = 0; let torsoLength;
      const lh = landmarks[L.LEFT_HIP]; const rh = landmarks[L.RIGHT_HIP];
      if (lh && rh && lh.visibility > 0.25 && rh.visibility > 0.25) {
        const hipY = (lh.y + rh.y) / 2; const hipX = (lh.x + rh.x) / 2;
        torsoLength = Math.max(Math.abs(hipY - shoulderY), shoulderWidth * 0.85);
        lean = Math.atan2(hipX - cx, Math.max(1e-3, hipY - shoulderY)) * (180 / Math.PI) * 0.35;
      } else {
        torsoLength = shoulderWidth * 1.15;
      }

      const topY = shoulderY - profile.shoulderOffset * shoulderWidth * 0.9;
      const half = (shoulderWidth * profile.widthFactor) / 2;
      const hemExtra = profile.hemFactor;
      let hemY = topY + shoulderWidth * profile.lengthFactor * 0.75 * (1 + 0.35 * (hemExtra - 1)) + torsoLength * 0.25;
      if (profile.hemAnchor === 'shoulder') hemY = topY + shoulderWidth * 0.5;

      const theta = (lean * Math.PI) / 180;
      const cos = Math.cos(theta); const sin = Math.sin(theta);
      const rotate = (dx, dy) => [cx + dx * cos - dy * sin, topY + dx * sin + dy * cos];

      return {
        topLeft: rotate(-half, 0),
        topRight: rotate(half, 0),
        bottomLeft: rotate(-half * 1.02, hemY - topY),
        bottomRight: rotate(half * 1.02, hemY - topY),
        center: [cx, (topY + hemY) / 2],
        shoulderWidth,
        torsoLength,
        confidence: Math.min(ls.visibility, rs.visibility),
      };
    },

    /** 3x3 homography (row-major array of 9) from garment to body. */
    transform(garmentAnchors, poseAnchors) {
      const src = [garmentAnchors.topLeft, garmentAnchors.topRight, garmentAnchors.bottomRight, garmentAnchors.bottomLeft];
      const dst = [poseAnchors.topLeft, poseAnchors.topRight, poseAnchors.bottomRight, poseAnchors.bottomLeft];
      let h = util.homography(src, dst);
      if (h && h.some((v) => !Number.isFinite(v))) h = null;
      if (!h) h = align.similarity(src, dst);
      return h;
    },

    /** Aspect/rotation-preserving fallback when the 4-point solve degenerates. */
    similarity(src, dst) {
      const sw = Math.hypot(src[1][0] - src[0][0], src[1][1] - src[0][1]) || 1;
      const dw = Math.hypot(dst[1][0] - dst[0][0], dst[1][1] - dst[0][1]);
      const scale = (dw / sw) || 1;
      const angle = Math.atan2(dst[1][1] - dst[0][1], dst[1][0] - dst[0][0]);
      const cos = Math.cos(angle) * scale; const sin = Math.sin(angle) * scale;
      return [cos, -sin, dst[0][0], sin, cos, dst[0][1], 0, 0, 1];
    },

    /** Interpolate a transform (for temporal blending). */
    blend(hA, hB, t) {
      if (!hA) return hB; if (!hB) return hA;
      return hA.map((v, i) => util.lerp(v, hB[i], t));
    },

    /**
     * Draw the garment onto a target context using a grid of affine-approximated quads,
     * which reproduces the full perspective warp (a single canvas transform cannot).
     */
    warp(ctx, garmentCanvas, alphaCanvas, h, cols, rows) {
      const c = cols || 4; const r = rows || 8;
      const sw = garmentCanvas.width; const sh = garmentCanvas.height;
      ctx.save();
      for (let i = 0; i < c; i += 1) {
        for (let j = 0; j < r; j += 1) {
          const sx0 = (i / c) * sw; const sx1 = ((i + 1) / c) * sw;
          const sy0 = (j / r) * sh; const sy1 = ((j + 1) / r) * sh;
          const p00 = util.applyH(h, sx0, sy0);
          const p10 = util.applyH(h, sx1, sy0);
          const p11 = util.applyH(h, sx1, sy1);
          const p01 = util.applyH(h, sx0, sy1);
          /* affine from the top-left triangle: (p00, p01, p10) */
          const a = (p10[0] - p00[0]) / (sx1 - sx0);
          const b = (p10[1] - p00[1]) / (sx1 - sx0);
          const cc = (p01[0] - p00[0]) / (sy1 - sy0);
          const d = (p01[1] - p00[1]) / (sy1 - sy0);
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(p00[0], p00[1]); ctx.lineTo(p10[0], p10[1]);
          ctx.lineTo(p11[0], p11[1]); ctx.lineTo(p01[0], p01[1]);
          ctx.closePath(); ctx.clip();
          ctx.setTransform(a, b, cc, d, p00[0] - a * sx0 - cc * sy0, p00[1] - b * sx0 - d * sy0);
          ctx.drawImage(garmentCanvas, 0, 0);
          ctx.restore();
        }
      }
      ctx.restore();
      void alphaCanvas;
    },

    /** Whether a projected garment lands inside the frame with a sensible size. */
    coverage(h, width, height) {
      const pts = [[0, 0], [1, 0], [0, 1], [1, 1]].map(([u, v]) => util.applyH(h, u * width, v * height));
      const xs = pts.map((p) => p[0]); const ys = pts.map((p) => p[1]);
      const w = Math.max(...xs) - Math.min(...xs);
      const hgt = Math.max(...ys) - Math.min(...ys);
      if (!Number.isFinite(w) || !Number.isFinite(hgt)) return 0;
      return (w * hgt) / (width * height);
    },
  };

  /* ================================================================================
   * Occlusion handling (mirrors backend/cv/occlusion.py)
   * ================================================================================ */
  const occlusion = {
    /**
     * Arm segments whose depth is in front of the torso plane.
     * Returns [{x0,y0,x1,y1, radius}] in pixel coordinates.
     */
    armSegments(landmarks, depthMargin) {
      const margin = depthMargin === undefined ? 0.15 : depthMargin;
      if (!landmarks) return [];
      const ls = landmarks[L.LEFT_SHOULDER]; const rs = landmarks[L.RIGHT_SHOULDER];
      if (!ls || !rs || ls.visibility < 0.3 || rs.visibility < 0.3) return [];
      const torsoZ = (ls.z + rs.z) / 2;
      const span = Math.hypot(ls.x - rs.x, ls.y - rs.y) + 1e-6;
      const segs = [];
      const chains = [[L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST], [L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST]];
      chains.forEach(([sh, el, wr]) => {
        const a = landmarks[sh]; const b = landmarks[el]; const c = landmarks[wr];
        const depth = (p) => (torsoZ - p.z) / span;
        if (b && b.visibility > 0.3 && depth(b) > margin) {
          segs.push({ x0: a.x, y0: a.y, x1: b.x, y1: b.y, radius: span * 0.11 });
        }
        if (c && c.visibility > 0.25 && depth(c) > margin) {
          if (b && b.visibility > 0.25) segs.push({ x0: b.x, y0: b.y, x1: c.x, y1: c.y, radius: span * 0.11 });
          segs.push({ x0: c.x, y0: c.y, x1: c.x, y1: c.y, radius: span * 0.16 });
        }
      });
      return segs;
    },

    /** Punch the occluding limbs out of the garment layer (destination-out). */
    applyToLayer(ctx, landmarks, width, height) {
      const segs = occlusion.armSegments(landmarks);
      if (!segs.length) return 0;
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';
      ctx.lineCap = 'round';
      segs.forEach((s) => {
        ctx.beginPath();
        ctx.lineWidth = s.radius * 2;
        ctx.moveTo(s.x0, s.y0);
        ctx.lineTo(s.x1, s.y1);
        ctx.strokeStyle = 'rgba(0,0,0,1)';
        ctx.stroke();
      });
      ctx.restore();
      return segs.length;
    },

    /** Softly protect the head/neck so a collar never covers the chin. */
    applyHeadProtection(ctx, landmarks, width, height) {
      if (!landmarks) return;
      const nose = landmarks[L.NOSE];
      const le = landmarks[L.LEFT_EAR]; const re = landmarks[L.RIGHT_EAR];
      if (!nose || nose.visibility < 0.3) return;
      let radius = 60;
      let cx = nose.x; let cy = nose.y;
      if (le && re && le.visibility > 0.25 && re.visibility > 0.25) {
        const span = Math.hypot(le.x - re.x, le.y - re.y);
        radius = Math.max(12, span * 1.0);
        cx = (le.x + re.x) / 2; cy = (le.y + re.y) / 2;
      } else {
        radius = Math.max(12, Math.abs(nose.z) > 1 ? Math.abs(nose.z) : 60);
      }
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';
      const gradient = ctx.createRadialGradient(cx, cy, radius * 0.4, cx, cy, radius);
      gradient.addColorStop(0, 'rgba(0,0,0,0.95)');
      gradient.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.ellipse(cx, cy, radius, radius * 1.15, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    },
  };

  window.VestiAI.align = align;
  window.VestiAI.occlusion = occlusion;
})();
