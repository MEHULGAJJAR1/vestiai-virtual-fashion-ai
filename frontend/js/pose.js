/* =====================================================================================
 * VestiAI — in-browser pose + hand tracking (MediaPipe Tasks Vision)
 *
 * The vendored model bundle (frontend/vendor/) is loaded from the local server, so pose
 * tracking works completely offline and no webcam frame ever leaves the machine in Fast
 * mode. If the bundle or model file is missing (for example when index.html is opened
 * directly from disk), `available` stays false and the UI shows an explicit explanation
 * instead of pretending tracking works.
 * ===================================================================================== */
(function () {
  'use strict';
  const { util } = window.VestiAI;

  /* BlazePose landmark indices (identical to the Python side) */
  const L = {
    NOSE: 0, LEFT_EAR: 7, RIGHT_EAR: 8,
    LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
    LEFT_ELBOW: 13, RIGHT_ELBOW: 14, LEFT_WRIST: 15, RIGHT_WRIST: 16,
    LEFT_HIP: 23, RIGHT_HIP: 24, LEFT_KNEE: 25, RIGHT_KNEE: 26,
    LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
  };
  const SKELETON = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
    [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [0, 11], [0, 12]];

  const VENDOR = '/static/vendor';

  const pose = {
    available: false,
    backend: 'none',
    reason: null,
    _module: null,
    _pose: null,
    _hands: null,
    _lastPoseTs: 0,
    _lastHandTs: 0,
    smoothing: new Map(),

    /** Load the MediaPipe module + models. Safe to call repeatedly. */
    async init() {
      if (pose._pose) return true;
      try {
        const module = await import(`${VENDOR}/vision_bundle.mjs`);
        pose._module = module;
      } catch (err) {
        pose.available = false;
        pose.reason = 'MediaPipe Tasks Vision bundle could not be loaded '
          + `(${err && err.message ? err.message : err}). Live tracking needs the app served over http(s) `
          + 'with frontend/vendor/ present. Open http://localhost:8000 instead of the raw file.';
        console.warn('[VestiAI] pose init failed:', err);
        return false;
      }
      try {
        const { FilesetResolver, PoseLandmarker } = pose._module;
        const fileset = await FilesetResolver.forVisionTasks(`${VENDOR}/wasm`);
        pose._pose = await PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${VENDOR}/models/pose_landmarker_lite.task`, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputSegmentationMasks: true,
        });
        pose.available = true;
        pose.backend = 'mediapipe_tasks_gpu';
      } catch (gpuErr) {
        try {
          const { FilesetResolver, PoseLandmarker } = pose._module;
          const fileset = await FilesetResolver.forVisionTasks(`${VENDOR}/wasm`);
          pose._pose = await PoseLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: `${VENDOR}/models/pose_landmarker_lite.task`, delegate: 'CPU' },
            runningMode: 'VIDEO', numPoses: 1, outputSegmentationMasks: true,
          });
          pose.available = true;
          pose.backend = 'mediapipe_tasks_cpu';
        } catch (cpuErr) {
          pose.available = false;
          pose.reason = `Pose model failed to initialise (${cpuErr && cpuErr.message ? cpuErr.message : cpuErr}).`;
          return false;
        }
      }
      return true;
    },

    /** Lazily create the hand landmarker (only when Gesture Mode is switched on). */
    async initHands() {
      if (pose._hands) return true;
      if (!pose._module) { const ok = await pose.init(); if (!ok) return false; }
      try {
        const { FilesetResolver, HandLandmarker } = pose._module;
        const fileset = await FilesetResolver.forVisionTasks(`${VENDOR}/wasm`);
        pose._hands = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${VENDOR}/models/hand_landmarker.task`, delegate: 'GPU' },
          runningMode: 'VIDEO', numHands: 1,
        });
        return true;
      } catch (err) {
        console.warn('[VestiAI] hand landmarker unavailable:', err);
        return false;
      }
    },

    /**
     * Detect pose on a video element.
     * @returns {{detected:boolean, landmarks:Array|null, mask:ImageData|null, ts:number}}
     */
    detect(video, timestampMs) {
      const empty = { detected: false, landmarks: null, mask: null, ts: timestampMs };
      if (!pose._pose || !video || !video.videoWidth) return empty;
      try {
        const result = pose._pose.detectForVideo(video, timestampMs);
        if (!result || !result.landmarks || !result.landmarks.length) return empty;
        const raw = result.landmarks[0].map((p) => ({
          x: p.x * video.videoWidth,
          y: p.y * video.videoHeight,
          z: (p.z || 0) * video.videoWidth,
          visibility: p.visibility === undefined ? 1 : p.visibility,
        }));
        let mask = null;
        if (result.segmentationMasks && result.segmentationMasks.length) {
          const m = result.segmentationMasks[0];
          mask = { data: m.getAsFloat32Array(), width: m.width, height: m.height };
        }
        return { detected: true, landmarks: raw, mask, ts: timestampMs };
      } catch (err) {
        console.warn('[VestiAI] pose detect error:', err);
        return empty;
      }
    },

    detectHands(video, timestampMs) {
      if (!pose._hands || !video || !video.videoWidth) return null;
      try {
        const result = pose._hands.detectForVideo(video, timestampMs);
        if (!result || !result.landmarks || !result.landmarks.length) return null;
        return result.landmarks[0].map((p) => ({ x: p.x * video.videoWidth, y: p.y * video.videoHeight, z: p.z || 0 }));
      } catch (err) {
        return null;
      }
    },

    /**
     * Temporal smoothing — mirrors backend/cv/smoothing.py.
     * Exponential moving average with faster tracking on genuine motion and a short hold
     * window so a dropped detection does not make the garment jump.
     */
    smooth(key, landmarks, options) {
      const opts = Object.assign({ alpha: 0.45, motionAlpha: 0.85, motionScale: 12, maxHoldFrames: 8 }, options || {});
      const previous = pose.smoothing.get(key);
      const now = performance.now();
      if (!landmarks) {
        if (!previous) return null;
        previous.miss += 1;
        if (previous.miss > opts.maxHoldFrames) { pose.smoothing.delete(key); return null; }
        // Extrapolate briefly using the last velocity.
        previous.points = previous.points.map((p) => ({
          x: p.x + previous.velocity.x * 0.5, y: p.y + previous.velocity.y * 0.5,
          z: p.z, visibility: p.visibility * 0.98,
        }));
        return previous.points;
      }
      if (!previous || previous.points.length !== landmarks.length) {
        pose.smoothing.set(key, { points: landmarks.map((p) => ({ ...p })), velocity: { x: 0, y: 0 }, miss: 0, t: now });
        return landmarks.map((p) => ({ ...p }));
      }
      let totalMotion = 0;
      const out = landmarks.map((p, i) => {
        const old = previous.points[i];
        totalMotion += Math.hypot(p.x - old.x, p.y - old.y);
        const alpha = (totalMotion / landmarks.length) > opts.motionScale ? opts.motionAlpha : opts.alpha;
        const w = p.visibility >= 0.3 ? alpha : 0;
        return {
          x: old.x * (1 - w) + p.x * w,
          y: old.y * (1 - w) + p.y * w,
          z: old.z * (1 - w) + p.z * w,
          visibility: p.visibility,
        };
      });
      const first = out[0]; const prevFirst = previous.points[0];
      previous.velocity = { x: (first.x - prevFirst.x) * 0.7, y: (first.y - prevFirst.y) * 0.7 };
      previous.points = out;
      previous.miss = 0;
      previous.t = now;
      return out;
    },

    resetSmoothing() { pose.smoothing.clear(); },

    /** Draw the skeleton on a 2-D context (debug overlay). */
    drawSkeleton(ctx, landmarks, color) {
      if (!landmarks) return;
      ctx.save();
      ctx.strokeStyle = color || 'rgba(34, 211, 238, .9)';
      ctx.lineWidth = 2;
      SKELETON.forEach(([a, b]) => {
        const pa = landmarks[a]; const pb = landmarks[b];
        if (!pa || !pb || pa.visibility < 0.3 || pb.visibility < 0.3) return;
        ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
      });
      ctx.fillStyle = 'rgba(124, 92, 255, .95)';
      landmarks.forEach((p) => {
        if (p.visibility < 0.3) return;
        ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fill();
      });
      ctx.restore();
    },

    status() {
      return {
        available: pose.available,
        backend: pose.backend,
        reason: pose.reason,
        hands: !!pose._hands,
      };
    },

    close() {
      util.safe(() => pose._pose && pose._pose.close());
      util.safe(() => pose._hands && pose._hands.close());
      pose._pose = null; pose._hands = null; pose.available = false;
    },
  };

  window.VestiAI.pose = pose;
  window.VestiAI.LANDMARKS = L;
})();
