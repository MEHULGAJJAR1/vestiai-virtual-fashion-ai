/* =====================================================================================
 * VestiAI — shared utilities (no dependencies)
 * Exposed on window.VestiAI.util so every other module can use it.
 * ===================================================================================== */
(function () {
  'use strict';
  window.VestiAI = window.VestiAI || {};

  const util = {
    $(sel, root) { return (root || document).querySelector(sel); },
    $$(sel, root) { return Array.from((root || document).querySelectorAll(sel)); },

    el(tag, attrs, children) {
      const node = document.createElement(tag);
      if (attrs) {
        Object.entries(attrs).forEach(([k, v]) => {
          if (v === null || v === undefined || v === false) return;
          if (k === 'class') node.className = v;
          else if (k === 'html') node.innerHTML = v;
          else if (k === 'text') node.textContent = v;
          else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
          else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
          else node.setAttribute(k, v);
        });
      }
      (children || []).forEach((child) => {
        if (child === null || child === undefined) return;
        node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
      });
      return node;
    },

    show(node, on) { if (node) node.classList.toggle('hidden', !on); },
    setText(sel, value) { const n = typeof sel === 'string' ? util.$(sel) : sel; if (n) n.textContent = value === null || value === undefined ? '—' : String(value); },

    fmt: {
      ms(v) { return v === null || v === undefined || Number.isNaN(v) ? '—' : `${Number(v).toFixed(0)} ms`; },
      num(v, digits) { return v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(digits === undefined ? 3 : digits); },
      pct(v, digits) { return v === null || v === undefined || Number.isNaN(v) ? '—' : `${(Number(v) * 100).toFixed(digits === undefined ? 0 : digits)}%`; },
      mb(v) { return v === null || v === undefined ? '—' : `${Number(v).toFixed(0)} MB`; },
      gb(v) { return v === null || v === undefined ? '—' : `${Number(v).toFixed(1)} GB`; },
      bytes(v) {
        if (!v && v !== 0) return '—';
        const units = ['B', 'KB', 'MB', 'GB'];
        let i = 0; let n = Number(v);
        while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
        return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
      },
      duration(seconds) {
        if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '—';
        const s = Math.max(0, Math.round(seconds));
        const h = Math.floor(s / 3600); const m = Math.floor((s % 3600) / 60); const sec = s % 60;
        if (h) return `${h}h ${m}m`;
        if (m) return `${m}m ${sec}s`;
        return `${sec}s`;
      },
      date(ts) { return ts ? new Date(ts * 1000).toLocaleString() : '—'; },
      title(text) { return String(text || '').replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()); },
    },

    /* ------------------------------------------------------------------ toasts */
    toast(title, body, kind) {
      const host = util.$('#toasts');
      if (!host) return;
      const node = util.el('div', { class: `toast ${kind || ''}` }, [
        util.el('div', { class: 'title', text: title }),
        body ? util.el('div', { class: 'body', text: body }) : null,
      ]);
      host.appendChild(node);
      const timeout = kind === 'err' ? 9000 : 5200;
      setTimeout(() => {
        node.style.transition = 'opacity .3s ease, transform .3s ease';
        node.style.opacity = '0';
        node.style.transform = 'translateX(16px)';
        setTimeout(() => node.remove(), 320);
      }, timeout);
    },

    /* ------------------------------------------------------------------ async */
    debounce(fn, wait) {
      let timer = null;
      return function debounced(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), wait || 200);
      };
    },
    throttle(fn, wait) {
      let last = 0;
      return function throttled(...args) {
        const now = performance.now();
        if (now - last >= (wait || 100)) { last = now; return fn.apply(this, args); }
        return undefined;
      };
    },
    sleep(ms) { return new Promise((r) => setTimeout(r, ms)); },

    /* ------------------------------------------------------------------ canvas */
    fitCanvas(canvas, width, height) {
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      return canvas;
    },
    canvasToBase64(canvas, quality, type) {
      const mime = type || 'image/jpeg';
      const data = canvas.toDataURL(mime, quality === undefined ? 0.86 : quality);
      return data.split(',')[1] || '';
    },
    async loadImage(src) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Could not load image: ${src}`));
        img.src = src;
      });
    },
    drawContain(ctx, image, w, h) {
      const iw = image.width || image.videoWidth;
      const ih = image.height || image.videoHeight;
      if (!iw || !ih) return { scale: 1, dx: 0, dy: 0 };
      const scale = Math.min(w / iw, h / ih);
      const dw = iw * scale; const dh = ih * scale;
      const dx = (w - dw) / 2; const dy = (h - dh) / 2;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(image, dx, dy, dw, dh);
      return { scale, dx, dy, dw, dh };
    },

    /* ------------------------------------------------------------------ math */
    clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); },
    lerp(a, b, t) { return a + (b - a) * t; },
    /** Solve a 3x3 linear system (used for the affine/perspective solvers). */
    solve3(A, b) {
      const M = A.map((row, i) => row.concat([b[i]]));
      for (let col = 0; col < 3; col += 1) {
        let pivot = col;
        for (let row = col + 1; row < 3; row += 1) if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
        if (Math.abs(M[pivot][col]) < 1e-12) return null;
        [M[col], M[pivot]] = [M[pivot], M[col]];
        const p = M[col][col];
        for (let k = col; k < 4; k += 1) M[col][k] /= p;
        for (let row = 0; row < 3; row += 1) {
          if (row === col) continue;
          const factor = M[row][col];
          if (!factor) continue;
          for (let k = col; k < 4; k += 1) M[row][k] -= factor * M[col][k];
        }
      }
      return [M[0][3], M[1][3], M[2][3]];
    },
    /** Full 8-DoF homography from 4 source → 4 destination point correspondences. */
    homography(src, dst) {
      const A = []; const b = [];
      for (let i = 0; i < 4; i += 1) {
        const [x, y] = src[i]; const [u, v] = dst[i];
        A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
        A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
      }
      const M = A.map((row, i) => row.concat([b[i]]));
      for (let col = 0; col < 8; col += 1) {
        let pivot = col;
        for (let row = col + 1; row < 8; row += 1) if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
        if (Math.abs(M[pivot][col]) < 1e-10) return null;
        [M[col], M[pivot]] = [M[pivot], M[col]];
        const p = M[col][col];
        for (let k = col; k < 9; k += 1) M[col][k] /= p;
        for (let row = 0; row < 8; row += 1) {
          if (row === col) continue;
          const f = M[row][col];
          if (!f) continue;
          for (let k = col; k < 9; k += 1) M[row][k] -= f * M[col][k];
        }
      }
      const h = M.map((row) => row[8]);
      h.push(1);
      return h;
    },
    /** Apply a 3x3 homography (row-major, length 9) to a point. */
    applyH(h, x, y) {
      const w = h[6] * x + h[7] * y + h[8];
      return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
    },

    /* ------------------------------------------------------------------ storage */
    store: {
      get(key, fallback) {
        try { const raw = localStorage.getItem(`vestiai:${key}`); return raw === null ? fallback : JSON.parse(raw); }
        catch (e) { return fallback; }
      },
      set(key, value) {
        try { localStorage.setItem(`vestiai:${key}`, JSON.stringify(value)); } catch (e) { /* private mode */ }
      },
      clear() {
        Object.keys(localStorage).filter((k) => k.startsWith('vestiai:')).forEach((k) => localStorage.removeItem(k));
      },
    },

    /** `node` may be an element or a CSS selector — callers use both forms. */
    statusBadge(node, kind, text) {
      const el = typeof node === 'string' ? util.$(node) : node;
      if (!el) return;
      el.className = `badge ${kind || ''}`;
      el.innerHTML = `<span class="dot ${kind || ''}"></span> ${text}`;
    },
    dot(node, kind) {
      const el = typeof node === 'string' ? util.$(node) : node;
      if (el) el.className = `dot ${kind || ''}`;
    },
    download(filename, text, mime) {
      const blob = new Blob([text], { type: mime || 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = util.el('a', { href: url, download: filename });
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1500);
    },
    safe(fn, fallback) {
      try { return fn(); } catch (err) { console.warn('[VestiAI]', err); return fallback; }
    },
  };

  window.VestiAI.util = util;
})();
