/* =====================================================================================
 * VestiAI — client-side state store + tiny SVG chart helpers
 * ===================================================================================== */
(function () {
  'use strict';
  const { util } = window.VestiAI;

  const listeners = new Set();

  const state = {
    /* server status */
    status: null,
    devices: null,
    settings: null,
    /* closet */
    garments: [],
    closetStats: {},
    selectedGarment: util.store.get('selectedGarment', null),
    /** full garment data (RGB + alpha) for the selected garment, loaded client-side */
    garmentImage: null,
    garmentAlpha: null,
    garmentMeta: null,
    /* live */
    live: {
      running: false,
      gestureMode: false,
      quality: util.store.get('quality', 'fast'),
      mirror: util.store.get('mirror', true),
      showSkeleton: util.store.get('showSkeleton', false),
      smoothing: util.store.get('smoothing', true),
      serverRefine: false,
      fps: 0,
      latency: 0,
      poseBackend: '—',
      personDetected: false,
      garmentVisible: false,
      transformMode: '—',
      aiState: 'off',
      gesture: 'unknown',
      stats: { frames: 0, aiFrames: 0, dropped: 0 },
      lastMessage: '',
    },
    /* training */
    training: null,
    trainingMode: 'QUICK_DEMO',
    trainingPoll: null,
    /* misc */
    view: 'home',
    toasts: [],
  };

  const store = {
    state,
    subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    emit(event) { listeners.forEach((fn) => { try { fn(event, state); } catch (e) { console.warn(e); } }); },
    set(patch, event) {
      Object.assign(state, patch);
      store.emit(event || 'change');
    },
    /* ---------------------------------------------------------------- helpers */
    garmentByKey(key) { return state.garments.find((g) => g.key === key) || null; },
    setSelectedGarment(key) {
      state.selectedGarment = key;
      util.store.set('selectedGarment', key);
      store.emit('garment');
    },
    setQuality(mode) {
      state.live.quality = mode;
      util.store.set('quality', mode);
      store.emit('quality');
    },
    setMirror(on) { state.live.mirror = !!on; util.store.set('mirror', !!on); store.emit('mirror'); },
    setSkeleton(on) { state.live.showSkeleton = !!on; util.store.set('showSkeleton', !!on); },
    setSmoothing(on) { state.live.smoothing = !!on; util.store.set('smoothing', !!on); },
    /** Cycle to the next/previous garment (used by buttons and gestures). */
    cycleGarment(direction) {
      const list = state.garments;
      if (!list.length) return null;
      const index = list.findIndex((g) => g.key === state.selectedGarment);
      const next = index < 0
        ? 0
        : (index + (direction || 1) + list.length) % list.length;
      store.setSelectedGarment(list[next].key);
      return list[next];
    },
  };

  /* ==================================================================== charts */
  const charts = {
    /**
     * Render one or more series into an <svg> element.
     * series: [{ name, color, points: [[x, y], ...] }]
     */
    line(svg, series, options) {
      if (!svg) return;
      const opts = Object.assign({ width: 400, height: 180, pad: 22, baseline: false }, options || {});
      const valid = (series || []).filter((s) => s.points && s.points.length);
      svg.setAttribute('viewBox', `0 0 ${opts.width} ${opts.height}`);
      svg.innerHTML = '';
      if (!valid.length) {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', opts.width / 2); text.setAttribute('y', opts.height / 2);
        text.setAttribute('fill', '#6f7889'); text.setAttribute('font-size', '12');
        text.setAttribute('text-anchor', 'middle'); text.textContent = 'no data yet';
        svg.appendChild(text);
        return;
      }
      const xs = valid.flatMap((s) => s.points.map((p) => p[0]));
      const ys = valid.flatMap((s) => s.points.map((p) => p[1]));
      let minX = Math.min(...xs); let maxX = Math.max(...xs);
      let minY = Math.min(...ys); let maxY = Math.max(...ys);
      if (minX === maxX) maxX = minX + 1;
      if (minY === maxY) { minY -= 0.5; maxY += 0.5; }
      const padY = (maxY - minY) * 0.12;
      minY -= padY; maxY += padY;

      const sx = (x) => opts.pad + ((x - minX) / (maxX - minX)) * (opts.width - opts.pad * 2);
      const sy = (y) => opts.height - opts.pad - ((y - minY) / (maxY - minY)) * (opts.height - opts.pad * 2);

      const ns = 'http://www.w3.org/2000/svg';
      /* grid */
      for (let i = 0; i <= 3; i += 1) {
        const y = opts.pad + (i / 3) * (opts.height - opts.pad * 2);
        const line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', opts.pad); line.setAttribute('x2', opts.width - opts.pad);
        line.setAttribute('y1', y); line.setAttribute('y2', y);
        line.setAttribute('stroke', 'rgba(255,255,255,.07)'); line.setAttribute('stroke-width', '1');
        svg.appendChild(line);
      }
      valid.forEach((s) => {
        const path = document.createElementNS(ns, 'polyline');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', s.color || '#7aa2f7');
        path.setAttribute('stroke-width', '2');
        path.setAttribute('stroke-linejoin', 'round');
        path.setAttribute('points', s.points.map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' '));
        svg.appendChild(path);
        const last = s.points[s.points.length - 1];
        const dot = document.createElementNS(ns, 'circle');
        dot.setAttribute('cx', sx(last[0])); dot.setAttribute('cy', sy(last[1])); dot.setAttribute('r', '3');
        dot.setAttribute('fill', s.color || '#7aa2f7');
        svg.appendChild(dot);
      });
      /* labels */
      const label = (x, y, text, anchor) => {
        const t = document.createElementNS(ns, 'text');
        t.setAttribute('x', x); t.setAttribute('y', y); t.setAttribute('fill', '#6f7889');
        t.setAttribute('font-size', '10'); t.setAttribute('font-family', 'ui-monospace, monospace');
        if (anchor) t.setAttribute('text-anchor', anchor);
        t.textContent = text; svg.appendChild(t);
      };
      label(opts.pad, 12, maxY.toFixed(3));
      label(opts.pad, opts.height - 6, minY.toFixed(3));
      label(opts.width - opts.pad, opts.height - 6, `step ${maxX}`, 'end');
    },

    /** Small inline sparkline for status cards. */
    sparkline(svg, values, color) {
      charts.line(svg, [{ points: (values || []).map((v, i) => [i, v]), color: color || '#22d3ee' }], { height: 54, pad: 6 });
    },
  };

  window.VestiAI.store = store;
  window.VestiAI.charts = charts;
})();
