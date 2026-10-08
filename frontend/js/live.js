/* =====================================================================================
 * VestiAI — Live Try-On engine
 *
 * Frame loop (Fast mode, all client-side, no network):
 *   video → BlazePose → temporal smoothing → garment anchors + body anchors → homography
 *   → perspective warp → occlusion punch-out → alpha composite → HUD
 *
 * AI mode additionally asks the trained model (server) for a photorealistic frame every
 * `aiIntervalMs`; the returned frame is cross-faded in for a moment and then the live
 * geometric composite takes over again, so the view never freezes and never flickers.
 *
 * Frames are only sent to the server in AI mode (and only downscaled crops), and never in
 * Fast mode — that is what the UI promises the user.
 * ===================================================================================== */
(function () {
  'use strict';
  const { util, api, store, pose, align, occlusion, gestures } = window.VestiAI;
  const liveState = store.state.live;

  const live = {
    video: null,
    overlay: null,
    ctx: null,
    garmentCanvas: null,     // RGB of the selected garment (square canvas)
    garmentAlpha: null,      // alpha mask of the selected garment
    garmentMeta: null,
    garmentLayer: null,      // offscreen layer used for warping + occlusion
    garmentLayerCtx: null,
    stream: null,
    rafId: null,
    sessionId: null,
    gestureController: new gestures.Controller({ cooldownMs: 1300, confirmFrames: 3 }),
    recorder: null,
    recordedChunks: [],
    recordingStartedAt: 0,
    smoothingMatrix: null,
    lastTransform: null,
    lastWarpMode: '—',
    lostFrames: 0,
    frameTimes: [],
    ai: {
      lastRequestAt: 0,
      lastFrame: null,
      lastFrameAt: 0,
      lastLatency: null,
      inflight: false,
      error: null,
      crossfadeUntil: 0,
    },
    serverRefine: {
      ws: null,
      pending: false,
      connected: false,
      lastLatency: null,
    },
    log: [],

    /* ------------------------------------------------------------------ init */
    init() {
      this.video = util.$('#camera');
      this.overlay = util.$('#overlay');
      this.ctx = this.overlay.getContext('2d');
      store.subscribe((event) => {
        if (event === 'garment') this.loadGarment();
      });
      if (store.state.selectedGarment) this.loadGarment();
    },

    /* ------------------------------------------------------------- garment */
    async loadGarment() {
      const key = store.state.selectedGarment;
      if (!key) { this.garmentCanvas = null; this.garmentAlpha = null; this.garmentMeta = null; this.updateGarmentUI(); return; }
      const record = store.garmentByKey(key);
      try {
        const [rgbImg, alphaImg] = await Promise.all([
          util.loadImage(record ? record.urls.image : `/api/garments/${key}/file/image`),
          util.loadImage(record ? record.urls.mask : `/api/garments/${key}/file/mask`),
        ]);
        const canvas = document.createElement('canvas');
        canvas.width = rgbImg.width; canvas.height = rgbImg.height;
        canvas.getContext('2d').drawImage(rgbImg, 0, 0);

        /* Build an RGBA sprite: garment RGB multiplied by its own alpha, so the warp can be
           drawn straight onto the layer without an extra mask pass. */
        const alphaCanvas = document.createElement('canvas');
        alphaCanvas.width = alphaImg.width; alphaCanvas.height = alphaImg.height;
        const actx = alphaCanvas.getContext('2d', { willReadFrequently: true });
        actx.drawImage(alphaImg, 0, 0);
        const imageData = actx.getImageData(0, 0, alphaCanvas.width, alphaCanvas.height);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
          const a = data[i];                    // mask is grayscale: use R as alpha
          data[i] = 255; data[i + 1] = 255; data[i + 2] = 255;
          data[i + 3] = a;
        }
        actx.putImageData(imageData, 0, 0);

        /* Multiply garment RGB by mask alpha for a clean cut-out sprite. */
        const sprite = document.createElement('canvas');
        sprite.width = canvas.width; sprite.height = canvas.height;
        const sctx = sprite.getContext('2d', { willReadFrequently: true });
        sctx.drawImage(canvas, 0, 0);
        const spriteData = sctx.getImageData(0, 0, sprite.width, sprite.height);
        const alphaData = actx.getImageData(0, 0, sprite.width, sprite.height).data;
        for (let i = 0; i < spriteData.data.length; i += 4) {
          spriteData.data[i + 3] = alphaData[i + 3];
        }
        sctx.putImageData(spriteData, 0, 0);

        this.garmentCanvas = sprite;
        this.garmentAlpha = alphaCanvas;
        this.garmentMeta = record || { key, label: key, category: 'unknown' };
        this.lastTransform = null;
        this.updateGarmentUI();
        util.toast('Garment ready', `${this.garmentMeta.label} loaded for live try-on.`, 'ok');
      } catch (err) {
        api.report(err, 'loading garment');
        this.garmentCanvas = null; this.garmentAlpha = null;
        this.updateGarmentUI();
      }
    },

    updateGarmentUI() {
      const slot = util.$('#live-garment-slot');
      const meta = this.garmentMeta;
      if (!meta) {
        slot.className = 'slot';
        slot.innerHTML = '<span class="muted">No garment selected</span>';
        util.setText('#live-category', '—'); util.setText('#live-garment-name', '—');
        const paletteNode = util.$('#live-garment-palette'); if (paletteNode) paletteNode.innerHTML = '';
        return;
      }
      slot.className = 'slot filled';
      slot.innerHTML = `<img src="${meta.urls.preview}" alt="${meta.label}" />`;
      util.setText('#live-category', meta.category_label || util.fmt.title(meta.category));
      util.setText('#live-garment-name', meta.label);
      const palette = util.$('#live-garment-palette');
      if (palette) palette.innerHTML = (meta.palette || []).slice(0, 4).map((c) => `<i style="background:${c};display:inline-block;width:12px;height:12px;border-radius:4px;margin-right:3px;border:1px solid rgba(255,255,255,.2)"></i>`).join('');
      util.dot('#hud-garment-dot', 'ok');
      util.setText('#hud-garment', meta.label.slice(0, 22));
    },

    /* -------------------------------------------------------------- camera */
    async startCamera() {
      if (liveState.running) return;
      util.setText('#live-hint', 'Requesting camera permission…');
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        this.cameraError('This browser does not expose camera APIs. Use a recent Chrome, Edge, Firefox or Safari and serve the app over http(s).');
        return;
      }
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false,
        });
      } catch (err) {
        const name = err && err.name ? err.name : 'Error';
        let message = 'Camera could not be started.';
        if (name === 'NotAllowedError' || name === 'SecurityError') {
          message = 'Camera permission was denied. Allow camera access in your browser (site settings → Camera) and press Start Camera again.';
        } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
          message = 'No camera was found. Connect a webcam and try again.';
        } else if (name === 'NotReadableError' || name === 'TrackStartError') {
          message = 'The camera is already in use by another application. Close it and retry.';
        } else if (name === 'OverconstrainedError') {
          message = 'The camera does not support the requested resolution — retrying with defaults.';
          try {
            this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          } catch (retryErr) { this.cameraError(message); return; }
        } else {
          message = `Camera error (${name}): ${err && err.message ? err.message : 'unknown'}`;
        }
        if (!this.stream) { this.cameraError(message); return; }
      }

      this.video.srcObject = this.stream;
      await this.video.play().catch(() => {});
      await this.waitForVideo();
      this.video.classList.toggle('mirror-off', !liveState.mirror);

      /* pose engine (vendored, local) */
      const ready = await pose.init();
      if (!ready) {
        util.toast('Pose tracking unavailable', pose.reason || 'MediaPipe could not be initialised.', 'warn');
      }
      if (liveState.gestureMode) await pose.initHands();

      liveState.running = true;
      liveState.stats = { frames: 0, aiFrames: 0, dropped: 0 };
      this.lostFrames = 0;
      this.lastTransform = null;
      this.smoothingMatrix = null;
      util.show(util.$('#stage-empty'), false);
      util.$('#btn-start').disabled = true;
      util.$('#btn-start-camera').disabled = true;
      util.$('#btn-stop').disabled = false;
      util.$('#btn-capture').disabled = false;
      util.$('#btn-record').disabled = false;
      util.setText('#live-hint', ready
        ? 'Tracking active. Move around — the garment follows your shoulders, torso and arms.'
        : 'Pose engine unavailable: the garment cannot be tracked. See the message above.');
      if (liveState.serverRefine) this.connectServerRefine();
      this.loop();
      this.startStatusPolling();
    },

    waitForVideo(timeoutMs) {
      return new Promise((resolve) => {
        const started = performance.now();
        const check = () => {
          if (this.video.videoWidth > 0) resolve(true);
          else if (performance.now() - started > (timeoutMs || 6000)) resolve(false);
          else requestAnimationFrame(check);
        };
        check();
      });
    },

    cameraError(message) {
      util.setText('#live-hint', message);
      util.toast('Camera problem', message, 'err');
      util.show(util.$('#stage-empty'), true);
      liveState.running = false;
    },

    async stopCamera(silent) {
      liveState.running = false;
      if (this.rafId) cancelAnimationFrame(this.rafId);
      this.rafId = null;
      if (this.recorder && this.recorder.state === 'recording') this.stopRecording(true);
      if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
      this.video.srcObject = null;
      if (this.ctx) this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
      pose.resetSmoothing();
      this.lastTransform = null;
      this.disconnectServerRefine();
      if (this.sessionId) { util.safe(() => api.closeSession(this.sessionId)); this.sessionId = null; }
      util.show(util.$('#stage-empty'), true);
      util.$('#btn-start').disabled = false;
      util.$('#btn-start-camera').disabled = false;
      util.$('#btn-stop').disabled = true;
      util.$('#btn-capture').disabled = true;
      util.$('#btn-record').disabled = true;
      this.updateStatus({ pose: 'idle', garment: 'idle', transform: '—' });
      /* The garment disappears the moment the camera stops — required behaviour. */
      liveState.garmentVisible = false;
      util.dot('#hud-garment-dot', '');
      util.setText('#hud-garment', this.garmentMeta ? `${this.garmentMeta.label.slice(0, 20)} (hidden)` : 'no garment');
      if (!silent) this.stopStatusPolling();
    },

    /**
     * Reset button: drop every piece of temporal state without touching the camera.
     * Nothing here is decorative — the next frame really does re-estimate the pose from
     * scratch, the AI cache really is discarded, and the smoothed matrix really is cleared.
     */
    reset() {
      this.smoothingMatrix = null;
      this.lastTransform = null;
      this.lastWarpMode = '—';
      this.lostFrames = 0;
      this.frameTimes = [];
      this.ai.lastRequestAt = 0;
      this.ai.lastFrame = null;
      this.ai.lastFrameAt = 0;
      this.ai.lastLatency = null;
      this.ai.inflight = false;
      this.ai.error = null;
      this.ai.warned = false;
      this.ai.crossfadeUntil = 0;
      this.serverRefine.lastLatency = null;
      this.gestureController.reset();
      util.safe(() => pose.resetSmoothing());
      liveState.latency = 0;
      liveState.fps = 0;
      liveState.garmentVisible = false;
      liveState.aiState = liveState.quality === 'ai' ? 'warming up' : 'off';
      liveState.gesture = 'unknown';
      util.setText('#hud-latency', '—');
      util.setText('#hud-ai', liveState.quality === 'ai' ? 'on' : 'off');
      util.dot('#hud-ai-dot', '');
      util.setText('#st-gesture', liveState.gestureMode ? 'listening' : 'off');
      this.updateStatus({ transform: '—', garment: 'reset' });
      if (this.ctx && this.overlay) this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    },

    /* ----------------------------------------------------------------- loop */
    loop() {
      if (!liveState.running) return;
      const start = performance.now();
      const width = this.video.videoWidth || 960;
      const height = this.video.videoHeight || 720;
      util.fitCanvas(this.overlay, width, height);
      const ctx = this.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const detected = pose.detect(this.video, start);
      let landmarks = null;
      if (detected.detected) {
        landmarks = liveState.smoothing
          ? pose.smooth('body', detected.landmarks, { alpha: 0.45, motionAlpha: 0.85 })
          : detected.landmarks;
      } else {
        landmarks = liveState.smoothing ? pose.smooth('body', null, {}) : null;
        if (!landmarks) liveState.stats.dropped += 1;
      }

      liveState.personDetected = !!(landmarks && landmarks.length);
      liveState.poseBackend = pose.available ? pose.backend.replace('mediapipe_tasks_', 'blaze ') : 'unavailable';

      let garmentDrawn = false;
      if (liveState.personDetected) {
        this.lostFrames = 0;
        garmentDrawn = this.renderGarment(landmarks, width, height);
      } else {
        this.lostFrames += 1;
        if (this.lostFrames > 8) { this.lastTransform = null; }
      }
      liveState.garmentVisible = garmentDrawn;

      /* AI refinement layer (cross-faded) */
      this.drawAiLayer(ctx, width, height);

      /* skeleton debug overlay */
      if (liveState.showSkeleton && landmarks) pose.drawSkeleton(ctx, landmarks);

      if (liveState.gestureMode) this.handleGestures(detected, width, height);

      /* FPS + HUD */
      const elapsed = performance.now() - start;
      liveState.latency = elapsed;
      this.frameTimes.push(elapsed);
      if (this.frameTimes.length > 30) this.frameTimes.shift();
      const mean = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
      liveState.fps = mean > 0 ? 1000 / mean : 0;
      liveState.stats.frames += 1;
      util.setText('#hud-fps', liveState.fps.toFixed(1));
      util.setText('#hud-latency', elapsed.toFixed(0));
      util.setText('#hud-pose', liveState.personDetected ? 'locked' : 'searching');
      util.dot('#hud-pose-dot', liveState.personDetected ? 'ok' : 'warn');
      util.dot('#hud-garment-dot', garmentDrawn ? 'ok live' : (this.garmentMeta ? 'warn' : ''));

      this.rafId = requestAnimationFrame(() => this.loop());
    },

    /** Warp + composite the garment for the current frame. */
    renderGarment(landmarks, width, height) {
      if (!this.garmentCanvas || !this.garmentMeta) return false;
      const category = this.garmentMeta.category || 'unknown';
      const profile = align.profile(category);  // Settings page fit multipliers are merged in here
      const poseAnchors = align.poseAnchors(landmarks, profile);
      let transform = null;
      let mode = '—';
      if (poseAnchors) {
        const garmentAnchors = align.garmentAnchors(this.garmentAlpha);
        if (garmentAnchors) {
          const solved = align.transform(garmentAnchors, poseAnchors);
          const coverage = align.coverage(solved, this.garmentCanvas.width, this.garmentCanvas.height);
          if (coverage > 1e-5) { transform = solved; mode = 'perspective'; }
        }
      }
      if (!transform && this.lastTransform && this.lostFrames < 8) {
        transform = this.lastTransform; mode = `${this.lastWarpMode} (held)`;
      }
      if (!transform) return false;

      /* temporal smoothing of the transform itself (matrix EMA) */
      if (liveState.smoothing) {
        const alpha = Math.max(0.15, Math.min(0.85, 1 - (store.state.settings ? store.state.settings.state?.realtime?.smoothing_alpha || 0.45 : 0.45)));
        this.smoothingMatrix = this.smoothingMatrix ? align.blend(this.smoothingMatrix, transform, alpha) : transform;
        transform = this.smoothingMatrix;
      } else { this.smoothingMatrix = transform; }

      this.lastTransform = transform;
      this.lastWarpMode = mode;
      liveState.transformMode = mode;

      /* --- warp onto an offscreen layer, then punch out occluders --- */
      if (!this.garmentLayer) {
        this.garmentLayer = document.createElement('canvas');
        this.garmentLayerCtx = this.garmentLayer.getContext('2d');
      }
      util.fitCanvas(this.garmentLayer, width, height);
      const lctx = this.garmentLayerCtx;
      lctx.setTransform(1, 0, 0, 1, 0, 0);
      lctx.clearRect(0, 0, width, height);
      lctx.globalCompositeOperation = 'source-over';
      align.warp(lctx, this.garmentCanvas, this.garmentAlpha, transform, 4, 8);
      const occluders = occlusion.applyToLayer(lctx, landmarks, width, height);
      occlusion.applyHeadProtection(lctx, landmarks, width, height);

      this.ctx.globalAlpha = 1;
      this.ctx.drawImage(this.garmentLayer, 0, 0);
      liveState.occluders = occluders;
      return true;
    },

    /* ------------------------------------------------------------ AI mode */
    setQuality(mode) {
      store.setQuality(mode);
      util.$$('#seg-quality button').forEach((b) => b.classList.toggle('active', b.dataset.quality === mode));
      util.setText('#hud-ai', mode === 'ai' ? 'on' : 'off');
      if (mode === 'ai') {
        const status = store.state.status || {};
        const adapter = (status.adapter || {});
        if (!adapter.ready) {
          util.setText('#live-hint', `AI mode requested but the model is not available: ${adapter.detail || 'no checkpoint'} — staying on Fast mode.`);
          util.toast('AI model not ready', adapter.detail || 'No trained checkpoint found. Fast mode remains active — see Model Status.', 'warn');
          store.setQuality('fast');
          util.$$('#seg-quality button').forEach((b) => b.classList.toggle('active', b.dataset.quality === 'fast'));
          util.setText('#hud-ai', 'off (unavailable)');
          return;
        }
        if (this.sessionId === null) this.ensureSession();
        util.setText('#live-hint', `AI mode: requesting a diffusion frame every ${store.state.settings ? store.state.settings.tryon.live_ai_interval_ms : 700} ms. Live tracking continues in between.`);
      } else {
        util.setText('#live-hint', 'Fast mode: fully local geometric tracking, no frames leave your machine.');
      }
    },

    async ensureSession() {
      try {
        const result = await api.createSession({ garment_key: store.state.selectedGarment, enable_ai: false });
        this.sessionId = result.session_id;
        return this.sessionId;
      } catch (err) { api.report(err, 'creating live session'); return null; }
    },

    /** Ask the trained model for one photorealistic frame (throttled). */
    async maybeRequestAiFrame(width, height) {
      if (liveState.quality !== 'ai' || !this.garmentMeta || !liveState.personDetected) return;
      const settings = store.state.settings;
      const interval = settings ? settings.tryon.live_ai_interval_ms : 700;
      const now = performance.now();
      if (this.ai.inflight || (now - this.ai.lastRequestAt) < interval) return;
      this.ai.lastRequestAt = now;
      this.ai.inflight = true;

      /* send a downscaled crop — never the full-resolution frame */
      const target = util.clamp(settings ? settings.tryon.resolution : 512, 256, 768);
      const canvas = document.createElement('canvas');
      const scale = Math.min(1, target / Math.max(width, height));
      canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
      canvas.getContext('2d').drawImage(this.video, 0, 0, canvas.width, canvas.height);
      const payload = util.canvasToBase64(canvas, 0.85);

      const started = performance.now();
      try {
        const result = await api.tryOn({
          garment_key: this.garmentMeta.key,
          person_base64: payload,
          resolution: target,
        });
        if (result && result.images && result.images.output) {
          const img = await util.loadImage(result.images.output);
          this.ai.lastFrame = img;
          this.ai.lastFrameAt = performance.now();
          this.ai.lastLatency = performance.now() - started;
          this.ai.crossfadeUntil = performance.now() + Math.min(1600, interval * 2);
          this.ai.error = null;
          liveState.stats.aiFrames += 1;
          liveState.aiState = `frame ${this.ai.lastLatency.toFixed(0)} ms`;
          util.setText('#hud-ai', `on · ${this.ai.lastLatency.toFixed(0)} ms`);
        }
      } catch (err) {
        this.ai.error = err && err.message ? err.message : 'AI inference failed';
        liveState.aiState = 'error';
        util.setText('#hud-ai', 'error');
        if (!this.ai.warned) {
          api.report(err, 'AI frame');
          this.ai.warned = true;
        }
      } finally {
        this.ai.inflight = false;
      }
    },

    drawAiLayer(ctx, width, height) {
      if (!this.ai.lastFrame) return;
      const age = performance.now() - this.ai.lastFrameAt;
      const showing = performance.now() < this.ai.crossfadeUntil;
      if (!showing) return;
      const fadeIn = util.clamp((performance.now() - (this.ai.crossfadeUntil - 1600)) / 300, 0, 1);
      const fadeOut = util.clamp((this.ai.crossfadeUntil - performance.now()) / 400, 0, 1);
      ctx.save();
      ctx.globalAlpha = util.clamp(Math.min(fadeIn, fadeOut) * 0.92, 0, 1);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(this.ai.lastFrame, 0, 0, width, height);
      ctx.restore();
      liveState.aiState = `AI frame · ${(age / 1000).toFixed(1)}s old`;
      liveState.aiFrameAge = age;
    },

    /* --------------------------------------------------- server refinement */
    connectServerRefine() {
      if (this.serverRefine.ws || !this.sessionId) return;
      try {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${proto}://${location.host}/ws/live/${this.sessionId}`);
        ws.onopen = () => {
          this.serverRefine.connected = true;
          ws.send(JSON.stringify({ type: 'config', ai: liveState.quality === 'ai', garment_key: store.state.selectedGarment }));
          util.setText('#live-hint', 'Server refinement connected — frames are now refined server-side too.');
        };
        ws.onclose = () => { this.serverRefine.connected = false; };
        ws.onerror = () => { util.toast('WebSocket error', 'Server-side refinement disconnected; local tracking continues.', 'warn'); };
        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this.serverRefine.pending = false;
            if (message.type === 'result' && message.overlay) {
              this.serverRefine.lastOverlayBase64 = message.overlay;
              this.serverRefine.lastLatency = message.status ? message.status.frame_latency_ms : null;
            }
          } catch (e) { /* ignore malformed */ }
        };
        this.serverRefine.ws = ws;
      } catch (err) { console.warn('[VestiAI] websocket failed', err); }
    },

    disconnectServerRefine() {
      if (this.serverRefine.ws) { util.safe(() => this.serverRefine.ws.close()); this.serverRefine.ws = null; }
      this.serverRefine.connected = false;
    },

    sendServerFrame(width, height) {
      const ws = this.serverRefine.ws;
      if (!ws || ws.readyState !== WebSocket.OPEN || this.serverRefine.pending) return;
      const canvas = document.createElement('canvas');
      const scale = Math.min(1, 480 / Math.max(width, height));
      canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
      canvas.getContext('2d').drawImage(this.video, 0, 0, canvas.width, canvas.height);
      this.serverRefine.pending = true;
      ws.send(JSON.stringify({
        type: 'frame',
        frame: util.canvasToBase64(canvas, 0.7),
        garment_key: store.state.selectedGarment,
        ai: liveState.quality === 'ai',
        mirror: liveState.mirror,
      }));
    },

    /* ------------------------------------------------------------ gestures */
    async toggleGestureMode(on) {
      const enable = on === undefined ? !liveState.gestureMode : !!on;
      liveState.gestureMode = enable;
      util.$('#btn-gestures').classList.toggle('primary', enable);
      if (enable) {
        const ok = await pose.initHands();
        if (!ok) {
          util.toast('Gesture Mode unavailable', 'The hand model could not be loaded (frontend/vendor/models/hand_landmarker.task).', 'warn');
          liveState.gestureMode = false;
          util.$('#btn-gestures').classList.remove('primary');
          return;
        }
        util.toast('Gesture Mode on', '🖐 next outfit · ✊ previous · 👍 capture · ✌ Fast/AI · pinch to nudge the garment.', 'ok');
        util.setText('#st-gesture', 'listening');
      } else {
        util.setText('#st-gesture', 'off');
        this.gestureController.reset();
      }
    },

    handleGestures(detected, width, height) {
      void detected;
      const hands = pose.detectHands(this.video, performance.now());
      const result = this.gestureController.update(hands, width, height);
      liveState.gesture = result.gesture;
      util.setText('#st-gesture', `${result.gesture}${result.triggered ? ' → ' + result.executed_action : ''}`);
      if (result.triggered) this.runGestureAction(result.executed_action, result);
    },

    runGestureAction(action, result) {
      switch (action) {
        case 'next_outfit': {
          const g = store.cycleGarment(1);
          if (g) util.toast('Next outfit', g.label, 'ok');
          break;
        }
        case 'previous_outfit': {
          const g = store.cycleGarment(-1);
          if (g) util.toast('Previous outfit', g.label, 'ok');
          break;
        }
        case 'capture':
          this.capture(false);
          util.toast('Gesture', 'Thumbs up → photo captured.', 'ok');
          break;
        case 'toggle_quality': {
          const next = liveState.quality === 'ai' ? 'fast' : 'ai';
          this.setQuality(next);
          util.toast('Gesture', `Switched to ${next === 'ai' ? 'AI' : 'Fast'} mode.`, 'ok');
          break;
        }
        case 'move_garment':
          if (this.smoothingMatrix && result && result.move_delta) {
            /* Nudge the transform vertically while pinching (grab & drag). */
            const shift = -result.move_delta * (this.overlay.height || 720) * 0.6;
            this.smoothingMatrix[5] += shift * 0.4;
            this.smoothingMatrix[2] += 0;
          }
          break;
        default: break;
      }
    },

    /* ------------------------------------------------------------- capture */
    capture(isVideoFrame) {
      const canvas = document.createElement('canvas');
      const width = this.video.videoWidth || 960;
      const height = this.video.videoHeight || 720;
      canvas.width = width; canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.save();
      if (liveState.mirror) { ctx.translate(width, 0); ctx.scale(-1, 1); }
      ctx.drawImage(this.video, 0, 0, width, height);
      ctx.restore();
      ctx.drawImage(this.overlay, 0, 0, width, height);

      const base64 = util.canvasToBase64(canvas, 0.92, 'image/png');
      api.capturePhoto({
        image_base64: base64,
        garment_key: store.state.selectedGarment,
        backend: liveState.quality === 'ai' ? 'diffusion-live' : 'lightweight-live',
        metadata: {
          fps: Number(liveState.fps.toFixed(1)),
          latency_ms: Number(liveState.latency.toFixed(1)),
          pose_backend: liveState.poseBackend,
          transform_mode: liveState.transformMode,
          ai_state: liveState.aiState,
          gesture: liveState.gesture,
        },
      }).then((result) => {
        util.toast('Photo saved', `Stored as ${result.capture.capture_id} in captures/.`, 'ok');
        this.refreshCaptures();
      }).catch((err) => api.report(err, 'saving photo'));
      void isVideoFrame;
    },

    /* ------------------------------------------------------------ recording */
    startRecording() {
      if (!this.stream) return;
      try {
        const mimeCandidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
        const mimeType = mimeCandidates.find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
        const source = this.overlay.captureStream(30);
        /* Composite video + overlay into one recording canvas stream. */
        const mix = document.createElement('canvas');
        mix.width = this.video.videoWidth || 960; mix.height = this.video.videoHeight || 720;
        const mctx = mix.getContext('2d');
        const draw = () => {
          if (!this.recorder) return;
          mctx.save();
          if (liveState.mirror) { mctx.translate(mix.width, 0); mctx.scale(-1, 1); }
          mctx.drawImage(this.video, 0, 0, mix.width, mix.height);
          mctx.restore();
          mctx.drawImage(this.overlay, 0, 0, mix.width, mix.height);
          requestAnimationFrame(draw);
        };
        draw();
        const mixed = mix.captureStream(30);
        void source;
        this.recordedChunks = [];
        this.recorder = new MediaRecorder(mixed, mimeType ? { mimeType, videoBitsPerSecond: 4_000_000 } : undefined);
        this.recorder.ondataavailable = (event) => { if (event.data && event.data.size) this.recordedChunks.push(event.data); };
        this.recorder.onstop = () => this.uploadRecording();
        this.recorder.start(1000);
        this.recordingStartedAt = performance.now();
        util.$('#btn-record').classList.add('danger');
        util.$('#btn-record').innerHTML = '■ Stop recording';
        util.toast('Recording started', 'Your session is being recorded locally in the browser, then saved to captures/.', 'ok');
      } catch (err) {
        util.toast('Recording unavailable', `${err.message}. Try Chrome or Edge.`, 'err');
      }
    },

    stopRecording(silent) {
      if (!this.recorder) return;
      const duration = (performance.now() - this.recordingStartedAt) / 1000;
      this.recorder.stop();
      this.recorder = null;
      util.$('#btn-record').classList.remove('danger');
      util.$('#btn-record').innerHTML = '● Record';
      if (!silent) util.toast('Recording stopped', `${duration.toFixed(1)}s — uploading…`, 'info');
      this.lastRecordingDuration = duration;
    },

    uploadRecording() {
      if (!this.recordedChunks.length) return;
      const blob = new Blob(this.recordedChunks, { type: this.recordedChunks[0].type || 'video/webm' });
      const form = new FormData();
      form.append('file', blob, `vestiai-${Date.now()}.webm`);
      if (store.state.selectedGarment) form.append('garment_key', store.state.selectedGarment);
      form.append('backend', liveState.quality === 'ai' ? 'diffusion-live' : 'lightweight-live');
      form.append('duration_s', String(this.lastRecordingDuration || 0));
      api.saveVideo(form)
        .then((result) => { util.toast('Recording saved', `${result.capture.capture_id} · ${util.fmt.bytes(result.capture.size_bytes)}`, 'ok'); this.refreshCaptures(); })
        .catch((err) => api.report(err, 'uploading recording'));
      this.recordedChunks = [];
    },

    refreshCaptures() {
      api.captures().then((data) => {
        const strip = util.$('#live-captures');
        if (!strip) return;
        const photos = (data.captures || []).filter((c) => c.kind === 'photo').slice(0, 8);
        strip.innerHTML = '';
        if (!photos.length) { strip.innerHTML = '<span class="muted">No captures yet — press Capture Photo.</span>'; return; }
        photos.forEach((c) => {
          const img = util.el('img', { src: c.url, alt: c.capture_id, title: `${c.created_at_iso} · ${c.backend || ''}` });
          img.style.cursor = 'pointer';
          img.addEventListener('click', () => window.open(c.url, '_blank'));
          strip.appendChild(img);
        });
      }).catch(() => {});
    },

    /* --------------------------------------------------------------- status */
    updateStatus(map) {
      if (map.pose !== undefined) util.setText('#st-pose', map.pose);
      if (map.segmentation !== undefined) util.setText('#st-seg', map.segmentation);
      if (map.garment !== undefined) util.setText('#st-track', map.garment);
      if (map.transform !== undefined) util.setText('#st-transform', map.transform);
    },

    statusTick() {
      util.setText('#st-pose', liveState.personDetected ? `locked (${liveState.poseBackend})` : (pose.available ? 'searching' : 'unavailable'));
      util.dot('#st-pose-dot', liveState.personDetected ? 'ok' : (pose.available ? 'warn' : 'err'));
      util.setText('#st-seg', pose.available ? 'mediapipe mask' : 'off');
      util.setText('#st-track', this.garmentMeta ? (liveState.garmentVisible ? 'tracking' : 'hidden') : 'no garment');
      util.setText('#st-transform', liveState.transformMode || '—');
      util.setText('#st-latency', `${liveState.latency.toFixed(0)} ms · ${liveState.fps.toFixed(1)} fps`);
      util.setText('#st-device', (store.state.devices && store.state.devices.label) || '—');
      const adapter = (store.state.status && store.state.status.adapter) || {};
      util.setText('#st-model', adapter.display_name ? `${adapter.display_name}${adapter.ready ? '' : ' (not ready)'}` : '—');
      util.setText('#hud-gpu', store.state.devices && store.state.devices.device && store.state.devices.device.cuda_available
        ? (store.state.devices.device.device_names[0] || 'CUDA').slice(0, 18) : 'CPU');
    },

    startStatusPolling() {
      if (this.statusTimer) return;
      this.statusTimer = setInterval(() => {
        if (!liveState.running) return;
        this.statusTick();
        const width = this.video.videoWidth; const height = this.video.videoHeight;
        if (!width) return;
        if (liveState.quality === 'ai') this.maybeRequestAiFrame(width, height);
        if (liveState.serverRefine && this.serverRefine.connected) this.sendServerFrame(width, height);
      }, 250);
    },

    stopStatusPolling() {
      if (this.statusTimer) { clearInterval(this.statusTimer); this.statusTimer = null; }
    },

    /* -------------------------------------------------------- full try-on */
    /** High-quality single-image try-on from the current frame (server, saved to results/). */
    async runFullTryOn() {
      if (!this.garmentMeta) { util.toast('No garment', 'Select a garment first.', 'warn'); return; }
      const canvas = document.createElement('canvas');
      const width = this.video.videoWidth || 960; const height = this.video.videoHeight || 720;
      canvas.width = width; canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.save();
      if (liveState.mirror) { ctx.translate(width, 0); ctx.scale(-1, 1); }
      ctx.drawImage(this.video, 0, 0, width, height);
      ctx.restore();
      util.toast('Generating…', 'Running the trained model on this frame.', 'info');
      try {
        const result = await api.tryOn({
          garment_key: this.garmentMeta.key,
          person_base64: util.canvasToBase64(canvas, 0.92),
        });
        util.toast('Try-on ready', `Saved to results/ (${result.latency_ms.toFixed(0)} ms, ${result.backend}).`, 'ok');
        window.__vestiaiLastTryOn = result;
        if (result.images && result.images.output) {
          const modal = util.$('#upload-modal');
          void modal;
          window.open(result.images.output, '_blank');
        }
      } catch (err) { api.report(err, 'full try-on'); }
    },
  };

  window.VestiAI.live = live;
})();
