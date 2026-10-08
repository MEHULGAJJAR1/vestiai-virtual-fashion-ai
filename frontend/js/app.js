/* =====================================================================================
 * VestiAI — application shell: routing, wiring, boot
 * ===================================================================================== */
(function () {
  'use strict';
  const { util, api, store, live, closet, outfits, training, status } = window.VestiAI;

  const VIEWS = {
    home: { title: 'Home', init: null },
    live: { title: 'Live Try-On', init: null },
    closet: { title: 'My Closet', init: () => closet.refresh() },
    outfits: { title: 'Outfit Builder', init: () => outfits.load() },
    training: { title: 'AI Training', init: () => training.refresh() },
    status: { title: 'Model Status', init: () => status.refresh() },
    settings: { title: 'Settings', init: () => status.refresh() },
    about: { title: 'About', init: () => app.loadAbout() },
  };

  const app = {
    /* ------------------------------------------------------------------ routing */
    go(view) {
      if (!VIEWS[view]) view = 'home';
      store.state.view = view;
      util.$$('.view').forEach((node) => util.show(node, node.id === `view-${view}`));
      util.$$('.nav-item').forEach((button) => button.classList.toggle('active', button.dataset.view === view));
      util.setText('#view-title', VIEWS[view].title);
      util.$('#sidebar').classList.remove('open');
      if (location.hash !== `#${view}`) history.replaceState(null, '', `#${view}`);
      if (view === 'live') live.refreshCaptures();
      if (VIEWS[view].init) util.safe(VIEWS[view].init);
    },

    /* ------------------------------------------------------------------ top bar */
    renderTopBar(statusData, devices) {
      const adapter = (statusData && statusData.adapter) || {};
      const device = (devices && devices.device) || (statusData && statusData.devices) || {};
      const cps = (statusData && statusData.checkpoints) || {};

      const deviceKind = device.cuda_available ? 'ok' : (device.mps_available ? 'info' : 'warn');
      util.statusBadge(util.$('#badge-device'), deviceKind, device.resolved_device_label || device.resolved_device || 'cpu');
      util.statusBadge(util.$('#badge-backend'), adapter.ready ? 'ok' : 'warn', adapter.display_name || adapter.name || 'no model');
      util.statusBadge(util.$('#badge-checkpoint'), cps.ready_for_inference ? 'ok' : 'warn',
        cps.ready_for_inference ? (cps.inference_path || '').split('/').slice(-1)[0] : 'no checkpoint');

      util.dot('#nav-status-dot', adapter.ready ? 'ok' : 'warn');
      util.setText('#nav-status-text', adapter.ready ? (adapter.name === 'diffusion' ? 'AI model ready' : 'fast mode ready') : 'fast mode only');
      util.setText('#nav-version', (statusData && statusData.devices && statusData.app_version) || '1.0.0');
    },

    renderHome(statusData, devices) {
      const host = util.$('#home-status-list');
      host.innerHTML = '';
      const adapter = (statusData && statusData.adapter) || {};
      const device = (devices && devices.device) || (statusData && statusData.devices) || {};
      const ml = (statusData && statusData.ml_stack) || {};
      const rows = [
        ['Real-time pipeline', 'available', 'ok'],
        ['Pose backend (browser)', window.VestiAI.pose.available ? window.VestiAI.pose.backend : 'will load with the camera', window.VestiAI.pose.available ? 'ok' : ''],
        ['Device', device.resolved_device_label || device.resolved_device || 'cpu', device.cuda_available ? 'ok' : 'warn'],
        ['ML stack (torch/diffusers)', ml.ready ? `${ml.torch_version || ''} ${ml.diffusers_version || ''}`.trim() : 'not installed', ml.ready ? 'ok' : 'warn'],
        ['AI try-on model', adapter.ready ? adapter.display_name : 'no checkpoint', adapter.ready ? 'ok' : 'warn'],
        ['Training', ml.ready ? 'available' : 'needs the ML extra', ml.ready ? 'ok' : 'warn'],
      ];
      rows.forEach(([name, value, kind]) => {
        host.appendChild(util.el('div', { class: 'status-row' }, [
          util.el('span', { class: `dot ${kind}` }),
          util.el('span', { class: 'name', text: name }),
          util.el('span', { class: 'spacer' }),
          util.el('span', { class: 'val', text: String(value) }),
        ]));
      });

      const warnHost = util.$('#home-warnings');
      warnHost.innerHTML = '';
      if (statusData && statusData.vram_warning) {
        warnHost.appendChild(util.el('div', { class: 'notice warn', text: statusData.vram_warning }));
      }
      if (!ml.ready) {
        warnHost.appendChild(util.el('div', { class: 'notice info', style: { marginTop: '8px' },
          text: 'Live tracking works right now. To unlock AI synthesis and training: python scripts/setup.py --profile ml' }));
      }
      util.setText('#home-stat-backend', adapter.name === 'diffusion' ? 'diffusion (trained)' : (adapter.name === 'lightweight' ? 'geometric warp' : adapter.name || '—'));
      const liveStats = store.state.live;
      util.setText('#home-stat-fps', liveStats.running ? `${liveStats.fps ? liveStats.fps.toFixed(0) : '—'} fps` : 'idle');
    },

    /** Dedicated overlay so showing a result never destroys the upload modal. */
    resultOverlay() {
      let overlay = util.$('#result-modal');
      if (!overlay) {
        overlay = util.el('div', { id: 'result-modal', class: 'modal-backdrop hidden' }, [util.el('div', { class: 'modal' })]);
        document.body.appendChild(overlay);
        overlay.addEventListener('click', (event) => { if (event.target === overlay) util.show(overlay, false); });
      }
      return overlay;
    },

    showTryOnResult(result, garment) {
      const overlay = this.resultOverlay();
      const body = overlay.querySelector('.modal');
      const label = garment ? (garment.label || garment.category || garment) : '—';
      const images = result.images || {};
      body.innerHTML = `
        <div class="card-head"><h2>Try-on result</h2><div class="spacer"></div>
          <span class="badge">${result.backend}</span>
          <span class="badge">${util.fmt.ms(result.latency_ms)}</span>
          <button class="btn sm" id="close-result">✕</button></div>
        <div class="grid cols-2">
          <div><img src="${images.output}" style="width:100%;border-radius:12px" alt="try-on output" /></div>
          <div>
            ${images.comparison ? `<h3>Comparison</h3><img src="${images.comparison}" style="width:100%;border-radius:12px" alt="comparison" />` : ''}
            <h3 style="margin-top:12px">Details</h3>
            <div class="status-list">
              <div class="status-row"><span class="name">Garment</span><span class="spacer"></span><span class="val">${label}</span></div>
              <div class="status-row"><span class="name">Backend</span><span class="spacer"></span><span class="val">${result.backend}</span></div>
              <div class="status-row"><span class="name">Latency</span><span class="spacer"></span><span class="val">${util.fmt.ms(result.latency_ms)}</span></div>
              ${result.result ? `<div class="status-row"><span class="name">Saved as</span><span class="spacer"></span><span class="val">${result.result.result_id}</span></div>` : ''}
            </div>
            ${(result.warnings || []).map((w) => `<div class="notice warn" style="margin-top:10px">${w}</div>`).join('')}
          </div>
        </div>`;
      util.show(overlay, true);
      body.querySelector('#close-result').addEventListener('click', () => util.show(overlay, false));
    },

    async loadAbout() {
      try {
        const about = await api.about();
        util.setText('#about-tagline', about.tagline || '');
        const pipelines = util.$('#about-pipelines');
        pipelines.innerHTML = '';
        (about.pipelines || []).forEach((pipeline) => {
          pipelines.appendChild(util.el('div', { class: 'card tight', style: { marginBottom: '10px' } }, [
            util.el('b', { text: pipeline.name }),
            util.el('div', { class: 'field-hint', text: pipeline.latency }),
            util.el('p', { style: { margin: '6px 0 0', fontSize: '.86rem' }, text: pipeline.details }),
          ]));
        });
        const honesty = util.$('#about-honesty');
        honesty.innerHTML = '';
        (about.honesty || []).forEach((line) => honesty.appendChild(util.el('div', { class: 'notice info', style: { marginBottom: '8px' }, text: line })));
      } catch (err) { /* non-fatal */ }

      try {
        const result = await api.get('/api/openapi/lite');
        util.setText('#about-endpoints', (result.endpoints || []).join('\n'));
      } catch (err) {
        util.setText('#about-endpoints', 'GET  /api/health\nGET  /api/status\nPOST /api/garments/upload\nPOST /api/tryon\nPOST /api/live/frame\nPOST /api/training/start\nGET  /api/training/status\n...  see /docs for the full list');
      }
    },

    /* ------------------------------------------------------------------- wiring */
    wire() {
      util.$$('.nav-item').forEach((button) => button.addEventListener('click', () => app.go(button.dataset.view)));
      util.$$('[data-go]').forEach((button) => button.addEventListener('click', () => app.go(button.dataset.go)));
      util.$('#menu-btn').addEventListener('click', () => util.$('#sidebar').classList.toggle('open'));
      util.$('#btn-refresh-top').addEventListener('click', () => status.refresh(true));
      util.$('#home-sample-closet').addEventListener('click', () => closet.generateSamples());

      /* camera controls */
      util.$('#btn-start').addEventListener('click', () => live.startCamera());
      util.$('#btn-start-camera').addEventListener('click', () => live.startCamera());
      util.$('#btn-stop').addEventListener('click', () => live.stopCamera());
      util.$('#btn-capture').addEventListener('click', () => live.capture());
      util.$('#btn-record').addEventListener('click', () => {
        if (live.recorder && live.recorder.state === 'recording') live.stopRecording();
        else live.startRecording();
      });
      util.$('#btn-mirror').addEventListener('click', () => {
        store.setMirror(!store.state.live.mirror);
        live.video.classList.toggle('mirror-off', !store.state.live.mirror);
        util.toast('Mirror', store.state.live.mirror ? 'Mirrored view on (natural for a webcam).' : 'Mirror off.', 'info');
      });
      util.$('#btn-fullscreen').addEventListener('click', () => {
        const stage = util.$('#stage-wrap');
        if (stage.requestFullscreen) {
          if (document.fullscreenElement) document.exitFullscreen();
          else stage.requestFullscreen().catch(() => util.toast('Fullscreen blocked', 'Your browser refused fullscreen; try clicking the page first.', 'warn'));
        }
      });
      util.$('#btn-next-garment').addEventListener('click', () => {
        const g = store.cycleGarment(1);
        if (g) util.toast('Outfit changed', g.label, 'ok');
        else util.toast('No garments', 'Upload or generate garments first.', 'warn');
      });
      util.$('#btn-remove-garment').addEventListener('click', () => {
        store.setSelectedGarment(null);
        util.toast('Garment removed', 'Live tracking continues without a garment.', 'ok');
      });
      util.$('#btn-gestures').addEventListener('click', () => live.toggleGestureMode());
      util.$('#btn-reset').addEventListener('click', () => {
        live.reset();
        util.toast('Reset', 'Tracking state, smoothing and AI cache cleared.', 'ok');
      });
      util.$('#btn-upload-live').addEventListener('click', () => util.show(util.$('#upload-modal'), true));
      util.$('#btn-refresh-closet-live').addEventListener('click', () => closet.refresh());

      util.$$('#seg-quality button').forEach((button) => button.addEventListener('click', () => live.setQuality(button.dataset.quality)));
      util.$('#opt-skeleton').addEventListener('change', (e) => store.setSkeleton(e.target.checked));
      util.$('#opt-smooth').addEventListener('change', (e) => store.setSmoothing(e.target.checked));
      util.$('#opt-server').addEventListener('change', (e) => {
        store.state.live.serverRefine = e.target.checked;
        if (e.target.checked) {
          live.ensureSession().then(() => live.connectServerRefine());
          util.toast('Server refinement on', 'Frames are also composited on the server (uses the trained model when AI mode is on).', 'info');
        } else {
          live.disconnectServerRefine();
        }
      });
      ['#live-interval', '#live-smoothing'].forEach((sel) => util.$(sel).addEventListener('input', () => {
        util.setText('#lbl-interval', util.$('#live-interval').value);
        util.setText('#lbl-smooth', util.$('#live-smoothing').value);
      }));
      util.$('#btn-apply-live').addEventListener('click', async () => {
        try {
          await api.setLiveConfig({
            ai_interval_ms: Number(util.$('#live-interval').value),
            smoothing_alpha: Number(util.$('#live-smoothing').value),
            resolution: Number(util.$('#live-resolution').value),
          });
          if (store.state.settings) {
            store.state.settings.tryon.live_ai_interval_ms = Number(util.$('#live-interval').value);
            store.state.settings.realtime.smoothing_alpha = Number(util.$('#live-smoothing').value);
          }
          util.toast('Live settings applied', `AI every ${util.$('#live-interval').value} ms · smoothing ${util.$('#live-smoothing').value}`, 'ok');
        } catch (err) { api.report(err, 'live config'); }
      });

      /* upload modal */
      const modal = util.$('#upload-modal');
      util.$('#upload-modal-close').addEventListener('click', () => util.show(modal, false));
      modal.addEventListener('click', (event) => { if (event.target === modal) util.show(modal, false); });
      const modalFile = util.$('#modal-file');
      util.$('#modal-dropzone').addEventListener('click', () => modalFile.click());
      let modalChosen = null;
      modalFile.addEventListener('change', async () => {
        modalChosen = modalFile.files[0];
        if (!modalChosen) return;
        util.$('#modal-preview').innerHTML = `<img src="${URL.createObjectURL(modalChosen)}" style="max-width:100%;max-height:260px;border-radius:12px" alt="preview" />`;
        util.$('#modal-preview-btn').disabled = false;
        util.$('#modal-upload-btn').disabled = false;
      });
      const buildForm = () => {
        const form = new FormData();
        form.append('file', modalChosen);
        const category = util.$('#modal-category').value;
        if (category) form.append('category', category);
        if (util.$('#modal-force').checked) form.append('skip_quality_gate', 'true');
        return form;
      };
      util.$('#modal-preview-btn').addEventListener('click', async () => {
        if (!modalChosen) return;
        const host = util.$('#modal-report');
        host.innerHTML = '<span class="muted">Analysing…</span>';
        try {
          const result = await api.previewGarment(buildForm());
          const quality = result.quality || {};
          host.innerHTML = '';
          host.appendChild(util.el('div', { class: `notice ${quality.ok ? 'ok' : 'err'}`,
            text: quality.ok ? 'This image will be accepted.' : `Would be rejected: ${(quality.issues || []).join(' ')}` }));
          host.appendChild(util.el('div', { class: 'row', style: { marginTop: '8px' } }, [
            util.el('span', { class: 'badge', text: result.classification.category_label || util.fmt.title(result.classification.category) }),
            util.el('span', { class: 'badge', text: `${result.classification.method} ${util.fmt.pct(result.classification.confidence)}` }),
          ]));
          host.appendChild(util.el('img', { src: result.preview_data_uri, style: { maxWidth: '100%', marginTop: '10px', borderRadius: '10px' }, alt: 'cut-out preview' }));
        } catch (err) { api.report(err, 'previewing garment'); host.innerHTML = `<div class="notice err">${err.message}</div>`; }
      });
      util.$('#modal-upload-btn').addEventListener('click', async () => {
        if (!modalChosen) return;
        try {
          const result = await api.uploadGarment(buildForm());
          await closet.refresh();
          store.setSelectedGarment(result.garment.key);
          util.show(modal, false);
          util.toast('Garment added', `${result.garment.label} (${result.garment.category}) is ready to wear.`, 'ok');
        } catch (err) { api.report(err, 'uploading garment'); }
      });

      /* full keyboard shortcuts */
      document.addEventListener('keydown', (event) => {
        if (event.target.matches('input, textarea, select')) return;
        if (store.state.view !== 'live') return;
        switch (event.key) {
          case 'c': util.$('#btn-capture').click(); break;
          case 'g': util.$('#btn-gestures').click(); break;
          case 'n': util.$('#btn-next-garment').click(); break;
          case 'f': util.$('#btn-fullscreen').click(); break;
          case 'a': live.setQuality(store.state.live.quality === 'ai' ? 'fast' : 'ai'); break;
          case 'Escape': if (document.fullscreenElement) document.exitFullscreen(); break;
          default: break;
        }
      });
    },

    /* --------------------------------------------------------------------- boot */
    async boot() {
      this.wire();
      closet.init();
      outfits.init();
      training.init();
      status.init();
      live.init();

      /* optimistic UI restore from localStorage */
      util.$$('#seg-quality button').forEach((b) => b.classList.toggle('active', b.dataset.quality === store.state.live.quality));
      util.setText('#hud-ai', store.state.live.quality === 'ai' ? 'on' : 'off');
      util.$('#opt-skeleton').checked = store.state.live.showSkeleton;
      util.$('#opt-smooth').checked = store.state.live.smoothing;

      const initial = (location.hash || '#home').slice(1);
      this.go(VIEWS[initial] ? initial : 'home');

      try {
        await status.refresh();
        await closet.refresh();
        this.renderHome(store.state.status, store.state.devices);
        if (store.state.selectedGarment) live.loadGarment();
        live.refreshCaptures();
      } catch (err) {
        api.report(err, 'initial load');
        util.statusBadge(util.$('#badge-device'), 'err', 'offline');
      }

      /* Periodic health beacon so the top bar always reflects reality. */
      setInterval(async () => {
        try {
          const data = await api.status();
          store.state.status = data;
          this.renderTopBar(data, store.state.devices);
        } catch (e) { util.statusBadge(util.$('#badge-device'), 'err', 'offline'); }
      }, 20000);

      window.addEventListener('beforeunload', () => { util.safe(() => live.stopCamera(true)); });
      console.log('%cVestiAI', 'font-weight:bold;color:#7c5cff', 'ready — press ? for shortcuts, docs at /docs');
    },
  };

  window.VestiAI.app = app;
  document.addEventListener('DOMContentLoaded', () => app.boot());
})();
