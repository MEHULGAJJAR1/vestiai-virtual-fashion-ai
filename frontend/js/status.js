/* =====================================================================================
 * VestiAI — Model Status + Settings pages
 * ===================================================================================== */
(function () {
  'use strict';
  const { util, api, store } = window.VestiAI;

  const status = {
    async init() {
      util.$('#btn-reload-backend').addEventListener('click', () => this.reloadBackend());
      util.$$('[data-backend]').forEach((button) => {
        button.addEventListener('click', () => this.setBackend(button.dataset.backend));
      });
      util.$('#btn-clear-cache').addEventListener('click', () => this.clearCache());
      util.$('#btn-save-settings').addEventListener('click', () => this.saveSettings());
      util.$('#btn-reset-settings').addEventListener('click', () => this.resetSettings());
      util.$('#btn-export-settings').addEventListener('click', () => this.exportSettings());
      ['#set-steps', '#set-guidance', '#set-smoothing', '#set-interval'].forEach((sel) => {
        const node = util.$(sel);
        if (node) node.addEventListener('input', () => this.syncLabels());
      });
    },

    syncLabels() {
      util.setText('#lbl-steps', util.$('#set-steps').value);
      util.setText('#lbl-guidance', Number(util.$('#set-guidance').value).toFixed(1));
      util.setText('#lbl-smoothing2', util.$('#set-smoothing').value);
      util.setText('#lbl-interval2', util.$('#set-interval').value);
    },

    async refresh(report) {
      try {
        const [statusData, devices, settings] = await Promise.all([api.status(), api.devices(), api.settings()]);
        store.state.status = statusData;
        store.state.devices = devices;
        store.state.settings = settings;
        this.render(statusData, devices);
        this.renderSettings(settings);
        window.VestiAI.app.renderTopBar(statusData, devices);
        if (report) util.toast('Status refreshed', statusData.summary ? statusData.summary.message : 'ok', 'ok');
      } catch (err) { api.report(err, 'status'); }
    },

    render(data, devices) {
      /* ---------------- devices */
      const host = util.$('#status-device');
      const device = data.devices || {};
      const vram = device.vram_total_gb && device.vram_total_gb.length ? `${device.vram_total_gb[0]} GB` : 'n/a';
      const rows = [
        ['Resolved device', device.resolved_device_label || device.resolved_device || '—'],
        ['PyTorch', device.torch_version || 'not installed'],
        ['CUDA', device.cuda_available ? `yes (${device.cuda_version || '?'})` : 'no'],
        ['GPU names', (device.device_names || []).join(', ') || '—'],
        ['Total VRAM', vram],
        ['FP16 / BF16', `${device.supports_fp16 ? 'yes' : 'no'} / ${device.supports_bf16 ? 'yes' : 'no'}`],
        ['Apple MPS', device.mps_available ? 'available' : 'no'],
        ['CPU threads', device.cpu_count || '—'],
        ['Python', device.python_version || '—'],
        ['Platform', device.platform || '—'],
        ['Recommended precision', data.recommended_precision || '—'],
      ];
      host.innerHTML = '';
      const table = util.el('table', {}, [util.el('tbody', {}, rows.map(([k, v]) => util.el('tr', {}, [
        util.el('td', { text: k }), util.el('td', { class: 'mono', text: String(v) }),
      ])))]);
      host.appendChild(table);
      if (data.gpu_live && data.gpu_live.gpus && data.gpu_live.gpus.length) {
        const gpu = data.gpu_live.gpus[0];
        host.appendChild(util.el('div', { class: 'notice info', style: { marginTop: '10px' },
          text: `Live GPU: ${gpu.gpu_util_pct || 0}% utilisation · ${Math.round(gpu.vram_used_mb || 0)} / ${Math.round(gpu.vram_total_mb || 0)} MB · ${gpu.temperature_c || '?'} °C` }));
      }
      (device.notes || []).forEach((note) => host.appendChild(util.el('div', { class: 'notice warn', style: { marginTop: '8px' }, text: note })));
      if (data.vram_warning) host.appendChild(util.el('div', { class: 'notice warn', style: { marginTop: '8px' }, text: data.vram_warning }));

      /* ---------------- adapter */
      const adapter = data.adapter || {};
      const ahost = util.$('#status-adapter');
      ahost.innerHTML = '';
      ahost.appendChild(util.el('div', { class: `notice ${adapter.ready ? 'ok' : 'warn'}` }, [
        util.el('b', { text: `${adapter.display_name || adapter.name || 'unknown'} — ${adapter.ready ? 'ready' : 'not ready'}` }),
        util.el('div', { style: { marginTop: '6px' }, text: adapter.detail || '' }),
      ]));
      if (adapter.hints && adapter.hints.length) {
        ahost.appendChild(util.el('pre', { style: { marginTop: '10px' }, text: adapter.hints.join('\n') }));
      }
      if (adapter.checkpoint_info) {
        const info = adapter.checkpoint_info;
        ahost.appendChild(util.el('div', { class: 'status-list', style: { marginTop: '10px' } }, [
          this.row('Checkpoint', info.path),
          this.row('Valid', info.valid ? 'yes' : `no — ${info.reason || ''}`),
          this.row('Size', `${info.size_mb} MB`),
          this.row('Weights', (info.weights || []).join(', ') || '—'),
        ]));
      }
      const cps = data.checkpoints || {};
      ahost.appendChild(util.el('div', { class: 'status-list', style: { marginTop: '10px' } }, [
        this.row('Checkpoints found', String(cps.count || 0)),
        this.row('Inference path', cps.inference_path || 'none — fast mode only'),
      ]));

      /* ---------------- components */
      api.components().then((result) => {
        const list = result.components || [];
        const chost = util.$('#status-components');
        chost.innerHTML = '';
        list.forEach((item) => {
          chost.appendChild(util.el('div', { class: 'status-row' }, [
            util.el('span', { class: `dot ${item.installed ? 'ok' : (item.required ? 'err' : '')}` }),
            util.el('span', { class: 'name', text: item.name }),
            util.el('span', { class: 'badge', text: item.required ? 'required' : 'optional' }),
            util.el('span', { class: 'spacer' }),
            util.el('span', { class: 'val', text: item.installed ? item.version : item.install }),
          ]));
        });
        util.setText('#status-deps-badge', `${list.filter((i) => i.installed).length}/${list.length} installed`);
      }).catch(() => {});

      /* ---------------- checkpoints + disk */
      const cphost = util.$('#status-checkpoints');
      const records = [cps.best, cps.latest].filter(Boolean).concat(cps.epochs || []);
      cphost.innerHTML = records.length ? '' : '<span class="muted">No checkpoints yet.</span>';
      records.forEach((cp) => {
        cphost.appendChild(util.el('div', { class: 'status-row' }, [
          util.el('span', { class: 'badge', text: cp.kind }),
          util.el('span', { class: 'name', text: cp.name }),
          util.el('span', { class: 'spacer' }),
          util.el('span', { class: 'val', text: cp.metric !== null && cp.metric !== undefined ? `val ${Number(cp.metric).toFixed(4)}` : '—' }),
          util.el('span', { class: 'val', text: `${cp.size_mb} MB` }),
        ]));
      });

      api.disk().then((disk) => {
        const dhost = util.$('#status-disk');
        dhost.innerHTML = '';
        Object.entries(disk).forEach(([name, info]) => {
          if (name === 'volume') return;
          dhost.appendChild(util.el('div', { class: 'status-row' }, [
            util.el('span', { class: 'name', text: util.fmt.title(name.replace('_dir', '')) }),
            util.el('span', { class: 'spacer' }),
            util.el('span', { class: 'val', text: `${info.size_mb} MB · ${info.files} files` }),
          ]));
        });
        if (disk.volume) {
          dhost.appendChild(util.el('div', { class: 'notice info', style: { marginTop: '10px' },
            text: `Disk: ${disk.volume.free_gb} GB free of ${disk.volume.total_gb} GB` }));
        }
      }).catch(() => {});
    },

    row(name, value) {
      return util.el('div', { class: 'status-row' }, [
        util.el('span', { class: 'name', text: name }),
        util.el('span', { class: 'spacer' }),
        util.el('span', { class: 'val', text: value === null || value === undefined ? '—' : String(value) }),
      ]);
    },

    renderSettings(settings) {
      if (!settings) return;
      const tryon = settings.tryon || {};
      const realtime = settings.realtime || {};
      const training = settings.training || {};
      util.$('#set-backend').value = tryon.backend || 'auto';
      util.$('#set-resolution').value = String(tryon.resolution || 512);
      util.$('#set-steps').value = String(tryon.num_inference_steps || 30);
      util.$('#set-guidance').value = String(tryon.guidance_scale || 2);
      util.$('#set-seed').value = String(tryon.seed === undefined ? 42 : tryon.seed);
      util.$('#set-cpu-diffusion').checked = !!tryon.enable_cpu_diffusion;
      util.$('#set-smoothing').value = String(realtime.smoothing_alpha === undefined ? 0.45 : realtime.smoothing_alpha);
      util.$('#set-interval').value = String(tryon.live_ai_interval_ms || 700);
      util.$('#set-bg').value = (settings.garment && settings.garment.background_removal) || 'auto';
      util.$('#set-segmentation').checked = realtime.segmentation !== false;
      util.$('#set-force-cpu').checked = !!(settings.device && settings.device.force_cpu);
      util.$('#set-batch').value = String(training.batch_size || 2);
      util.$('#set-epochs').value = String(training.num_epochs || 10);
      util.$('#set-lr').value = String(training.learning_rate || 0.00001);
      util.$('#set-accum').value = String(training.gradient_accumulation || 4);
      util.$('#set-precision').value = training.mixed_precision || 'fp16';
      util.$('#set-tracker').value = training.tracker || 'tensorboard';
      this.syncLabels();

      const paths = util.$('#set-paths');
      paths.innerHTML = '';
      Object.entries(settings.paths || {}).forEach(([name, value]) => {
        paths.appendChild(this.row(util.fmt.title(name), value));
      });
      paths.appendChild(util.el('div', { style: { height: '1px', background: 'var(--border)', margin: '10px 0' } }));
      Object.entries(settings.runtime || {}).forEach(([name, value]) => {
        paths.appendChild(this.row(name, value || 'not installed'));
      });

      /* per-category fit tuning — a real client-side warp multiplier (stored in localStorage) */
      const tuning = util.$('#set-fit-tuning');
      tuning.innerHTML = '';
      const categories = ['t-shirt', 'shirt', 'jacket', 'kurta', 'dress', 'top', 'sweater', 'traditional', 'formal', 'bottom', 'shoes', 'accessory'];
      const table = window.VestiAI.align.getOverrides();
      categories.forEach((category) => {
        const base = window.VestiAI.align.PROFILES[category];
        const current = Object.assign({ widthFactor: base.widthFactor, lengthFactor: base.lengthFactor }, table[category] || {});
        const commit = () => {
          window.VestiAI.align.setOverrides(Object.assign({}, window.VestiAI.align.getOverrides(), {
            [category]: { widthFactor: Number(current.widthFactor.toFixed(3)), lengthFactor: Number(current.lengthFactor.toFixed(3)) },
          }));
        };
        const summary = util.el('span', { class: 'mono', text: `${util.fmt.title(category)} · W ×${current.widthFactor.toFixed(2)} · L ×${current.lengthFactor.toFixed(2)}` });
        tuning.appendChild(util.el('div', { class: 'field' }, [
          summary,
          util.el('input', {
            type: 'range', min: '0.6', max: '1.6', step: '0.02', value: String(current.widthFactor), title: 'width',
            oninput: (event) => {
              current.widthFactor = Number(event.target.value); commit();
              summary.textContent = `${util.fmt.title(category)} · W ×${current.widthFactor.toFixed(2)} · L ×${current.lengthFactor.toFixed(2)}`;
            },
          }),
          util.el('input', {
            type: 'range', min: '0.6', max: '3.0', step: '0.02', value: String(current.lengthFactor), title: 'length',
            oninput: (event) => {
              current.lengthFactor = Number(event.target.value); commit();
              summary.textContent = `${util.fmt.title(category)} · W ×${current.widthFactor.toFixed(2)} · L ×${current.lengthFactor.toFixed(2)}`;
            },
          }),
        ]));
      });
      tuning.appendChild(util.el('p', { class: 'field-hint', style: { gridColumn: '1 / -1' },
        text: 'Left slider = width, right slider = length. These multipliers change how much of your body the warp covers; they are stored locally in your browser and applied to the next frame. "Reset" restores the tuned defaults.' }));
    },

    async saveSettings() {
      const payload = {
        tryon_backend: util.$('#set-backend').value,
        resolution: Number(util.$('#set-resolution').value),
        num_inference_steps: Number(util.$('#set-steps').value),
        guidance_scale: Number(util.$('#set-guidance').value),
        seed: Number(util.$('#set-seed').value),
        enable_cpu_diffusion: util.$('#set-cpu-diffusion').checked,
        smoothing_alpha: Number(util.$('#set-smoothing').value),
        live_ai_interval_ms: Number(util.$('#set-interval').value),
        background_removal: util.$('#set-bg').value,
        segmentation_enabled: util.$('#set-segmentation').checked,
        force_cpu: util.$('#set-force-cpu').checked,
        batch_size: Number(util.$('#set-batch').value),
        num_epochs: Number(util.$('#set-epochs').value),
        learning_rate: Number(util.$('#set-lr').value),
        gradient_accumulation: Number(util.$('#set-accum').value),
        mixed_precision: util.$('#set-precision').value,
        tracker: util.$('#set-tracker').value,
        category_overrides: (store.state.settings && store.state.settings.fitOverrides) || null,
      };
      try {
        const result = await api.saveSettings(payload);
        util.setText('#set-status', `saved ${new Date().toLocaleTimeString()} · ${Object.keys(result.applied).length} fields`);
        util.toast('Settings saved', 'Live sessions pick up tracking changes immediately.', 'ok');
        await this.refresh();
      } catch (err) { api.report(err, 'saving settings'); }
    },

    /**
     * Reset = put every control back to the values the server is currently running with and
     * drop the local-only fit multipliers. It never invents values: it re-reads them.
     */
    async resetSettings() {
      window.VestiAI.align.setOverrides({});
      try {
        const settings = await api.settings();
        store.state.settings = settings;
        this.renderSettings(settings);
        util.setText('#set-status', 'reset to the running configuration');
        util.toast('Settings reset', 'Controls restored to the server configuration; fit multipliers cleared.', 'ok');
      } catch (err) { api.report(err, 'resetting settings'); }
    },

    exportSettings() {
      const data = store.state.settings || {};
      const yaml = [
        '# VestiAI runtime settings (generated from the Settings page)',
        `resolution: ${(data.tryon || {}).resolution}`,
        `num_inference_steps: ${(data.tryon || {}).num_inference_steps}`,
        `guidance_scale: ${(data.tryon || {}).guidance_scale}`,
        `live_ai_interval_ms: ${(data.tryon || {}).live_ai_interval_ms}`,
        `smoothing_alpha: ${(data.realtime || {}).smoothing_alpha}`,
        `batch_size: ${(data.training || {}).batch_size}`,
        `num_epochs: ${(data.training || {}).num_epochs}`,
        `learning_rate: ${(data.training || {}).learning_rate}`,
        `mixed_precision: ${(data.training || {}).mixed_precision}`,
        `tracker: ${(data.training || {}).tracker}`,
      ].join('\n');
      util.download('vestiai-settings.yaml', yaml, 'text/yaml');
    },

    async reloadBackend() {
      try {
        const result = await api.reloadModel();
        util.toast('Backend reloaded', result.detail || 'Checkpoints re-scanned.', result.ready ? 'ok' : 'warn');
        await this.refresh();
      } catch (err) { api.report(err, 'reloading model'); }
    },

    async setBackend(name) {
      try {
        const result = await api.setBackend(name);
        util.toast('Backend switched', `${result.display_name || name} — ${result.ready ? 'ready' : 'not ready'}`, result.ready ? 'ok' : 'warn');
        await this.refresh();
      } catch (err) { api.report(err, 'switching backend'); }
    },

    async clearCache() {
      try {
        await api.clearCache();
        util.toast('Cache cleared', 'CUDA cache emptied.', 'ok');
      } catch (err) { api.report(err, 'clearing cache'); }
    },
  };

  window.VestiAI.status = status;
})();
