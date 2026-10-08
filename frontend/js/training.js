/* =====================================================================================
 * VestiAI — AI Training Dashboard
 * Polls /api/training/status and renders real numbers: dataset size, epoch, losses, VRAM,
 * timings, checkpoints, validation sample grids, evaluation reports and the live log.
 * ===================================================================================== */
(function () {
  'use strict';
  const { util, api, store, charts } = window.VestiAI;

  const MODE_DESCRIPTIONS = {
    QUICK_DEMO: 'Small subset (≤64 samples), 256 px, 1–2 epochs — verifies the whole pipeline in minutes, even on CPU.',
    FINE_TUNE: 'Adapts the pretrained model to your garment categories with ControlNet + UNet LoRA at your chosen resolution.',
    FULL_TRAINING: 'The complete dataset with full epochs, periodic validation and sample generation.',
  };

  const training = {
    pollTimer: null,
    lastHistory: [],

    async init() {
      util.$$('#seg-mode button').forEach((button) => {
        button.addEventListener('click', () => {
          store.state.trainingMode = button.dataset.mode;
          util.$$('#seg-mode button').forEach((b) => b.classList.toggle('active', b === button));
          this.applyModeDefaults(button.dataset.mode);
        });
      });
      util.$('#btn-start-training').addEventListener('click', () => this.start());
      util.$('#btn-stop-training').addEventListener('click', () => this.stop());
      util.$('#btn-prepare-samples').addEventListener('click', () => this.generateDataset());
      util.$('#btn-refresh-training').addEventListener('click', () => this.refresh(true));
      util.$('#btn-reload-model').addEventListener('click', () => this.reloadModel());
      util.setText('#tr-cli-hint', 'CLI equivalents: python scripts/train.py --mode QUICK_DEMO | FINE_TUNE | FULL_TRAINING');
      this.applyModeDefaults(store.state.trainingMode);
      await this.refresh();
      this.startPolling();
    },

    applyModeDefaults(mode) {
      const presets = {
        QUICK_DEMO: { resolution: 256, batch: 1, epochs: 2, lr: 0.00001, accum: 2, lora: false },
        FINE_TUNE: { resolution: 512, batch: 2, epochs: 5, lr: 0.00001, accum: 4, lora: true },
        FULL_TRAINING: { resolution: 512, batch: 4, epochs: 30, lr: 0.00001, accum: 4, lora: false },
      }[mode] || {};
      if (presets.resolution) util.$('#tr-resolution').value = String(presets.resolution);
      if (presets.batch) util.$('#tr-batch').value = String(presets.batch);
      if (presets.epochs) util.$('#tr-epochs').value = String(presets.epochs);
      if (presets.lr) util.$('#tr-lr').value = String(presets.lr);
      if (presets.accum) util.$('#tr-accum').value = String(presets.accum);
      util.$('#tr-lora').checked = !!presets.lora;
      util.setText('#tr-mode-desc', MODE_DESCRIPTIONS[mode] || '');
    },

    async generateDataset() {
      util.toast('Generating dataset', 'Creating a VITON-HD-compatible sample dataset (person, garment, mask, agnostic, pose, pairs).', 'info');
      try {
        const result = await api.datasetGenerate({ train: 24, val: 6, test: 6, size: 512 });
        util.toast('Dataset ready', `${result.path} · ${JSON.stringify(result.counts)}`, 'ok');
        await this.refresh(true);
      } catch (err) { api.report(err, 'generating dataset'); }
    },

    async start() {
      const mode = store.state.trainingMode;
      const overrides = {
        resolution: Number(util.$('#tr-resolution').value),
        batch_size: Number(util.$('#tr-batch').value),
        num_epochs: Number(util.$('#tr-epochs').value),
        learning_rate: Number(util.$('#tr-lr').value),
        gradient_accumulation: Number(util.$('#tr-accum').value),
        train_unet_lora: util.$('#tr-lora').checked,
        enable_perceptual_loss: util.$('#tr-perceptual').checked,
      };
      const dataset = util.$('#tr-dataset-select').value || null;
      const dryRun = util.$('#tr-dry').checked;
      util.$('#btn-start-training').disabled = true;
      util.toast('Training starting', `${mode} · ${util.$('#tr-resolution').value}px · batch ${overrides.batch_size}`
        + (dryRun ? ' (dry run: 2 steps)' : ''), 'info');
      try {
        const result = await api.startTraining({ mode, overrides, dataset_root: dataset, dry_run: dryRun });
        util.toast('Run started', `Job ${result.job.job_id} (pid ${result.job.pid}).`, 'ok');
        await this.refresh(true);
      } catch (err) {
        api.report(err, 'starting training');
      } finally {
        util.$('#btn-start-training').disabled = false;
      }
    },

    async stop() {
      if (!window.confirm('Stop the active training run? The trainer checkpoints before exiting when possible.')) return;
      try {
        await api.stopTraining();
        util.toast('Stop requested', 'The trainer will finish the current step and exit.', 'warn');
        await this.refresh(true);
      } catch (err) { api.report(err, 'stopping training'); }
    },

    async reloadModel() {
      try {
        const result = await api.reloadModel();
        util.toast('Model reloaded', result.detail || 'Done.', result.ready ? 'ok' : 'warn');
        window.VestiAI.status.refresh();
      } catch (err) { api.report(err, 'reloading model'); }
    },

    startPolling() {
      if (this.pollTimer) return;
      const tick = async () => {
        try { await this.refresh(); } catch (e) { /* keep polling */ }
        const active = store.state.training && store.state.training.active_job;
        const delay = active ? 2000 : 12000;
        this.pollTimer = setTimeout(tick, delay);
      };
      this.pollTimer = setTimeout(tick, 4000);
    },

    async refresh(report) {
      const data = await api.trainingStatus();
      store.state.training = data;
      this.render(data);
      if (report) util.toast('Dashboard refreshed', data.active_job ? `run ${data.active_job.state}` : 'no active run', 'ok');
    },

    render(data) {
      const active = data.active_job;
      const trainer = data.trainer || {};
      const ds = data.dataset || {};
      const history = (data.history && data.history.epochs) || [];

      /* ----- dataset select */
      const select = util.$('#tr-dataset-select');
      const datasets = (ds.available || []);
      const currentValue = select.value;
      select.innerHTML = '';
      if (!datasets.length) {
        select.appendChild(util.el('option', { value: '', text: 'no dataset — press “Generate sample dataset”' }));
      }
      datasets.forEach((entry) => {
        const total = entry.total || 0;
        select.appendChild(util.el('option', { value: entry.path, text: `${entry.name} · ${total} samples` }));
      });
      if (currentValue) select.value = currentValue;
      if (!select.value && ds.active) select.value = ds.active;

      const activeEntry = datasets.find((d) => d.path === (active ? (active.config || {}).dataset_root : select.value)) || datasets[0];
      util.setText('#tr-dataset', activeEntry ? `${activeEntry.total} samples` : 'none');
      const counts = activeEntry ? activeEntry.counts : {};
      util.setText('#tr-epoch', trainer.epoch ? `${trainer.epoch}` : '—');
      util.setText('#tr-loss', trainer.loss !== null && trainer.loss !== undefined ? Number(trainer.loss).toFixed(4) : '—');
      util.setText('#tr-valloss', trainer.val_loss !== null && trainer.val_loss !== undefined ? Number(trainer.val_loss).toFixed(4) : '—');
      util.setText('#tr-vram', trainer.vram_mb ? util.fmt.mb(trainer.vram_mb) : (data.device && data.device.vram_total_gb && data.device.vram_total_gb.length
        ? `0 / ${data.device.vram_total_gb[0]} GB` : 'CPU'));
      util.setText('#tr-time', `${util.fmt.duration(trainer.elapsed_s)} / ${util.fmt.duration(trainer.eta_s)}`);
      util.setText('#tr-checkpoint', (data.checkpoints && data.checkpoints.inference_path) ? data.checkpoints.inference_path.split('/').slice(-2).join('/') : 'none');
      util.setText('#tr-model', data.checkpoints && data.checkpoints.ready_for_inference ? 'ready for inference' : 'no trained checkpoint');

      /* ----- progress */
      const progress = trainer.total_steps ? util.clamp((trainer.step || 0) / trainer.total_steps, 0, 1) : 0;
      util.$('#tr-progress-bar').style.width = `${(progress * 100).toFixed(1)}%`;
      util.setText('#tr-state', trainer.state || (active ? active.state : 'idle'));
      util.setText('#tr-step', `${trainer.step || 0} / ${trainer.total_steps || '—'}`);
      util.setText('#tr-lr-live', trainer.lr ? Number(trainer.lr).toExponential(2) : '—');
      const device = data.device || {};
      util.setText('#tr-device', `${device.resolved_device_label || device.resolved_device || '—'} · ${data.trainer && data.trainer.mixed_precision ? data.trainer.mixed_precision : '—'}`);
      util.setText('#tr-params', data.checkpoints && data.checkpoints.model_card && data.checkpoints.model_card.trainable_parameters
        ? `${(data.checkpoints.model_card.trainable_parameters / 1e6).toFixed(1)} M` : 'see run log');

      util.setText('#tr-job-state', active ? `${active.state} · pid ${active.pid || '—'}` : 'idle');
      util.$('#btn-stop-training').disabled = !active;

      /* ----- warnings */
      const warnHost = util.$('#tr-warnings');
      warnHost.innerHTML = '';
      const warnings = (trainer.warnings || []).concat(data.vram_warning ? [data.vram_warning] : []);
      warnings.slice(0, 3).forEach((text) => warnHost.appendChild(util.el('div', { class: 'notice warn', style: { marginBottom: '8px' }, text })));

      const preWarn = util.$('#tr-warning');
      preWarn.innerHTML = '';
      const mlReady = data.device && data.device.torch_available && data.device.torch_version;
      if (!mlReady) {
        preWarn.appendChild(util.el('div', { class: 'notice err', text: 'PyTorch is not installed in this environment — training will fail immediately. Run: python scripts/setup.py --profile ml' }));
      } else if (!data.device.cuda_available) {
        preWarn.appendChild(util.el('div', { class: 'notice warn', text: 'No CUDA GPU detected: QUICK_DEMO works on CPU (minutes), but FINE_TUNE/FULL_TRAINING belong on a CUDA machine or Colab/RunPod. See docs/TRAINING.md.' }));
      }
      if (ds.sample_mode_ready && activeEntry && activeEntry.name === 'samples') {
        preWarn.appendChild(util.el('div', { class: 'notice info', text: 'The sample dataset is procedurally generated — ideal for verifying the pipeline, not for photorealistic quality. Point --dataset at a VITON-HD/DressCode conversion for real training.' }));
      }

      /* ----- charts */
      const curves = (data.metrics && data.metrics.curves) || {};
      const seriesFrom = (name) => (curves[name] || []).map((p) => [p.step, p.value]);
      charts.line(util.$('#chart-loss'), [
        { points: seriesFrom('train/loss'), color: '#7aa2f7' },
        { points: seriesFrom('val/loss'), color: '#f7768e' },
      ]);
      charts.line(util.$('#chart-quality'), [
        { points: seriesFrom('val/ssim'), color: '#9ece6a' },
        { points: seriesFrom('val/psnr'), color: '#e0af68' },
        { points: seriesFrom('val/masked_l1'), color: '#bb9af7' },
      ]);

      /* ----- validation grids */
      const grids = (data.validation && data.validation.grids) || [];
      const gridHost = util.$('#tr-validation-grids');
      const badge = util.$('#tr-val-badge');
      util.setText('#tr-val-badge', grids.length ? `${grids.length} grids` : 'none');
      void badge;
      if (grids.length) {
        gridHost.innerHTML = '';
        grids.slice(-8).forEach((url) => {
          const img = util.el('img', { src: url, alt: 'validation grid', title: url });
          img.style.cursor = 'pointer';
          img.addEventListener('click', () => window.open(url, '_blank'));
          gridHost.appendChild(img);
        });
      } else if (!gridHost.dataset.filled) {
        gridHost.dataset.filled = '1';
      }

      /* ----- checkpoints */
      const cpHost = util.$('#tr-checkpoints');
      const checkpoints = (data.checkpoints && data.checkpoints.epochs) || [];
      cpHost.innerHTML = '';
      if (data.checkpoints && data.checkpoints.best) {
        cpHost.appendChild(util.el('div', { class: 'notice ok', style: { marginBottom: '10px' },
          text: `best_model · epoch ${data.checkpoints.best.epoch || '?'} · val_loss ${data.checkpoints.best.metric !== null && data.checkpoints.best.metric !== undefined ? Number(data.checkpoints.best.metric).toFixed(4) : '—'} · ${data.checkpoints.best.size_mb} MB` }));
      }
      if (!checkpoints.length) {
        cpHost.appendChild(util.el('div', { class: 'muted', text: 'No epoch checkpoints yet.' }));
      } else {
        checkpoints.slice(0, 8).forEach((cp) => {
          const row = util.el('div', { class: 'status-row' }, [
            util.el('span', { class: 'name', text: cp.name }),
            util.el('span', { class: 'badge', text: `epoch ${cp.epoch || '?'}` }),
            util.el('span', { class: 'val', text: cp.metric !== null && cp.metric !== undefined ? `val ${Number(cp.metric).toFixed(4)}` : '—' }),
            util.el('span', { class: 'spacer' }),
            util.el('span', { class: 'val', text: `${cp.size_mb} MB` }),
          ]);
          const promote = util.el('button', { class: 'btn xs', text: 'Promote' });
          promote.addEventListener('click', async () => {
            try {
              await api.promoteCheckpoint(cp.name);
              util.toast('Promoted', `${cp.name} is now best_model.`, 'ok');
              await this.refresh();
            } catch (err) { api.report(err, 'promoting checkpoint'); }
          });
          const remove = util.el('button', { class: 'btn xs danger', text: '✕' });
          remove.addEventListener('click', async () => {
            if (!window.confirm(`Delete ${cp.name}?`)) return;
            try { await api.deleteCheckpoint(cp.name); await this.refresh(); } catch (err) { api.report(err, 'deleting checkpoint'); }
          });
          row.appendChild(promote); row.appendChild(remove);
          cpHost.appendChild(row);
        });
      }

      /* ----- evaluations */
      const evalHost = util.$('#tr-evaluations');
      const evaluations = data.evaluations || [];
      evalHost.innerHTML = '';
      if (!evaluations.length) {
        evalHost.appendChild(util.el('div', { class: 'muted', text: 'Run: python scripts/evaluate.py --checkpoint checkpoints/best_model' }));
      } else {
        evaluations.forEach((report) => {
          const metrics = report.metrics || {};
          evalHost.appendChild(util.el('div', { class: 'status-row' }, [
            util.el('span', { class: 'name', text: report.created_at_iso || '' }),
            util.el('span', { class: 'badge', text: `${report.samples} samples` }),
            util.el('span', { class: 'spacer' }),
            util.el('span', { class: 'val', text: `SSIM ${util.fmt.num(metrics.ssim, 3)} · PSNR ${util.fmt.num(metrics.psnr, 1)} · L1 ${util.fmt.num(metrics.masked_l1, 3)}` }),
          ]));
        });
      }

      /* ----- history table */
      if (history.length && history !== this.lastHistory) {
        this.lastHistory = history;
      }

      /* ----- log */
      const logHost = util.$('#tr-log');
      const lines = data.log_tail || [];
      if (lines.length) {
        logHost.textContent = lines.join('\n');
        logHost.scrollTop = logHost.scrollHeight;
      } else if (!logHost.dataset.kept) {
        logHost.textContent = 'No training log yet. Start a QUICK_DEMO run to see live output here.';
      }
      void counts;
    },
  };

  window.VestiAI.training = training;
})();
