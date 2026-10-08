/* =====================================================================================
 * VestiAI — My Closet: upload, validate, browse, edit, delete
 * ===================================================================================== */
(function () {
  'use strict';
  const { util, api, store } = window.VestiAI;

  const GROUPS = {
    'All': null,
    'T-Shirts': ['t-shirt', 'top'],
    'Shirts': ['shirt'],
    'Jackets': ['jacket', 'sweater'],
    'Kurtas': ['kurta'],
    'Dresses': ['dress'],
    'Traditional': ['traditional'],
    'Formal': ['formal'],
    'Bottoms': ['bottom'],
    'Shoes': ['shoes'],
    'Accessories': ['accessory'],
    'Favourites': 'favourite',
  };

  const closet = {
    filter: 'All',
    uploading: false,

    init() {
      this.renderTabs();
      this.bindUpload();
      util.$('#btn-closet-refresh').addEventListener('click', () => this.refresh(true));
      util.$('#btn-sample-closet').addEventListener('click', () => this.generateSamples());
      store.subscribe((event) => { if (event === 'garments') this.render(); });
    },

    renderTabs() {
      const host = util.$('#closet-tabs');
      host.innerHTML = '';
      Object.keys(GROUPS).forEach((name) => {
        const button = util.el('button', { class: `tab${this.filter === name ? ' active' : ''}`, text: name });
        button.addEventListener('click', () => { this.filter = name; this.renderTabs(); this.render(); });
        host.appendChild(button);
      });
    },

    bindUpload() {
      const dropzone = util.$('#dropzone');
      const input = util.$('#file-input');
      util.$('#btn-browse').addEventListener('click', () => input.click());
      dropzone.addEventListener('click', (event) => { if (event.target === dropzone) input.click(); });
      input.addEventListener('change', () => { this.uploadFiles(Array.from(input.files)); input.value = ''; });

      ['dragenter', 'dragover'].forEach((type) => dropzone.addEventListener(type, (e) => {
        e.preventDefault(); dropzone.classList.add('over');
      }));
      ['dragleave', 'drop'].forEach((type) => dropzone.addEventListener(type, (e) => {
        e.preventDefault(); dropzone.classList.remove('over');
      }));
      dropzone.addEventListener('drop', (e) => {
        const files = Array.from((e.dataTransfer && e.dataTransfer.files) || []);
        if (files.length) this.uploadFiles(files);
      });
      /* paste support (Cmd/Ctrl+V) */
      window.addEventListener('paste', (e) => {
        if (store.state.view !== 'closet') return;
        const items = Array.from(e.clipboardData ? e.clipboardData.files : []);
        if (items.length) this.uploadFiles(items);
      });
    },

    async uploadFiles(files, force) {
      if (this.uploading) { util.toast('Upload in progress', 'Please wait for the current file to finish.', 'warn'); return; }
      const images = files.filter((f) => f.type.startsWith('image/'));
      if (!images.length) { util.toast('No images', 'Drop JPG, PNG or WebP files.', 'warn'); return; }
      const category = util.$('#upload-category').value;
      this.uploading = true;
      const progress = util.$('#upload-progress');
      progress.innerHTML = '';

      for (let i = 0; i < images.length; i += 1) {
        const file = images[i];
        const row = util.el('div', { class: 'status-row' }, [
          util.el('span', { class: 'name', text: file.name.slice(0, 34) }),
          util.el('span', { class: 'spacer' }),
          util.el('span', { class: 'val', text: `${i + 1}/${images.length} · preprocessing…` }),
        ]);
        progress.appendChild(row);
        const form = new FormData();
        form.append('file', file);
        if (category) form.append('category', category);
        if (force) form.append('skip_quality_gate', 'true');
        try {
          const result = await api.uploadGarment(form);
          const garment = result.garment;
          row.querySelector('.val').textContent = `${garment.classification.category} · ${util.fmt.pct(garment.classification.confidence)} confidence`;
          row.classList.add('ok');
          this.showQuality(garment.quality, garment.classification);
          store.setSelectedGarment(garment.key);
        } catch (err) {
          row.querySelector('.val').textContent = err.message;
          row.style.borderColor = 'rgba(248,113,113,.5)';
          if (err.code === 'unsupported_garment' && !force) {
            const report = err.details && err.details.quality_report;
            this.showQuality(report, null, err);
            if (report) {
              const issues = (report.issues || []).join(' ');
              const override = util.el('button', { class: 'btn xs', text: 'Accept anyway' });
              override.addEventListener('click', () => this.uploadFiles([file], true));
              const box = util.el('div', { class: 'notice warn', style: { marginTop: '8px' } }, [
                util.el('div', { text: issues || err.message }), override,
              ]);
              progress.appendChild(box);
            }
          } else {
            api.report(err, file.name);
          }
        }
      }
      this.uploading = false;
      await this.refresh();
      util.toast('Upload finished', `${images.length} file(s) processed.`, 'ok');
    },

    showQuality(quality, classification, error) {
      const host = util.$('#closet-quality');
      if (!quality) {
        host.innerHTML = error ? `<div class="notice err">${error.message}</div>` : '<span class="muted">—</span>';
        return;
      }
      const issues = quality.issues || [];
      const warnings = quality.warnings || [];
      host.innerHTML = '';
      const kind = issues.length ? 'err' : (warnings.length ? 'warn' : 'ok');
      host.appendChild(util.el('div', { class: `notice ${kind}` }, [
        util.el('div', { text: issues.length ? `Rejected: ${issues.join(' ')}` : (warnings.length ? `Accepted with notes: ${warnings.join(' ')}` : 'Passed the quality gate.') }),
      ]));
      const table = util.el('table', {}, [
        util.el('tbody', {}, [
          util.el('tr', {}, [util.el('td', { text: 'Background removal' }), util.el('td', { class: 'mono', text: quality.background_removal || '—' })]),
          util.el('tr', {}, [util.el('td', { text: 'Resolution' }), util.el('td', { class: 'mono', text: `${quality.width}×${quality.height}` })]),
          util.el('tr', {}, [util.el('td', { text: 'Coverage' }), util.el('td', { class: 'mono', text: util.fmt.pct(quality.coverage, 1) })]),
          util.el('tr', {}, [util.el('td', { text: 'Aspect ratio' }), util.el('td', { class: 'mono', text: util.fmt.num(quality.aspect_ratio, 2) })]),
          util.el('tr', {}, [util.el('td', { text: 'Skin ratio' }), util.el('td', { class: 'mono', text: util.fmt.pct(quality.skin_ratio, 1) })]),
          util.el('tr', {}, [util.el('td', { text: 'Hole ratio' }), util.el('td', { class: 'mono', text: util.fmt.pct(quality.hole_ratio, 1) })]),
        ]),
      ]);
      host.appendChild(table);
      if (classification) {
        const scores = Object.entries(classification.scores || {}).slice(0, 4)
          .map(([k, v]) => `${util.fmt.title(k)} ${(v * 100).toFixed(0)}%`).join(' · ');
        host.appendChild(util.el('div', { class: 'field-hint', style: { marginTop: '8px' } }, [
          util.el('b', { text: `${util.fmt.title(classification.category)} ` }),
          document.createTextNode(`${classification.method} · ${scores}`),
        ]));
      }
    },

    async generateSamples() {
      util.toast('Generating', 'Drawing procedural garments and running them through the real preprocessing pipeline…', 'info');
      try {
        const result = await api.sampleCloset(10, Math.floor(Math.random() * 1000));
        await this.refresh();
        util.toast('Sample closet ready', `${result.created} garments created (no downloads, procedurally generated).`, 'ok');
      } catch (err) { api.report(err, 'generating sample closet'); }
    },

    async refresh(report) {
      try {
        const data = await api.garments({ include_preview: false });
        store.state.garments = data.garments || [];
        store.state.closetStats = data.stats || {};
        util.setText('#closet-count', `${data.count} item${data.count === 1 ? '' : 's'}`);
        store.emit('garments');
        if (report) util.toast('Closet refreshed', `${data.count} garments.`, 'ok');
        if (!store.state.selectedGarment && data.garments && data.garments.length) {
          store.setSelectedGarment(data.garments[0].key);
        }
        this.renderLiveStrip();
      } catch (err) { api.report(err, 'loading closet'); }
    },

    renderLiveStrip() {
      const strip = util.$('#live-garment-strip');
      if (!strip) return;
      strip.innerHTML = '';
      const garments = store.state.garments;
      if (!garments.length) {
        strip.innerHTML = '<span class="muted">No garments yet — upload one or generate samples.</span>';
        return;
      }
      garments.slice(0, 14).forEach((g) => {
        const img = util.el('img', { src: g.urls.preview, alt: g.label, title: `${g.label} · ${g.category_label}` });
        img.style.cursor = 'pointer';
        if (g.key === store.state.selectedGarment) img.style.outline = '2px solid var(--accent)';
        img.addEventListener('click', () => {
          store.setSelectedGarment(g.key);
          this.renderLiveStrip();
        });
        strip.appendChild(img);
      });
    },

    visibleGarments() {
      const list = store.state.garments;
      const group = GROUPS[this.filter];
      if (group === undefined) return list;               // "All"
      if (group === 'favourite') return list.filter((g) => g.favourite);
      if (group === null) return list;
      return list.filter((g) => group.includes(g.category));
    },

    render() {
      const grid = util.$('#closet-grid');
      const items = this.visibleGarments();
      grid.innerHTML = '';
      util.show(util.$('#closet-empty'), store.state.garments.length === 0);

      items.forEach((garment) => {
        const card = util.el('div', { class: `garment-card${garment.key === store.state.selectedGarment ? ' selected' : ''}` });
        card.innerHTML = `
          <div class="garment-thumb"><img src="${garment.urls.preview}" alt="${garment.label}" loading="lazy" /></div>
          <div class="garment-meta">
            <div class="garment-title">${garment.favourite ? '<span class="fav">★</span>' : ''}${garment.label}</div>
            <div class="garment-sub">
              <span class="badge">${garment.category_label}</span>
              <span>${garment.garment_type}</span>
            </div>
            <div class="palette">${(garment.palette || []).slice(0, 5).map((c) => `<i style="background:${c}"></i>`).join('')}</div>
            <div class="row" style="gap:6px">
              <button class="btn xs" data-act="select">Wear</button>
              <button class="btn xs" data-act="try">AI try-on</button>
              <button class="btn xs" data-act="fav">${garment.favourite ? 'Unstar' : 'Star'}</button>
              <button class="btn xs" data-act="edit">Edit</button>
              <button class="btn xs danger" data-act="delete">✕</button>
            </div>
          </div>`;

        card.addEventListener('click', (event) => {
          const action = event.target.dataset ? event.target.dataset.act : null;
          if (action === 'select') { store.setSelectedGarment(garment.key); this.render(); this.renderLiveStrip(); window.VestiAI.app.go('live'); }
          else if (action === 'try') this.aiTryOn(garment);
          else if (action === 'fav') this.toggleFavourite(garment);
          else if (action === 'edit') this.edit(garment);
          else if (action === 'delete') this.remove(garment);
          else { store.setSelectedGarment(garment.key); this.render(); this.renderLiveStrip(); }
        });
        grid.appendChild(card);
      });
    },

    async toggleFavourite(garment) {
      try {
        await api.updateGarment(garment.key, { favourite: !garment.favourite });
        await this.refresh();
      } catch (err) { api.report(err, 'updating garment'); }
    },

    async edit(garment) {
      const label = window.prompt('Garment name', garment.label);
      if (label === null) return;
      const category = window.prompt(
        `Category (one of: t-shirt, shirt, jacket, kurta, dress, top, sweater, traditional, formal, bottom, shoes, accessory)`,
        garment.category,
      );
      if (category === null) return;
      try {
        await api.updateGarment(garment.key, { label, category: category.trim() });
        await this.refresh();
        util.toast('Updated', `${label} → ${category}`, 'ok');
      } catch (err) { api.report(err, 'updating garment'); }
    },

    async remove(garment) {
      if (!window.confirm(`Delete "${garment.label}"? The files are removed from garments/.`)) return;
      try {
        await api.deleteGarment(garment.key);
        if (store.state.selectedGarment === garment.key) store.setSelectedGarment(null);
        await this.refresh();
        util.toast('Deleted', garment.label, 'ok');
      } catch (err) { api.report(err, 'deleting garment'); }
    },

    /** Run the server-side AI try-on on a generated mannequin + this garment. */
    async aiTryOn(garment) {
      let personBase64 = window.__vestiaiLastPersonFrame;
      if (!personBase64) {
        const proceed = window.confirm(
          'No camera frame is available yet.\n\nOK = use the most recent capture from captures/ if there is one,\n'
          + 'Cancel = open Live Try-On and capture a frame first.',
        );
        if (!proceed) { window.VestiAI.app.go('live'); return; }
        try {
          const captures = await api.captures('photo');
          if (!captures.captures.length) { util.toast('No photo available', 'Capture a frame in Live Try-On first.', 'warn'); return; }
          const response = await fetch(captures.captures[0].url);
          const blob = await response.blob();
          personBase64 = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result).split(',')[1]);
            reader.readAsDataURL(blob);
          });
        } catch (err) { api.report(err, 'loading capture'); return; }
      }
      util.toast('Running try-on', 'Calling the active backend — this may take a few seconds on GPU, longer on CPU.', 'info');
      try {
        const result = await api.tryOn({ garment_key: garment.key, person_base64: personBase64 });
        window.__vestiaiLastTryOn = result;
        window.VestiAI.app.showTryOnResult(result, garment);
      } catch (err) { api.report(err, 'AI try-on'); }
    },
  };

  window.VestiAI.closet = closet;
})();
