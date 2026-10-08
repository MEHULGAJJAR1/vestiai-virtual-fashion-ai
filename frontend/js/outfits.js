/* =====================================================================================
 * VestiAI — Outfit Builder + style recommendations
 * ===================================================================================== */
(function () {
  'use strict';
  const { util, api, store } = window.VestiAI;

  const outfits = {
    style: 'casual',
    styles: [],
    current: { top: null, bottom: null, shoes: null, accessory: null },

    async init() {
      try {
        const data = await api.styles();
        this.styles = data.styles || [];
      } catch (err) { /* offline: fall back to the built-in list */ }
      if (!this.styles.length) {
        this.styles = ['casual', 'formal', 'traditional', 'party', 'streetwear'].map((key) => ({
          key, label: util.fmt.title(key), description: '', categories: [],
        }));
      }
      this.renderStyleSegments();
      util.$('#btn-auto-outfit').addEventListener('click', () => this.autoBuild());
      util.$('#btn-random-outfit').addEventListener('click', () => this.autoBuild(true));
      util.$('#btn-try-outfit').addEventListener('click', () => this.wearOutfit());
    },

    renderStyleSegments() {
      const host = util.$('#seg-style');
      host.innerHTML = '';
      this.styles.forEach((style) => {
        const button = util.el('button', { class: `${style.key === this.style ? 'active' : ''}`, text: style.label });
        button.title = style.description || '';
        button.addEventListener('click', () => { this.style = style.key; this.renderStyleSegments(); this.load(); });
        host.appendChild(button);
      });
      util.setText('#outfit-style-badge', util.fmt.title(this.style));
    },

    async load() {
      await this.recommend();
      await this.combos();
    },

    async recommend() {
      try {
        const data = await api.recommend({ style: this.style, limit: 8 });
        if (!data.ok) {
          util.$('#reco-grid').innerHTML = `<span class="muted">${data.reason || 'No recommendations yet.'}</span>`;
          return;
        }
        util.setText('#reco-engine', data.engine === 'colour_rules' ? 'colour rules (explainable)' : data.engine);
        const grid = util.$('#reco-grid');
        grid.innerHTML = '';
        (data.suggestions || []).forEach((item) => {
          const garment = store.garmentByKey(item.garment_key);
          if (!garment) return;
          const card = util.el('div', { class: 'garment-card' });
          card.innerHTML = `
            <div class="garment-thumb"><img src="${garment.urls.preview}" alt="${item.label}" loading="lazy" /></div>
            <div class="garment-meta">
              <div class="garment-title">${item.label}</div>
              <div class="palette">${(item.palette || []).slice(0, 4).map((c) => `<i style="background:${c}"></i>`).join('')}</div>
              <div class="field-hint">${(item.reasons || []).slice(0, 2).join(' ')}</div>
              <div class="row" style="gap:6px">
                <span class="badge">score ${item.score.toFixed(2)}</span>
                <button class="btn xs" data-act="wear">Wear</button>
              </div>
            </div>`;
          card.addEventListener('click', (event) => {
            if (event.target.dataset.act === 'wear') {
              store.setSelectedGarment(garment.key);
              window.VestiAI.app.go('live');
            } else {
              this.pairing(garment.key);
            }
          });
          grid.appendChild(card);
        });
        if (!grid.children.length) grid.innerHTML = '<span class="muted">Your closet is empty — upload garments first.</span>';
      } catch (err) { api.report(err, 'recommendations'); }
    },

    async pairing(key) {
      try {
        const data = await api.pairing(key);
        const host = util.$('#outfit-pairing');
        host.innerHTML = '<div class="row" style="margin-bottom:8px">'
          + `<span class="badge">base ${data.base}</span>`
          + `<span style="width:26px;height:26px;border-radius:8px;background:${data.base};display:inline-block"></span></div>`;
        data.pairings.forEach((p) => {
          host.appendChild(util.el('div', { class: 'status-row' }, [
            util.el('span', { class: 'name', text: p.relation }),
            util.el('span', { class: 'spacer' }),
            util.el('span', { class: 'val', text: p.hex }),
            util.el('span', { style: { width: '18px', height: '18px', borderRadius: '6px', background: p.hex, display: 'inline-block' } }),
          ]));
        });
      } catch (err) { api.report(err, 'colour pairing'); }
    },

    async combos() {
      try {
        const data = await api.recommend({ style: this.style, limit: 24 });
        const tops = (data.suggestions || []).filter((s) => {
          const g = store.garmentByKey(s.garment_key); return g && g.garment_type === 'top';
        });
        const bottoms = store.state.garments.filter((g) => g.garment_type === 'bottom');
        const host = util.$('#combo-list');
        host.innerHTML = '';
        if (!tops.length || !bottoms.length) {
          host.innerHTML = '<span class="muted">Add at least one top and one bottom to your closet to see combinations.</span>';
          return;
        }
        const rows = [];
        tops.slice(0, 4).forEach((top) => bottoms.slice(0, 4).forEach((bottom) => rows.push({ top, bottom })));
        rows.slice(0, 8).forEach(({ top, bottom }) => {
          host.appendChild(util.el('div', { class: 'status-row' }, [
            util.el('span', { class: 'name', text: `${top.label} + ${bottom.label}` }),
            util.el('span', { class: 'spacer' }),
            util.el('button', {
              class: 'btn xs',
              text: 'Build',
              onclick: () => {
                this.current.top = top.garment_key; this.current.bottom = bottom.key;
                this.renderSlots();
              },
            }),
          ]));
        });
      } catch (err) { /* non-fatal */ }
    },

    async autoBuild(surprise) {
      try {
        const data = await api.buildOutfit({ style: this.style, seed: surprise ? Math.floor(Math.random() * 1e6) : 42 });
        if (!data.ok) { util.toast('Could not build an outfit', data.reason || 'Not enough garments.', 'warn'); return; }
        const slots = data.slots || {};   /* the server omits empty slots */
        this.current = {
          top: slots.top ? slots.top.garment_key : null,
          bottom: slots.bottom ? slots.bottom.garment_key : null,
          shoes: slots.shoes ? slots.shoes.garment_key : null,
          accessory: slots.accessory ? slots.accessory.garment_key : null,
        };
        this.renderSlots();
        const notes = util.$('#outfit-notes');
        notes.innerHTML = '';
        if (data.notes && data.notes.length) {
          notes.appendChild(util.el('div', { class: 'notice info', text: data.notes.join(' ') }));
        }
        util.toast('Outfit built', `${data.style_label} · score ${data.score}`, 'ok');
      } catch (err) { api.report(err, 'building outfit'); }
    },

    renderSlots() {
      const host = util.$('#outfit-slots');
      host.innerHTML = '';
      const slots = [['top', 'Top'], ['bottom', 'Bottom'], ['shoes', 'Shoes'], ['accessory', 'Accessory']];
      let filled = 0;
      slots.forEach(([slot, label]) => {
        const key = this.current[slot];
        const garment = key ? store.garmentByKey(key) : null;
        if (garment) filled += 1;
        const node = util.el('div', { class: `slot${garment ? ' filled' : ''}` });
        if (garment) {
          node.innerHTML = `<div><img src="${garment.urls.preview}" alt="${garment.label}" />
            <div class="field-hint" style="margin-top:6px">${garment.label}</div></div>`;
        } else {
          node.innerHTML = `<div><div class="muted" style="font-size:1.6rem">＋</div><div class="field-hint">${label}<br/>empty</div></div>`;
        }
        node.style.cursor = 'pointer';
        node.addEventListener('click', () => this.pickGarment(slot));
        host.appendChild(node);
      });
      util.$('#btn-try-outfit').disabled = filled === 0;
    },

    pickGarment(slot) {
      const candidates = store.state.garments.filter((g) => {
        if (slot === 'top') return g.garment_type === 'top';
        return g.garment_type === slot;
      });
      if (!candidates.length) { util.toast('Nothing to choose', `No ${slot} garments in your closet yet.`, 'warn'); return; }
      const names = candidates.map((g, i) => `${i + 1}. ${g.label} (${g.category})`).join('\n');
      const answer = window.prompt(`Choose a ${slot} by number:\n\n${names}`, '1');
      if (!answer) return;
      const index = parseInt(answer, 10) - 1;
      if (index >= 0 && index < candidates.length) {
        this.current[slot] = candidates[index].key;
        this.renderSlots();
        this.pairing(candidates[index].key);
      }
    },

    wearOutfit() {
      const keys = ['top', 'bottom', 'shoes', 'accessory'].map((s) => this.current[s]).filter(Boolean);
      if (keys.length) {
        store.setSelectedGarment(keys[0]);
        util.toast('Outfit applied', `${keys.length} pieces — starting with the top. Use Change Outfit to rotate.`, 'ok');
        window.VestiAI.app.go('live');
      }
    },
  };

  window.VestiAI.outfits = outfits;
})();
