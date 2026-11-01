// OP Sensor content script: puts a Sensor button on video players (YouTube,
// Twitch, Kick, Facebook, or any site the user enables) and, while it is on,
// finds the One Piece card under the mouse and shows it in high resolution with
// its text and price. Frames never leave the machine: they go to the
// extension's own offscreen engine.
(() => {
  if (window.__opsLens) return;
  window.__opsLens = true;

  const DET_W = 640;
  const DET_H = 384;
  const ID_SIZE = 224;
  const MIN_SCORE = 0.45;      // fallback thresholds; the real ones come from faces.json (calibrated at build time)
  const MIN_JUNK_GAP = 0.04;   // ...and it must beat the "not a card" prototype
  const MIN_MARGIN = 0.03;     // ...and the next-best *different* card (near-ties were wrong on real frames)
  const DETECT_TTL_MS = 450;   // how long detections stay fresh on a playing video
  const HOST = location.hostname;
  const IS_YT = /(^|\.)youtube(-nocookie)?\.com$/.test(HOST);
  const IS_TWITCH = /(^|\.)twitch\.tv$/.test(HOST);

  let settings = {
    enabled: true, size: 'm', highlight: true, corner: 'right',
    lang: 'auto', showEffect: true, showPrice: true, priceSource: 'liga', currency: 'BRL',
    sens: 'balanced', debug: false,
  };
  const tr = (key, vars) => OPS.t(settings.lang, key, vars);
  const STORE_NAMES = { liga: 'Liga One Piece', tcgplayer: 'TCGplayer', cardmarket: 'Cardmarket' };

  function fmtMoney(amount, currency) {
    const locale = { BRL: 'pt-BR', USD: 'en-US', EUR: 'de-DE' }[currency] || undefined;
    try {
      return new Intl.NumberFormat(locale, { style: 'currency', currency }).format(amount);
    } catch (e) {
      return `${currency} ${amount.toFixed(2)}`;
    }
  }

  chrome.storage.sync.get(settings).then((s) => {
    settings = { ...settings, ...s };
    scan();
  });
  chrome.storage.onChanged.addListener((changes) => {
    for (const [k, v] of Object.entries(changes)) settings[k] = v.newValue;
    for (const lens of lenses.values()) lens.applySettings();
    scan();
  });

  function send(msg, timeoutMs = 10000) {
    return new Promise((resolve) => {
      // never let a lost reply wedge the hover loop
      const timer = setTimeout(() => resolve({ error: 'timed out' }), timeoutMs);
      const done = (res) => {
        clearTimeout(timer);
        resolve(res);
      };
      try {
        chrome.runtime.sendMessage(msg, (res) => {
          if (chrome.runtime.lastError) done({ error: chrome.runtime.lastError.message });
          else done(res || { error: 'no response' });
        });
      } catch (e) {
        done({ error: String(e) }); // extension was reloaded under us
      }
    });
  }

  // ------------------------------------------------------------ DOM helpers
  function el(tag, attrs = {}, children = []) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;
      else n.setAttribute(k, v);
    }
    for (const c of children) n.append(c);
    return n;
  }

  function svg(tag, attrs = {}, children = []) {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    for (const c of children) n.append(c);
    return n;
  }

  // the logo as a one-colour glyph: a card locked in targeting brackets
  function lensIcon() {
    return svg('svg', { viewBox: '0 0 24 24', width: '20', height: '20', 'aria-hidden': 'true' }, [
      svg('path', {
        d: 'M3 8V3h5 M16 3h5v5 M3 16v5h5 M16 21h5v-5',
        fill: 'none', stroke: 'currentColor', 'stroke-width': '2.2', 'stroke-linecap': 'square',
      }),
      svg('rect', { x: '8.4', y: '6.6', width: '7.2', height: '10.8', rx: '1.1', fill: 'currentColor' }),
    ]);
  }

  const CSS = `
    :host { all: initial; }
    .root { position: absolute; inset: 0; pointer-events: none; z-index: 70; font-family: "Segoe UI", Roboto, Arial, sans-serif; }
    .btn { position: absolute; top: 12px; right: 12px; width: 38px; height: 38px; border-radius: 50%;
      display: grid; place-items: center; cursor: pointer; pointer-events: auto; color: #fff;
      background: rgba(12, 14, 20, 0.62); border: 1.5px solid rgba(255,255,255,0.35);
      box-shadow: 0 2px 10px rgba(0,0,0,0.45); transition: opacity .35s, border-color .2s, box-shadow .2s, transform .15s;
      opacity: 0; }
    /* like the player's own controls: only while the mouse is moving (or starting up) */
    .root.show .btn, .btn:hover, .btn.busy { opacity: 1; }
    .root.left .btn { right: auto; left: 12px; }
    .btn:hover { transform: scale(1.06); }
    .btn.on { color: #ff5a4d; border-color: #ff5a4d; box-shadow: 0 0 0 2px rgba(255,90,77,0.25), 0 0 16px rgba(255,90,77,0.55); }
    .btn.busy::after { content: ""; position: absolute; inset: -4px; border-radius: 50%;
      border: 2px solid transparent; border-top-color: #ff5a4d; animation: spin .8s linear infinite; }
    .btn.err { color: #ff6b6b; border-color: #ff6b6b; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .toast { position: absolute; top: 16px; right: 60px; max-width: 60%; padding: 7px 12px; border-radius: 8px;
      background: rgba(12,14,20,0.82); color: #f2f2f2; font-size: 13px; line-height: 1.3; opacity: 0;
      transition: opacity .25s; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .toast.show { opacity: 1; }
    .root.left .toast { right: auto; left: 60px; }
    /* targeting reticle: four corner brackets that snap onto the card */
    .hl { position: absolute; opacity: 0; pointer-events: none;
      transition: opacity .12s, left .1s linear, top .1s linear, width .1s linear, height .1s linear; }
    .hl.show { opacity: 1; }
    .hl::before { content: ""; position: absolute; inset: 0; border: 1px solid rgba(255,90,77,0.35); border-radius: 5px;
      box-shadow: inset 0 0 10px rgba(255,90,77,0.18); }
    .hl i { position: absolute; width: 24%; height: 18%; max-width: 30px; max-height: 30px; border: 0 solid #ff5a4d;
      filter: drop-shadow(0 0 4px rgba(255,90,77,0.85)); }
    .hl i:nth-child(1) { left: -2px; top: -2px; border-width: 3px 0 0 3px; border-top-left-radius: 5px; }
    .hl i:nth-child(2) { right: -2px; top: -2px; border-width: 3px 3px 0 0; border-top-right-radius: 5px; }
    .hl i:nth-child(3) { left: -2px; bottom: -2px; border-width: 0 0 3px 3px; border-bottom-left-radius: 5px; }
    .hl i:nth-child(4) { right: -2px; bottom: -2px; border-width: 0 3px 3px 0; border-bottom-right-radius: 5px; }
    .hl.lock { transition: opacity .1s; animation: lockon .34s cubic-bezier(.2,.9,.3,1.15) both; }
    @keyframes lockon { 0% { transform: scale(1.4) rotate(-10deg); opacity: 0; } 60% { transform: scale(0.97) rotate(0); opacity: 1; } 100% { transform: none; opacity: 1; } }    /* the big card: slides in from the hovered card's side, tilts a little with the mouse */
    .card { position: absolute; opacity: 0; perspective: 900px;
      transform: translate(var(--fx, 0px), var(--fy, 0px)) scale(0.95);
      transition: opacity .2s ease-out, transform .24s cubic-bezier(.2,.8,.2,1); }
    .card.show { opacity: 1; transform: none; }
    .card.reset { transition: none; }
    .face { position: relative; width: 100%; height: 100%; border-radius: 4.5% / 3.2%; overflow: hidden; background: #111;
      transform: rotateX(var(--rx, 0deg)) rotateY(var(--ry, 0deg)); transition: transform .18s ease-out;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.10), 0 4px 12px rgba(0,0,0,0.45), 0 18px 44px rgba(0,0,0,0.55); }
    .card.land .face { border-radius: 3.2% / 4.5%; }
    .face img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .face::after { content: ""; position: absolute; inset: 0; pointer-events: none; mix-blend-mode: screen;
      background: linear-gradient(115deg, transparent 38%, rgba(255,255,255,0.13) 50%, transparent 62%);
      background-size: 260% 100%; background-position: var(--mx, 50%) 0; transition: background-position .18s ease-out; }
    .under { position: absolute; left: 0; right: 0; top: 100%; margin-top: 8px; display: flex; flex-direction: column;
      align-items: center; gap: 6px; }
    .cap { box-sizing: border-box; max-width: 100%; padding: 4px 11px; border-radius: 999px; color: #fff; font-size: 12.5px; line-height: 1.35;
      background: rgba(10,12,18,0.78); border: 1px solid rgba(255,255,255,0.09); backdrop-filter: blur(6px);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .cap b { font-weight: 600; }
    .cap span { opacity: 0.72; }
    .cap .px { opacity: 1; margin-left: 8px; padding: 1px 7px; border-radius: 4px; font-size: 11.5px; font-weight: 700;
      color: #111; background: #ffd666; }
    .cap .px.ref { background: transparent; color: #ffd666; border: 1px solid rgba(255,214,102,0.55); font-weight: 600; }
    .cap .px.na { background: transparent; color: #c9ccd6; border: 1px solid rgba(255,255,255,0.25); font-weight: 500; }
    .fx { box-sizing: border-box; width: 100%; padding: 6px 10px; border-radius: 8px; color: #eceef4; font-size: 11.5px; line-height: 1.35;
      background: rgba(10,12,18,0.80); border: 1px solid rgba(255,255,255,0.09); backdrop-filter: blur(6px);
      display: -webkit-box; -webkit-line-clamp: 5; -webkit-box-orient: vertical; overflow: hidden; text-align: left; }
    .fx[hidden] { display: none; }
    .fx .tg { display: block; margin-top: 3px; color: #ffd666; }
    .fx .stat { display: block; margin-bottom: 3px; opacity: 0.7; font-size: 10.8px; }
  `;

  // ------------------------------------------------------------ geometry
  function contentRect(video) {
    const r = video.getBoundingClientRect();
    const vw = video.videoWidth || r.width;
    const vh = video.videoHeight || r.height;
    const s = Math.min(r.width / vw, r.height / vh);
    const w = vw * s;
    const h = vh * s;
    return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h, scale: s };
  }

  // ------------------------------------------------------------ per-player lens
  class Lens {
    constructor(container, video) {
      this.container = container;
      this.video = video;
      this.active = false;
      this.dets = null;        // {key, time, boxes (video px), snap}
      this.current = null;     // box currently under the mouse
      this.ids = new Map();    // box key -> identify result
      this.mouse = null;
      this.busy = false;
      this.mount();
    }

    mount() {
      if (getComputedStyle(this.container).position === 'static') this.container.style.position = 'relative';
      this.host = el('div', { class: 'ops-lens-host' });
      Object.assign(this.host.style, { position: 'absolute', inset: '0', pointerEvents: 'none', zIndex: '70' });
      const shadow = this.host.attachShadow({ mode: 'open' });
      const style = el('style');
      style.textContent = CSS;
      this.root = el('div', { class: 'root' });
      this.btn = el('button', { class: 'btn', title: tr('btn_title'), 'aria-label': 'Toggle OP Sensor' }, [lensIcon()]);
      this.toastEl = el('div', { class: 'toast' });
      this.hl = el('div', { class: 'hl' }, [el('i'), el('i'), el('i'), el('i')]);
      this.hl.addEventListener('animationend', (e) => {
        if (e.animationName === 'lockon') this.hl.classList.remove('lock');
      });
      this.img = el('img', { alt: '' });
      this.cap = el('div', { class: 'cap' });
      this.fx = el('div', { class: 'fx' });
      this.fx.hidden = true;
      this.card = el('div', { class: 'card' }, [el('div', { class: 'face' }, [this.img]), el('div', { class: 'under' }, [this.cap, this.fx])]);
      this.root.append(this.hl, this.card, this.toastEl, this.btn);
      shadow.append(style, this.root);
      this.container.append(this.host);

      // start loading the models as soon as the pointer heads for the button
      this.btn.addEventListener('mouseenter', () => send({ op: 'status' }), { once: true });
      this.btn.addEventListener('click', (e) => {
        e.stopPropagation();
        e.preventDefault();
        this.toggle();
      });
      for (const t of ['mousedown', 'mouseup', 'dblclick', 'pointerdown', 'pointerup']) {
        this.btn.addEventListener(t, (e) => e.stopPropagation()); // don't pause the video
      }
      this.onMove = (e) => this.move(e);
      this.onLeave = () => this.leave();
      this.container.addEventListener('mousemove', this.onMove, { passive: true, capture: true });
      this.container.addEventListener('mouseleave', this.onLeave, { passive: true });
      this.applySettings();
    }

    applySettings() {
      const show = settings.enabled && !inPip(this.container, this.video);
      this.host.style.display = show ? '' : 'none';
      this.root.classList.toggle('left', settings.corner === 'left');
      this.btn.title = tr('btn_title');
      if (!show && this.active) this.toggle();
    }

    destroy() {
      this.container.removeEventListener('mousemove', this.onMove, { capture: true });
      this.container.removeEventListener('mouseleave', this.onLeave);
      clearInterval(this.timer);
      this.host.remove();
    }

    toast(text, ms = 2200) {
      this.toastEl.textContent = text;
      this.toastEl.classList.add('show');
      clearTimeout(this.toastTimer);
      if (ms) this.toastTimer = setTimeout(() => this.toastEl.classList.remove('show'), ms);
    }

    async toggle() {
      this.active = !this.active;
      this.btn.classList.toggle('on', this.active);
      this.btn.classList.remove('err');
      if (!this.active) {
        clearInterval(this.timer);
        this.hide();
        this.toast(tr('off'), 1200);
        return;
      }
      this.btn.classList.add('busy');
      this.toast(tr('starting'), 0);
      let st;
      try {
        st = await this.waitReady();
      } catch (e) {
        st = { ready: false, error: String(e) };
      }
      this.btn.classList.remove('busy');
      if (!st.ready) {
        this.active = false;
        this.btn.classList.remove('on');
        this.btn.classList.add('err');
        this.toast(tr('failed', { err: st.error || 'unknown error' }), 8000);
        return;
      }
      this.toast(tr('on'), 2000);
      // keep tracking a stationary mouse on a playing video
      this.timer = setInterval(() => {
        if (this.mouse && !this.video.paused) this.tick();
      }, 650);
      if (this.mouse) this.tick();
    }

    async waitReady() {
      for (let i = 0; i < 120; i++) {
        const st = await send({ op: 'status' });
        if (st && (st.ready || st.error)) return st;
        await new Promise((r) => setTimeout(r, 250));
      }
      return { ready: false, error: 'timed out loading models' };
    }

    move(e) {
      this.root.classList.add('show');
      clearTimeout(this.showTimer);
      this.showTimer = setTimeout(() => this.root.classList.remove('show'), 2500);
      if (!this.active) return;
      this.mouse = { x: e.clientX, y: e.clientY };
      this.tilt();
      if (this.moveTimer) return;
      this.moveTimer = setTimeout(() => {
        this.moveTimer = null;
        this.tick();
      }, 90);
    }

    leave() {
      this.mouse = null;
      this.root.classList.remove('show');
      this.hide();
    }

    hide() {
      this.current = null;
      this.lockFace = null; // the next card gets a fresh lock-on
      this.card.style.transitionDelay = '0s';
      this.hl.classList.remove('show');
      this.card.classList.remove('show');
    }

    // the big card leans toward where the mouse sits on the hovered card
    tilt() {
      const r = this.curRect;
      if (!this.current || !r || !this.mouse) return;
      const pr = this.container.getBoundingClientRect();
      const u = Math.max(-1, Math.min(1, ((this.mouse.x - pr.left - r.x) / r.w) * 2 - 1));
      const v = Math.max(-1, Math.min(1, ((this.mouse.y - pr.top - r.y) / r.h) * 2 - 1));
      const s = this.card.style;
      s.setProperty('--ry', `${(u * 6).toFixed(2)}deg`);
      s.setProperty('--rx', `${(-v * 6).toFixed(2)}deg`);
      s.setProperty('--mx', `${((1 - u) * 50).toFixed(1)}%`);
    }

    // big-card size and spot: beside the hovered card on the roomier side
    // (sticking to the last side while it fits), shrinking to fit before
    // ever covering it; above/below when neither side has room
    place(bx, by, bw, bh, land, pr) {
      const sizeK = { s: 0.5, m: 0.64, l: 0.8 }[settings.size] || 0.64;
      const gap = 14;
      const m = 8;
      const capH = settings.showEffect ? 34 + 92 : 34; // caption (+ effect box under it)
      // the long edge is what the size means, so a sideways base or leader
      // is the same card size as an upright one, just turned
      const long = Math.min(Math.max(pr.height * sizeK, 220), 620, pr.height - m - capH);
      let w = land ? long : long * (63 / 88);
      let h = land ? long * (63 / 88) : long;
      const maxW = pr.width * (land ? 0.5 : 0.42);
      if (w > maxW) {
        h *= maxW / w;
        w = maxW;
      }
      const fit = (room, size, minK) => (room >= size ? 1 : room / size >= minK ? room / size : 0);
      const roomR = pr.width - m - (bx + bw + gap);
      const roomL = bx - gap - m;
      const prefer = this.side === 'r' || this.side === 'l' ? this.side : null;
      const order = prefer && fit(prefer === 'r' ? roomR : roomL, w, 1) ? [prefer] : roomR >= roomL ? ['r', 'l'] : ['l', 'r'];
      for (const side of order) {
        const k = fit(side === 'r' ? roomR : roomL, w, 0.6);
        if (!k) continue;
        w *= k;
        h *= k;
        const left = side === 'r' ? bx + bw + gap : bx - gap - w;
        const top = Math.max(m, Math.min(pr.height - h - capH, by + bh / 2 - h / 2));
        return { side, left, top, w, h };
      }
      const roomB = pr.height - capH - (by + bh + gap);
      const roomT = by - gap - m;
      for (const side of roomB >= roomT ? ['b', 't'] : ['t', 'b']) {
        const k = fit(side === 'b' ? roomB : roomT, h, 0.5);
        if (!k) continue;
        w *= k;
        h *= k;
        const left = Math.max(m, Math.min(pr.width - w - m, bx + bw / 2 - w / 2));
        const top = side === 'b' ? by + bh + gap : by - gap - h;
        return { side, left, top, w, h };
      }
      // a huge card on a small player: pin to the roomier edge
      const side = roomR >= roomL ? 'r' : 'l';
      return { side, left: side === 'r' ? pr.width - w - m : m, top: Math.max(m, (pr.height - h - capH) / 2), w, h };
    }

    frameKey() {
      return `${this.video.currentSrc}|${this.video.currentTime.toFixed(3)}`;
    }

    async detectFrame() {
      const v = this.video;
      const vw = v.videoWidth;
      const vh = v.videoHeight;
      if (!vw || !vh) return null;
      const key = this.frameKey();
      const now = performance.now();
      if (this.dets && (this.dets.key === key || (!v.paused && now - this.dets.time < DETECT_TTL_MS))) return this.dets;
      // full-resolution snapshot: crops for the identifier come from here
      if (!this.snap || this.snap.width !== vw || this.snap.height !== vh) {
        this.snap = el('canvas');
        this.snap.width = vw;
        this.snap.height = vh;
        this.snapCtx = this.snap.getContext('2d', { willReadFrequently: true });
      }
      const s = Math.min(DET_W / vw, DET_H / vh);
      const cw = Math.round(vw * s);
      const ch = Math.round(vh * s);
      const ox = Math.round((DET_W - cw) / 2);
      if (!this.det) {
        this.det = el('canvas');
        this.det.width = DET_W;
        this.det.height = DET_H;
        this.detCtx = this.det.getContext('2d', { willReadFrequently: true });
      }
      let image;
      try {
        this.snapCtx.drawImage(v, 0, 0, vw, vh);
        this.detCtx.fillStyle = '#000';
        this.detCtx.fillRect(0, 0, DET_W, DET_H);
        this.detCtx.imageSmoothingQuality = 'high';
        this.detCtx.drawImage(this.snap, 0, 0, vw, vh, ox, 0, cw, ch);
        image = this.det.toDataURL('image/jpeg', 0.92);
      } catch (e) {
        this.fail(tr('protected'));
        return null;
      }
      const res = await send({ op: 'detect', image });
      if (res.error) {
        this.fail(res.error);
        return null;
      }
      const boxes = res.boxes.map((b) => ({
        x0: (b.x0 - ox) / s, y0: b.y0 / s, x1: (b.x1 - ox) / s, y1: b.y1 / s, score: b.score, cls: b.cls,
      }));
      this.dets = { key, time: now, boxes, vw, vh };
      this.ids.clear();
      return this.dets;
    }

    fail(msg) {
      this.btn.classList.add('err');
      this.toast(msg, 5000);
    }

    async tick() {
      if (!this.active || !this.mouse) return;
      if (this.busy) {
        this.again = true;
        return;
      }
      this.busy = true;
      try {
        const dets = await this.detectFrame();
        if (!dets || !this.mouse) return this.hide();
        const cr = contentRect(this.video);
        const px = ((this.mouse.x - cr.left) / cr.width) * dets.vw;
        const py = ((this.mouse.y - cr.top) / cr.height) * dets.vh;
        const pad = 0.02 * dets.vh;
        let hit = null;
        for (const b of dets.boxes) {
          if (px < b.x0 - pad || px > b.x1 + pad || py < b.y0 - pad || py > b.y1 + pad) continue;
          const area = (b.x1 - b.x0) * (b.y1 - b.y0);
          if (!hit || area < hit.area) hit = { ...b, area };
        }
        if (!hit || hit.cls !== 0) {
          this.host.dataset.last = JSON.stringify({ hit: hit ? 'back' : null, px: Math.round(px), py: Math.round(py), boxes: dets.boxes.length });
          if (settings.debug) {
            const msg = hit ? tr('dbg_back') : tr('dbg_no_det', { px: Math.round(px), py: Math.round(py) });
            this.toast(msg, 3000);
          }
          return this.hide();
        }
        const key = `${Math.round(hit.x0)},${Math.round(hit.y0)},${Math.round(hit.x1)},${Math.round(hit.y1)}`;
        let res = this.ids.get(key);
        if (!res) {
          res = await this.identify(hit);
          this.ids.set(key, res);
        }
        if (!this.mouse) return this.hide();
        // the pointer may have moved on while we awaited the engine: only show
        // this card if it's still under the cursor, otherwise look again
        const cr2 = contentRect(this.video);
        const qx = ((this.mouse.x - cr2.left) / cr2.width) * dets.vw;
        const qy = ((this.mouse.y - cr2.top) / cr2.height) * dets.vh;
        if (qx < hit.x0 - pad || qx > hit.x1 + pad || qy < hit.y0 - pad || qy > hit.y1 + pad) {
          this.again = true;
          return;
        }
        this.show(hit, res, dets);
      } finally {
        this.busy = false;
        if (this.again) {
          this.again = false;
          setTimeout(() => this.tick(), 0);
        }
      }
    }

    async identify(b) {
      if (!this.crop) {
        this.crop = el('canvas');
        this.crop.width = ID_SIZE;
        this.crop.height = ID_SIZE;
        this.cropCtx = this.crop.getContext('2d', { willReadFrequently: true });
      }
      const x0 = Math.max(0, b.x0);
      const y0 = Math.max(0, b.y0);
      const x1 = Math.min(this.snap.width, b.x1);
      const y1 = Math.min(this.snap.height, b.y1);
      if (x1 - x0 < 8 || y1 - y0 < 8) return { face: -1 };
      this.cropCtx.imageSmoothingQuality = 'high';
      this.cropCtx.drawImage(this.snap, x0, y0, x1 - x0, y1 - y0, 0, 0, ID_SIZE, ID_SIZE);
      const res = await send({ op: 'identify', image: this.crop.toDataURL('image/jpeg', 0.95) });
      return res.error ? { face: -1, error: res.error } : res;
    }

    // Compute effective thresholds based on sensitivity setting.
    // 'strict'   = use calibrated values as-is (highest precision)
    // 'balanced' = relax a bit (default; helps with generic embeddings)
    // 'loose'    = very permissive (shows best guess even if not confident)
    effectiveThresholds(T) {
      const s = settings.sens || 'balanced';
      if (s === 'strict') return { score: T.score, junk: T.junk, margin: T.margin };
      if (s === 'loose')  return { score: 0.30, junk: -1,   margin: 0 };
      // balanced
      return {
        score:  Math.max(0.35, T.score  - 0.15),
        junk:   Math.max(-1,   T.junk   - 0.05),
        margin: 0,
      };
    }

    show(b, res, dets) {
      const baseT = res.thr || { score: MIN_SCORE, junk: MIN_JUNK_GAP, margin: MIN_MARGIN }; // calibrated by build_database.py
      const T = this.effectiveThresholds(baseT);
      const scoreOk  = res.face >= 0 && res.score >= T.score;
      const junkOk   = scoreOk && (res.score - res.junk) >= T.junk;
      const marginOk = junkOk  && res.margin >= T.margin;
      const ok = marginOk;
      this.host.dataset.last = JSON.stringify({
        box: [b.x0, b.y0, b.x1, b.y1].map(Math.round), ok, face: res.face,
        score: res.score, junk: res.junk, title: res.card ? `${res.card.id} ${res.card.n}` : res.top && res.top[0] ? `#${res.top[0].face}` : null,
      });
      // --- diagnostic toast ---
      if (settings.debug) {
        const fmt = (n) => (n == null ? '?' : n.toFixed(3));
        let msg;
        if (res.face < 0) {
          msg = tr('dbg_low', { score: fmt(res.score), min: fmt(T.score) });
        } else if (!scoreOk) {
          msg = tr('dbg_low', { score: fmt(res.score), min: fmt(T.score) });
        } else if (!junkOk) {
          msg = tr('dbg_junk', { gap: fmt(res.score - res.junk), min: fmt(T.junk) });
        } else if (!marginOk) {
          msg = tr('dbg_margin', { margin: fmt(res.margin), min: fmt(T.margin) });
        } else {
          const name = res.card ? res.card.n : `face #${res.face}`;
          msg = tr('dbg_ok', { name, score: fmt(res.score) });
        }
        this.toast(msg, ok ? 2500 : 4000);
      }
      const cr = contentRect(this.video);
      const pr = this.container.getBoundingClientRect();
      const sx = cr.width / dets.vw;
      const sy = cr.height / dets.vh;
      const bx = cr.left - pr.left + b.x0 * sx;
      const by = cr.top - pr.top + b.y0 * sy;
      const bw = (b.x1 - b.x0) * sx;
      const bh = (b.y1 - b.y0) * sy;
      // a new card gets the lock-on: the reticle snaps in without gliding from the last one
      const lock = ok && settings.highlight && this.lockFace !== res.face;
      if (lock) {
        this.hl.classList.remove('lock', 'show');
        void this.hl.offsetWidth;
        this.hl.classList.add('lock');
      }
      Object.assign(this.hl.style, { left: `${bx - 3}px`, top: `${by - 3}px`, width: `${bw + 6}px`, height: `${bh + 6}px` });
      this.hl.classList.toggle('show', settings.highlight && ok);
      if (!ok) {
        this.lockFace = null;
        this.card.classList.remove('show');
        return;
      }
      this.lockFace = res.face;
      const f = res.card;
      const land = false; // One Piece cards are always drawn upright
      const p = this.place(bx, by, bw, bh, land, pr);
      this.side = p.side;
      Object.assign(this.card.style, { left: `${p.left}px`, top: `${p.top}px`, width: `${p.w}px`, height: `${p.h}px` });
      this.card.classList.toggle('land', land);
      this.current = b;
      this.curRect = { x: bx, y: by, w: bw, h: bh };
      this.tilt();
      this.card.style.transitionDelay = lock ? '.14s' : '0s'; // let the reticle land first
      if (this.shownFace === res.face) {
        if (this.loadedFace === res.face) this.card.classList.add('show');
        return;
      }
      this.shownFace = res.face;
      // start the slide-in from the hovered card's side
      const from = { r: ['-18px', '0px'], l: ['18px', '0px'], b: ['0px', '-18px'], t: ['0px', '18px'] }[p.side];
      this.card.classList.add('reset');
      this.card.classList.remove('show');
      this.card.style.setProperty('--fx', from[0]);
      this.card.style.setProperty('--fy', from[1]);
      void this.card.offsetWidth;
      this.card.classList.remove('reset');
      this.curCard = f;
      this.renderInfo(f);
      // swap in only once the full image is decoded, so it never paints half-loaded
      const face = res.face;
      loadCardImage(f).then((url) => {
        if (this.shownFace !== face || !url) return;
        this.img.src = url;
        this.loadedFace = face;
        if (this.current) this.card.classList.add('show');
      });
    }

    variantLabel(f) {
      const m = /^([pr])(\d+)$/i.exec(f.v || '');
      if (!m) return f.v || '';
      const name = m[1].toLowerCase() === 'p' ? 'Alt Art' : 'Reprint'; // _pN = parallel art, _rN = reprint
      return `${name}${Number(m[2]) > 1 ? ` ${m[2]}` : ''}`;
    }

    renderInfo(f) {
      const bits = [f.id.split('_')[0], f.r].filter(Boolean);
      const vl = this.variantLabel(f);
      if (vl) bits.push(vl);
      this.cap.replaceChildren(el('b', { text: f.n }), el('span', { text: `  ·  ${bits.join(' · ')}` }));

      // effect text (the dataset only has the official English text)
      if (settings.showEffect && (f.e || f.tg)) {
        const stat = [f.k, f.c && f.c.join('/'), f.cost != null ? `Cost ${f.cost}` : '', f.p != null ? `Power ${f.p}` : '',
          f.ctr ? `Counter +${f.ctr}` : ''].filter(Boolean).join(' · ');
        const kids = [el('span', { class: 'stat', text: stat })];
        if (f.e) kids.push(document.createTextNode(f.e));
        if (f.tg) kids.push(el('span', { class: 'tg', text: `[${tr('trigger')}] ${f.tg}` }));
        this.fx.replaceChildren(...kids);
        this.fx.hidden = false;
      } else {
        this.fx.hidden = true;
      }

      if (settings.showPrice) this.loadPrice(f);
    }

    async loadPrice(f) {
      const chip = el('span', { class: 'px na', text: '…' });
      this.cap.append(chip);
      const { priceSource: source, currency } = settings;
      const key = `${f.id}|${source}|${currency}`;
      let res = priceCache.get(key);
      if (!res) {
        res = await send({ op: 'price', card: { id: f.id, b: f.b, v: f.v }, source, currency }, 12000);
        if (res && !res.error) priceCache.set(key, res);
      }
      if (this.curCard !== f || !chip.isConnected) return; // moved on to another card
      this.priceUrl = res && res.url;
      if (res && res.amount != null) {
        chip.className = res.approx ? 'px ref' : 'px';
        chip.textContent = `${res.approx ? '≈ ' : ''}${fmtMoney(res.amount, currency)}`;
        chip.title = res.approx ? `Ref. ${res.refStore}` : res.refStore || res.store;
      } else {
        chip.textContent = tr('price_na');
      }
    }
  }

  const priceCache = new Map(); // `${id}|${source}|${currency}` -> price reply

  // O: open the hovered card in the chosen store (L, J, K... belong to the players)
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'o' && e.key !== 'O') return;
    if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
    const t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    for (const lens of lenses.values()) {
      if (lens.active && lens.current && lens.curCard) {
        const f = lens.curCard;
        const base = { liga: 'https://www.ligaonepiece.com.br/?view=cards/search&card=',
          tcgplayer: 'https://www.tcgplayer.com/search/one-piece-card-game/product?q=',
          cardmarket: 'https://www.cardmarket.com/en/OnePiece/Products/Search?searchString=' }[settings.priceSource];
        if (!base) return;
        e.preventDefault();
        e.stopPropagation();
        send({ op: 'openUrl', url: base + encodeURIComponent(f.b) });
        return;
      }
    }
  }, true);

  // ------------------------------------------------------------ card images
  const imageCache = new Map(); // face url -> usable src (http or data:)

  function tryImage(src) {
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => resolve(im.naturalWidth > 0 ? src : null);
      im.onerror = () => resolve(null);
      im.src = src;
    });
  }

  async function loadCardImage(f) {
    if (imageCache.has(f.img)) return imageCache.get(f.img);
    // the official site's scan first; pages whose CSP blocks that host get the
    // same bytes fetched through the extension
    let src = await tryImage(f.img);
    if (!src) {
      const res = await send({ op: 'fetchImage', url: f.img });
      src = res.dataUrl ? await tryImage(res.dataUrl) : null;
    }
    if (src) imageCache.set(f.img, src);
    return src;
  }

  // ------------------------------------------------------------ player discovery
  const lenses = new Map(); // container -> Lens

  // Facebook, Kick and unknown sites: climb from the <video> to the highest
  // ancestor that is still about the size of the video, so the site's control
  // overlay (which receives the mouse events) lives inside our container.
  function genericContainer(v) {
    const vr = v.getBoundingClientRect();
    let c = v.parentElement;
    for (let i = 0; i < 6 && c && c.parentElement && c.parentElement !== document.body; i++) {
      const pr = c.parentElement.getBoundingClientRect();
      if (pr.width > vr.width * 1.12 || pr.height > vr.height * 1.12) break;
      c = c.parentElement;
    }
    return c;
  }

  function findPlayers() {
    const found = new Map();
    for (const v of document.querySelectorAll('video')) {
      const r = v.getBoundingClientRect();
      if (r.width < 280 || r.height < 150) continue;
      let c = null;
      if (IS_YT) {
        if (v.closest('ytd-video-preview, #inline-preview-player, ytd-thumbnail')) continue; // hover previews
        c = v.closest('.html5-video-player');
      } else if (IS_TWITCH) {
        // the outer player: Twitch's control overlay (which gets the mouse
        // events) is a sibling of .video-player__container, not inside it
        c = v.closest('[data-a-target="video-player"]') || v.closest('.video-player') || v.closest('.video-player__container');
      }
      if (!c) c = genericContainer(v);
      if (c) found.set(c, v);
    }
    return found;
  }

  // picture-in-picture: Chrome's floating video window, or the site's own mini player
  function inPip(container, video) {
    if (document.pictureInPictureElement === video) return true;
    if (IS_YT) return container.classList.contains('ytp-player-minimized') || !!container.closest('ytd-miniplayer');
    if (IS_TWITCH) return !!video.closest('[data-a-player-type="site_mini"], [data-a-player-state="mini"]');
    return false;
  }

  function scan() {
    if (!settings) return;
    const found = findPlayers();
    for (const [c, lens] of lenses) {
      if (!found.has(c) || !c.isConnected) {
        lens.destroy();
        lenses.delete(c);
      }
    }
    for (const [c, v] of found) {
      const lens = lenses.get(c);
      if (!lens) lenses.set(c, new Lens(c, v));
      else if (lens.video !== v) {
        lens.video = v; // player swapped its <video> (ads, quality switch, channel change)
        lens.dets = null;
      }
    }
    for (const lens of lenses.values()) lens.applySettings(); // follow PiP changes
  }

  // react at once instead of waiting for the next scan
  document.addEventListener('enterpictureinpicture', scan, true);
  document.addEventListener('leavepictureinpicture', scan, true);

  setInterval(scan, 1500);
  document.addEventListener('yt-navigate-finish', () => setTimeout(scan, 300));
  scan();
})();
