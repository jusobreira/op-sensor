const DEFAULTS = {
  enabled: true, size: 'm', highlight: true, corner: 'right',
  lang: 'auto', showEffect: true, showPrice: true, priceSource: 'liga', currency: 'BRL',
  sens: 'balanced', debug: false,
};
const ALL_SITES_ID = 'ops-all-sites';
const ALL_ORIGINS = ['https://*/*', 'http://*/*'];

let state = { ...DEFAULTS };
let lastStatus = null;

const $ = (id) => document.getElementById(id);
const t = (key, vars) => OPS.t(state.lang, key, vars);
const save = (patch) => { state = { ...state, ...patch }; return chrome.storage.sync.set(patch); };

function render() {
  document.documentElement.lang = OPS.resolve(state.lang);
  for (const n of document.querySelectorAll('[data-i18n]')) n.textContent = t(n.dataset.i18n);
  const corner = t('corner_name', { v: t(state.corner) });
  // the "how" text carries markup from our own dictionary (no user input)
  $('how').innerHTML = t('how', { corner });
  paintStatus();
  paintData();
}

function toggle(id, key) {
  const box = $(id);
  box.checked = !!state[key];
  box.addEventListener('change', () => save({ [key]: box.checked }));
}

function picker(key, onPick) {
  const seg = $(key);
  const paint = () => seg.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === state[key]));
  paint();
  seg.addEventListener('click', (e) => {
    const v = e.target.dataset && e.target.dataset.v;
    if (!v) return;
    save({ [key]: v });
    paint();
    if (onPick) onPick();
  });
}

function select(id, key, onPick) {
  const sel = $(id);
  sel.value = state[key];
  sel.addEventListener('change', () => {
    save({ [key]: sel.value });
    if (onPick) onPick();
  });
}

async function initAllSites() {
  const box = $('allSites');
  const note = $('sitesNote');
  const scripts = await chrome.scripting.getRegisteredContentScripts({ ids: [ALL_SITES_ID] }).catch(() => []);
  box.checked = scripts.length > 0;
  box.addEventListener('change', async () => {
    note.hidden = true;
    try {
      if (box.checked) {
        const granted = await chrome.permissions.request({ origins: ALL_ORIGINS });
        if (!granted) throw new Error(t('all_sites_denied'));
        await chrome.scripting.registerContentScripts([{
          id: ALL_SITES_ID,
          js: ['i18n.js', 'content/sensor.js'],
          matches: ALL_ORIGINS,
          allFrames: true,
          runAt: 'document_idle',
          persistAcrossSessions: true,
        }]);
      } else {
        await chrome.scripting.unregisterContentScripts({ ids: [ALL_SITES_ID] });
        await chrome.permissions.remove({ origins: ALL_ORIGINS }).catch(() => {});
      }
    } catch (e) {
      box.checked = !box.checked;
      note.textContent = String(e && e.message ? e.message : e);
      note.hidden = false;
    }
  });
}

function paintStatus() {
  const el = $('status');
  const st = lastStatus;
  el.className = 'status';
  if (!st) {
    el.textContent = t('status_none');
  } else if (st.ready) {
    el.className = 'status ok';
    const ms = st.detectMs ? t('status_ms', { ms: st.detectMs + (st.identifyMs || 0) }) : '';
    el.textContent = t('status_ready', { faces: st.faces.toLocaleString(), threads: st.threads, ms });
  } else if (st.error) {
    el.className = 'status bad';
    el.textContent = t('status_error', { err: st.error });
  } else {
    el.textContent = t('status_loading');
  }
}

let dataInfo = null;
function paintData() {
  const el = $('dataLine');
  el.hidden = !dataInfo;
  if (!dataInfo) return;
  const date = dataInfo.built ? new Date(dataInfo.built).toLocaleDateString(OPS.resolve(state.lang) === 'pt' ? 'pt-BR' : 'en-US') : '?';
  el.textContent = t('data_line', { cards: dataInfo.faces.toLocaleString(), date });
}

async function loadDataInfo() {
  try {
    const r = await fetch(chrome.runtime.getURL('data/status.json'));
    if (r.ok) dataInfo = await r.json();
  } catch { /* database not built yet */ }
}

async function init() {
  state = { ...DEFAULTS, ...(await chrome.storage.sync.get(DEFAULTS)) };
  toggle('enabled', 'enabled');
  toggle('highlight', 'highlight');
  toggle('showEffect', 'showEffect');
  toggle('showPrice', 'showPrice');
  toggle('debug', 'debug');
  picker('size');
  picker('corner', render);
  picker('currency');
  picker('sens');
  select('lang', 'lang', render);
  select('priceSource', 'priceSource');
  await loadDataInfo();
  render();
  initAllSites();

  // Only report on the recognizer if it is already running; opening the
  // popup shouldn't spin up the models.
  const ctx = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
  if (!ctx.length) return;
  lastStatus = (await chrome.runtime.sendMessage({ op: 'status' })) || null;
  paintStatus();
}

init();
