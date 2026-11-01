// OP Sensor service worker.
//
// Content scripts can't run the recognition models themselves (page CSPs, no
// threads), so they send frames here and we relay them to an offscreen document
// that hosts ONNX Runtime. The offscreen document is created on first use and
// closed after a quiet spell.

import { lookupPrice, STORE_HOSTS } from './prices.js';

const OFFSCREEN_URL = 'offscreen.html';
const IMAGE_HOSTS = ['https://en.onepiece-cardgame.com/'];
let creating = null;

async function hasOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });
  return contexts.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  if (!creating) {
    creating = chrome.offscreen.createDocument({
      url: OFFSCREEN_URL,
      reasons: ['WORKERS'],
      justification: 'Runs the card detection and recognition models locally.',
    }).finally(() => { creating = null; });
  }
  await creating;
}

const IDLE_CLOSE_MIN = 15;

async function touch() {
  await chrome.storage.session.set({ engineLastUse: Date.now() });
  const alarm = await chrome.alarms.get('engine-idle');
  if (!alarm) chrome.alarms.create('engine-idle', { periodInMinutes: 5 });
}

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== 'engine-idle') return;
  const { engineLastUse = 0 } = await chrome.storage.session.get('engineLastUse');
  if (Date.now() - engineLastUse < IDLE_CLOSE_MIN * 60 * 1000) return;
  if (await hasOffscreen()) await chrome.offscreen.closeDocument();
  chrome.alarms.clear('engine-idle');
});

async function toEngine(msg) {
  if (msg.op !== 'status') touch();
  await ensureOffscreen();
  for (let attempt = 0; attempt < 40; attempt++) {
    try {
      const res = await chrome.runtime.sendMessage({ ...msg, to: 'engine' });
      if (res !== undefined) return res;
    } catch (e) {
      // offscreen page still booting: its listener isn't registered yet
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  return { error: 'engine did not respond' };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || msg.to === 'engine') return false; // meant for the offscreen page

  if (msg.op === 'detect' || msg.op === 'identify' || msg.op === 'status') {
    toEngine(msg).then(sendResponse, (e) => sendResponse({ error: String(e) }));
    return true;
  }

  if (msg.op === 'price') {
    lookupPrice(msg)
      .then(sendResponse, (e) => sendResponse({ error: String(e) }));
    return true;
  }

  if (msg.op === 'openUrl') {
    let host = '';
    try { host = new URL(String(msg.url)).hostname; } catch { /* invalid */ }
    if (!STORE_HOSTS.includes(host)) {
      sendResponse({ error: 'host not allowed' });
      return false;
    }
    chrome.tabs.create({ url: msg.url });
    sendResponse({ ok: true });
    return false;
  }

  if (msg.op === 'fetchImage') {
    // Fallback for pages whose CSP blocks direct <img> loads from the card
    // host. Only card images from hosts in host_permissions are fetched.
    const url = String(msg.url || '');
    if (!IMAGE_HOSTS.some((h) => url.startsWith(h))) {
      sendResponse({ error: 'not a card image host' });
      return false;
    }
    fetch(url)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const buf = new Uint8Array(await r.arrayBuffer());
        let bin = '';
        for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
        sendResponse({ dataUrl: `data:${r.headers.get('content-type') || 'image/png'};base64,${btoa(bin)}` });
      })
      .catch((e) => sendResponse({ error: String(e) }));
    return true;
  }
  return false;
});
