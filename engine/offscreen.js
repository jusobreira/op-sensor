// Offscreen document: hosts the recognition engine and answers the requests
// the service worker relays from content scripts.

import { Engine } from './engine.js';

const engine = new Engine(chrome.runtime.getURL(''));
const loading = engine.load().catch((e) => console.error('OP Sensor engine failed to load', e));

async function decode(dataUrl) {
  const blob = await (await fetch(dataUrl)).blob();
  return createImageBitmap(blob);
}

async function handle(msg) {
  if (msg.op === 'status') return engine.status();
  await loading;
  if (!engine.ready) return { error: engine.error || 'engine not ready' };
  const bmp = await decode(msg.image);
  try {
    if (msg.op === 'detect') return { boxes: await engine.detect(bmp) };
    if (msg.op === 'identify') return await engine.identify(bmp);
  } finally {
    bmp.close();
  }
  return { error: `unknown op ${msg.op}` };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || msg.to !== 'engine') return false;
  handle(msg).then(sendResponse, (e) => sendResponse({ error: String(e && e.message ? e.message : e) }));
  return true;
});
