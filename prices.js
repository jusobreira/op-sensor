// Price lookup + store links for OP Sensor (runs in the service worker).
//
// Reality check on sources:
//  - TCGplayer (USD): no open API. We read the community mirror optcgapi.com,
//    which republishes TCGplayer market prices. Its schema is not formally
//    documented, so the parser below is defensive. EXPERIMENTAL.
//  - Liga One Piece (BRL) and Cardmarket (EUR): no public price API. For these
//    we show the TCGplayer price converted to the chosen currency, labelled as a
//    reference, and the "L" shortcut opens the real store page.
//
// To plug in a better source, add a function to PRICE_PROVIDERS that returns
// { usd: number } (or { native: {amount, currency} }) for a card.

export const STORES = {
  liga: {
    name: 'Liga One Piece',
    url: (id) => `https://www.ligaonepiece.com.br/?view=cards/search&card=${encodeURIComponent(id)}`,
  },
  tcgplayer: {
    name: 'TCGplayer',
    url: (id) => `https://www.tcgplayer.com/search/one-piece-card-game/product?q=${encodeURIComponent(id)}&view=grid`,
  },
  cardmarket: {
    name: 'Cardmarket',
    url: (id) => `https://www.cardmarket.com/en/OnePiece/Products/Search?searchString=${encodeURIComponent(id)}`,
  },
};

export const STORE_HOSTS = ['www.ligaonepiece.com.br', 'www.tcgplayer.com', 'www.cardmarket.com'];

const PRICE_TTL = 6 * 3600 * 1000;
const RATE_TTL = 6 * 3600 * 1000;

async function cacheGet(key, ttl) {
  const o = await chrome.storage.local.get(key);
  const e = o[key];
  return e && Date.now() - e.t < ttl ? e.v : undefined;
}
function cacheSet(key, v) {
  return chrome.storage.local.set({ [key]: { t: Date.now(), v } });
}

export async function getRates() {
  const hit = await cacheGet('rates', RATE_TTL);
  if (hit) return hit;
  try {
    const r = await fetch('https://open.er-api.com/v6/latest/USD');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    const rates = { USD: 1, BRL: j.rates.BRL, EUR: j.rates.EUR };
    if (!rates.BRL || !rates.EUR) throw new Error('rates missing');
    await cacheSet('rates', rates);
    return rates;
  } catch (e) {
    const stale = (await chrome.storage.local.get('rates')).rates;
    return stale ? stale.v : null;
  }
}

const num = (x) => {
  const n = typeof x === 'string' ? parseFloat(x.replace(',', '.')) : x;
  return Number.isFinite(n) && n > 0 ? n : null;
};

// optcgapi: GET /api/sets/card/<ID>/ -> array of printings (base + parallels)
const PRICE_PROVIDERS = {
  async optcgapi(card) {
    const r = await fetch(`https://optcgapi.com/api/sets/card/${encodeURIComponent(card.b)}/`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    let j = await r.json();
    if (!Array.isArray(j)) j = j && Array.isArray(j.results) ? j.results : [j];
    const isPar = (x) => /parallel|alt/i.test(`${x.card_name || ''} ${x.card_image || ''} ${x.card_image_id || ''}`)
      || /_p\d/i.test(String(x.card_image_id || x.card_image || ''));
    const want = !!card.v;
    const pick = j.find((x) => isPar(x) === want) || j[0];
    if (!pick) return null;
    const usd = num(pick.market_price) ?? num(pick.inventory_price);
    return usd ? { usd } : null;
  },
};

export async function lookupPrice({ card, source, currency }) {
  const store = STORES[source] || STORES.tcgplayer;
  const out = { url: store.url(card.b), store: store.name, source, currency };
  const key = `price:${card.id}`;
  let usd = await cacheGet(key, PRICE_TTL);
  if (usd === undefined) {
    usd = null;
    try {
      const p = await PRICE_PROVIDERS.optcgapi(card);
      usd = p ? p.usd : null;
    } catch (e) {
      out.error = String(e && e.message ? e.message : e);
    }
    if (!out.error) await cacheSet(key, usd);
  }
  if (usd == null) return out;
  const rates = await getRates();
  const rate = rates && rates[currency];
  if (!rate) return out;
  out.amount = usd * rate;
  out.approx = source !== 'tcgplayer'; // converted reference, not the chosen store's own price
  out.refStore = 'TCGplayer';
  return out;
}
