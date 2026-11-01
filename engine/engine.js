// OP Sensor recognition engine: card detector + card identifier on ONNX Runtime
// (WASM). Kept free of extension APIs so it can be exercised from a plain page.
//
// Data contract (produced by tools/build_database.py):
//   data/faces.json  { dim, img_base, built, faces: [{ id, b, n, r, ... }] }
//   data/protos.bin  int8 [faces.length + 1, dim], scale 1/127; the last row is
//                    the "not a card" prototype
//   models/embed.onnx  input  image     float32 [N,3,224,224]  (RGB, 0..1)
//                      output embedding float32 [N,dim]        (L2-normalised)
//   models/det.onnx    input  image float32 [1,3,384,640]; outputs hm, wh, off

import * as ort from '../lib/ort/ort.wasm.min.mjs';

export const DET_W = 640;
export const DET_H = 384;
export const STRIDE = 8;
export const ID_SIZE = 224;

const DET_THRESH = 0.3;
const NEEDS_BUILD = 'Card database missing. Run tools/build_database.py and copy its output into data/ and models/ (see README).';

export class Engine {
  constructor(baseUrl) {
    this.base = baseUrl;
    this.ready = false;
    this.error = null;
    this.stats = { detect: [], identify: [] };
  }

  async load() {
    try {
      ort.env.wasm.wasmPaths = this.base + 'lib/ort/';
      const cores = navigator.hardwareConcurrency || 2;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, cores - 1)) : 1;
      const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all',
        extra: { session: { disable_quant_qdq: '1' } } };

      const getJson = async (p) => {
        const r = await fetch(this.base + p);
        if (!r.ok) throw new Error(NEEDS_BUILD);
        return r.json();
      };
      const getBuf = async (p) => {
        const r = await fetch(this.base + p);
        if (!r.ok) throw new Error(NEEDS_BUILD);
        return r.arrayBuffer();
      };

      const [det, emb, meta, protoBuf] = await Promise.all([
        getBuf('models/det.onnx').then((b) => ort.InferenceSession.create(new Uint8Array(b), opts)),
        getBuf('models/embed.onnx').then((b) => ort.InferenceSession.create(new Uint8Array(b), opts)),
        getJson('data/faces.json'),
        getBuf('data/protos.bin'),
      ]);
      this.det = det;
      this.emb = emb;
      this.faces = meta.faces;
      this.meta = meta;
      for (const f of this.faces) f.img = meta.img_base + f.i;
      this.dim = meta.dim;
      this.junk = meta.faces.length;

      const q = new Int8Array(protoBuf);
      const n = q.length / this.dim;
      if (n !== this.faces.length + 1) {
        throw new Error(`protos.bin has ${n} rows but faces.json has ${this.faces.length} faces (+1 junk row expected). Rebuild the database.`);
      }
      this.protos = new Float32Array(q.length);
      for (let r = 0; r < n; r++) {
        let ss = 0;
        for (let d = 0; d < this.dim; d++) {
          const v = q[r * this.dim + d] / 127;
          this.protos[r * this.dim + d] = v;
          ss += v * v;
        }
        const inv = 1 / Math.sqrt(ss || 1);
        for (let d = 0; d < this.dim; d++) this.protos[r * this.dim + d] *= inv;
      }
      // printings of the same card (base art, parallels) count as one when
      // judging how clearly the best match beats the runner-up
      this.cardKey = this.faces.map((f) => f.b);
      this.canvas = new OffscreenCanvas(DET_W, DET_H);
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
      this.threads = ort.env.wasm.numThreads;
      this.ready = true;
    } catch (e) {
      this.error = String(e && e.message ? e.message : e);
      throw e;
    }
  }

  status() {
    const avg = (a) => (a.length ? Math.round(a.reduce((s, v) => s + v, 0) / a.length) : null);
    return {
      ready: this.ready,
      error: this.error,
      faces: this.faces ? this.faces.length : 0,
      built: this.meta ? this.meta.built || null : null,
      threads: this.threads || 0,
      isolated: !!self.crossOriginIsolated,
      detectMs: avg(this.stats.detect),
      identifyMs: avg(this.stats.identify),
    };
  }

  _pixels(img, w, h) {
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas = new OffscreenCanvas(w, h);
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.drawImage(img, 0, 0, w, h);
    return this.ctx.getImageData(0, 0, w, h).data;
  }

  _stat(key, ms) {
    const a = this.stats[key];
    a.push(ms);
    if (a.length > 30) a.shift();
  }

  /** img: bitmap already letterboxed to 640x384. Returns boxes in input pixels. */
  async detect(img) {
    const t0 = performance.now();
    const px = this._pixels(img, DET_W, DET_H);
    const plane = DET_W * DET_H;
    const x = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < plane; i++, p += 4) {
      x[i] = px[p] / 255;
      x[i + plane] = px[p + 1] / 255;
      x[i + 2 * plane] = px[p + 2] / 255;
    }
    const out = await this.det.run({ image: new ort.Tensor('float32', x, [1, 3, DET_H, DET_W]) });
    const hm = out.hm.data;
    const wh = out.wh.data;
    const off = out.off.data;
    const [, C, H, W] = out.hm.dims;
    const HW = H * W;
    const boxes = [];
    for (let c = 0; c < C; c++) {
      for (let i = 0; i < HW; i++) {
        const s = hm[c * HW + i];
        if (s < DET_THRESH) continue;
        const gx = i % W;
        const gy = (i / W) | 0;
        const w = Math.exp(wh[i]) * STRIDE;
        const h = Math.exp(wh[HW + i]) * STRIDE;
        const cx = (gx + off[i]) * STRIDE;
        const cy = (gy + off[HW + i]) * STRIDE;
        boxes.push({ x0: cx - w / 2, y0: cy - h / 2, x1: cx + w / 2, y1: cy + h / 2, score: s, cls: c });
      }
    }
    const kept = nms(boxes, 0.6);
    this._stat('detect', performance.now() - t0);
    return kept;
  }

  /** img: bitmap of one card crop (any size; drawn to 224x224). */
  async identify(img) {
    const t0 = performance.now();
    const S = ID_SIZE;
    const px = this._pixels(img, S, S);
    const plane = S * S;
    const x = new Float32Array(4 * 3 * plane);
    // the crop is tried at all four quarter-turns: rested characters, sideways
    // cameras and the opponent's upside-down side of the table
    for (let k = 0; k < 4; k++) {
      const base = k * 3 * plane;
      for (let y = 0; y < S; y++) {
        for (let xx = 0; xx < S; xx++) {
          let sy, sx;
          if (k === 0) { sy = y; sx = xx; }
          else if (k === 1) { sy = S - 1 - xx; sx = y; }
          else if (k === 2) { sy = S - 1 - y; sx = S - 1 - xx; }
          else { sy = xx; sx = S - 1 - y; }
          const p = (sy * S + sx) * 4;
          const o = y * S + xx;
          x[base + o] = px[p] / 255;
          x[base + plane + o] = px[p + 1] / 255;
          x[base + 2 * plane + o] = px[p + 2] / 255;
        }
      }
    }
    const out = await this.emb.run({ image: new ort.Tensor('float32', x, [4, 3, S, S]) });
    const e = out.embedding.data;
    const D = this.dim;
    const N = this.protos.length / D;
    // Pick the quarter-turn whose best *card* is strongest, then judge that
    // card against the junk prototype for the same turn.
    const sims = new Float32Array(4 * N);
    let bestRot = 0, bestCard = -2;
    for (let k = 0; k < 4; k++) {
      const eo = k * D;
      for (let r = 0; r < N; r++) {
        let s = 0;
        const po = r * D;
        for (let d = 0; d < D; d++) s += this.protos[po + d] * e[eo + d];
        sims[k * N + r] = s;
        if (r !== this.junk && s > bestCard) { bestCard = s; bestRot = k; }
      }
    }
    const scores = sims.subarray(bestRot * N, (bestRot + 1) * N);
    const order = Array.from(scores.keys()).filter((r) => r !== this.junk)
      .sort((a, b) => scores[b] - scores[a]).slice(0, 60);
    const top = [];
    const seen = new Set();
    for (const r of order) {
      if (seen.has(this.cardKey[r])) continue;
      seen.add(this.cardKey[r]);
      top.push({ face: r, score: +scores[r].toFixed(4) });
      if (top.length >= 5) break;
    }
    this._stat('identify', performance.now() - t0);
    const face = top[0].face;
    return {
      face,
      card: this.faces[face],
      score: top[0].score,
      thr: this.meta.thresholds || null,
      rot: bestRot,
      junk: +scores[this.junk].toFixed(4),
      margin: top.length > 1 ? +(top[0].score - top[1].score).toFixed(4) : 1,
      top,
    };
  }
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0));
  const iy = Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
  const inter = ix * iy;
  const ua = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
  return inter / Math.max(ua, 1e-6);
}

function nms(boxes, thr) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) {
    if (keep.every((k) => iou(k, b) < thr)) keep.push(b);
  }
  return keep;
}
