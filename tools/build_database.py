#!/usr/bin/env python3
"""Build the OP Sensor card database and embedding model.

Run this on YOUR computer (it needs internet + a few GB of disk). It:
  1. reads the card list from a clone of https://github.com/buhbbl/punk-records
  2. downloads every card image from the official One Piece Card Game site
  3. exports a pretrained image-embedding model to ONNX  -> models/embed.onnx
  4. embeds each card (with video-like augmentations)    -> data/protos.bin
  5. writes the card metadata                            -> data/faces.json
  6. calibrates the match thresholds on held-out augmentations and prints
     a top-1 accuracy estimate

Setup (Python 3.10+):
  pip install torch torchvision transformers timm onnx onnxruntime pillow numpy truststore certifi
  # optional, for --fp16:  pip install onnxconverter-common

Usage:
  git clone --depth 1 https://github.com/buhbbl/punk-records
  python tools/build_database.py --punk-records ./punk-records --out .
  python tools/build_database.py --punk-records ./punk-records --out . --metadata-only   # quick check, no ML

Re-run it whenever a new set comes out (git pull the punk-records clone first);
downloaded images are cached in --cache, so only new cards are fetched.
"""
import argparse
import concurrent.futures as cf
import glob
import io
import json
import os
import random
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ID_SIZE = 224  # must match engine/engine.js
UA = {"User-Agent": "Mozilla/5.0 (OP-Sensor database builder)"}


def make_ssl_context(insecure=False):
    """HTTPS context for the image downloads. Order: the operating system's own
    certificate store (handles antivirus/corporate HTTPS inspection) -> certifi ->
    Python's default. --insecure disables verification (last resort)."""
    import ssl
    if insecure:
        print("  ! --insecure: certificate verification is OFF", file=sys.stderr)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        import truststore  # pip install truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    try:
        import certifi  # pip install certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CTX = None


# ----------------------------------------------------------------- card list
def load_cards(root, lang):
    lang_dir = os.path.join(root, lang)
    packs = {}
    pj = os.path.join(lang_dir, "packs.json")
    if os.path.exists(pj):
        with open(pj, encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.values() if isinstance(raw, dict) else raw
        for p in items:
            packs[str(p["id"])] = (p.get("title_parts") or {}).get("label") or (p.get("title_parts") or {}).get("title") or str(p["id"])

    cards = []
    per_card = sorted(glob.glob(os.path.join(lang_dir, "cards", "*", "*.json")))
    if per_card:
        for p in per_card:
            with open(p, encoding="utf-8") as f:
                cards.append(json.load(f))
    else:  # one JSON array per pack
        for p in sorted(glob.glob(os.path.join(lang_dir, "cards", "*.json")) or glob.glob(os.path.join(lang_dir, "data", "*.json"))):
            with open(p, encoding="utf-8") as f:
                cards.extend(json.load(f))
    if not cards:
        sys.exit(f"No cards found under {lang_dir}. Is --punk-records the repo root?")

    faces, seen = [], set()
    for c in cards:
        cid = c["id"]
        if cid in seen:
            continue
        seen.add(cid)
        base, _, var = cid.partition("_")
        url = c.get("img_full_url") or ""
        stem = re.search(r"/([^/?]+)\.(?:png|jpg|jpeg|webp)", url)
        if not url or not stem:
            continue
        rel = url.split("/images/cardlist/card/", 1)[-1]  # "ST01-001.png?261016"
        face = {
            "id": cid, "b": base, "n": c.get("name") or base, "r": c.get("rarity") or "",
            "k": c.get("category") or "", "c": c.get("colors") or [],
            "i": rel, "s": packs.get(str(c.get("pack_id")), ""),
        }
        if var:
            face["v"] = var
        for key, short in (("cost", "cost"), ("power", "p"), ("counter", "ctr")):
            if c.get(key) is not None:
                face[short] = c[key]
        if c.get("effect"):
            face["e"] = c["effect"]
        if c.get("trigger"):
            face["tg"] = c["trigger"]
        if c.get("types"):
            face["ty"] = c["types"]
        face["_url"] = url
        faces.append(face)
    faces.sort(key=lambda f: (f["b"], f.get("v", "")))
    return faces


# ------------------------------------------------------------------- images
def fetch(face, cache):
    path = os.path.join(cache, re.sub(r"[^A-Za-z0-9_.-]", "_", face["id"]) + ".png")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    for attempt in range(4):
        try:
            req = urllib.request.Request(face["_url"], headers=UA)
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
                data = r.read()
            Image.open(io.BytesIO(data)).verify()
            with open(path, "wb") as f:
                f.write(data)
            return path
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  ! {face['id']}: {err}", file=sys.stderr)
    return None


def download_all(faces, cache, workers):
    os.makedirs(cache, exist_ok=True)
    paths = [None] * len(faces)
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, f, cache): i for i, f in enumerate(faces)}
        done = 0
        for fut in cf.as_completed(futs):
            paths[futs[fut]] = fut.result()
            done += 1
            if done % 200 == 0 or done == len(faces):
                print(f"  images {done}/{len(faces)}")
    return paths


# ------------------------------------------------------------ augmentations
def squash(im):
    """What the extension does to a detected card crop: stretch to 224x224."""
    return im.convert("RGB").resize((ID_SIZE, ID_SIZE), Image.BICUBIC)


def _perspective_warp(im, rng, strength=0.12):
    """Simulate a card held at an angle: apply a random perspective transform."""
    try:
        import PIL.ImageTransform  # noqa: F401 — check it's available
        w, h = im.size
        s = strength
        # Four corner offsets: each corner shifts by a random fraction of its dimension
        def rp(scale): return rng.uniform(-scale, scale)
        coeffs_input = [
            (rp(s)*w, rp(s)*h),          # top-left
            (w + rp(s)*w, rp(s)*h),      # top-right
            (w + rp(s)*w, h + rp(s)*h),  # bottom-right
            (rp(s)*w, h + rp(s)*h),      # bottom-left
        ]
        # PIL PERSPECTIVE needs the 8 coefficients of the projective transform
        # Compute them from the 4 corner mappings (src = corners of the image)
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = coeffs_input
        # solve A * coeff = b for the 8 perspective coefficients
        A, b = [], []
        for (sx, sy), (dx, dy) in zip(src, dst):
            A += [[sx, sy, 1, 0, 0, 0, -dx*sx, -dx*sy],
                  [0, 0, 0, sx, sy, 1, -dy*sx, -dy*sy]]
            b += [dx, dy]
        try:
            coeffs = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64)).tolist()
            return im.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        except np.linalg.LinAlgError:
            return im
    except Exception:
        return im


def _add_noise(im, rng, amount=0.04):
    """Camera sensor noise / grain."""
    arr = np.asarray(im).astype(np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**30)).normal(0, amount * 255, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _motion_blur(im, rng, max_px=5):
    """Horizontal or vertical motion blur (camera shake / card movement)."""
    px = rng.randint(2, max_px)
    if rng.random() < 0.5:
        kernel = ImageFilter.Kernel((px * 2 + 1, 1), [1] * (px * 2 + 1), scale=px * 2 + 1)
    else:
        kernel = ImageFilter.Kernel((1, px * 2 + 1), [1] * (px * 2 + 1), scale=px * 2 + 1)
    try:
        return im.filter(kernel)
    except Exception:
        return im.filter(ImageFilter.GaussianBlur(px * 0.5))


def _holo_glare(im, rng):
    """Simulate holographic/foil glare on physical cards: a bright, often
    rainbow-tinted streak sweeping across the card."""
    arr = np.asarray(im).astype(np.float32)
    h, w = arr.shape[:2]
    # random streak angle
    angle = rng.uniform(0, np.pi)
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
    xs = np.arange(w)[None, :] - cx
    ys = np.arange(h)[:, None] - cy
    # distance from streak line
    dist = abs(xs * np.sin(angle) - ys * np.cos(angle))
    width = rng.uniform(0.05, 0.25) * min(w, h)
    streak = np.exp(-(dist / width) ** 2) * rng.uniform(40, 130)
    # slight colour tint (rainbow effect)
    tint = np.array([rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3)])
    glare = streak[:, :, None] * tint[None, None, :]
    return Image.fromarray(np.clip(arr + glare, 0, 255).astype(np.uint8))


def _finger_shadow(im, rng):
    """A dark curved shadow at the edge — as if someone is holding the card."""
    arr = np.asarray(im).astype(np.float32)
    h, w = arr.shape[:2]
    side = rng.randint(0, 3)
    t = np.linspace(0, 1, w if side in (0, 2) else h)
    shadow = (1 - np.exp(-t * rng.uniform(3, 8))) * rng.uniform(0.3, 0.65)
    if side == 0:    # top
        mask = shadow[None, :, None] * np.ones((h, 1, 1))
        mask = mask.T.reshape(1, w, 1) * np.linspace(1, 0, h)[:, None, None]
    elif side == 1:  # bottom
        mask = np.linspace(0, 1, h)[:, None, None] * rng.uniform(0.3, 0.6)
        mask = mask * (1 - np.linspace(0, 1, w)[None, :, None] * 0)
    elif side == 2:  # left
        mask = np.linspace(1, 0, w)[None, :, None] * rng.uniform(0.3, 0.6)
    else:            # right
        mask = np.linspace(0, 1, w)[None, :, None] * rng.uniform(0.3, 0.6)
    arr = arr * (1 - np.clip(mask, 0, 0.7))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def video_like(im, rng, strong=False, physical=None):
    """Mimic a card seen in a stream. physical=True simulates a real physical card
    (holographic glare, finger shadows, perspective, camera noise); physical=False
    simulates digital card art on screen; physical=None chooses randomly."""
    im = im.convert("RGB")
    w, h = im.size

    # decide physical vs digital
    if physical is None:
        physical = rng.random() < 0.55  # slightly bias toward physical (harder case)

    # sloppy detector box: crop/extend each side by a few percent
    m = 0.06 if strong else 0.04
    l = rng.uniform(-m, m) * w
    t = rng.uniform(-m, m) * h
    r = w + rng.uniform(-m, m) * w
    b = h + rng.uniform(-m, m) * h
    im = im.crop((int(l), int(t), int(r), int(b)))

    # perspective warp (physical cards at an angle)
    if physical and rng.random() < 0.75:
        strength = rng.uniform(0.06, 0.20 if strong else 0.14)
        im = _perspective_warp(im, rng, strength)
    elif rng.random() < 0.3:  # digital: slight tilt only
        im = _perspective_warp(im, rng, 0.04)

    # rotation
    max_rot = 12 if (strong and physical) else (8 if strong else 5)
    im = im.rotate(rng.uniform(-max_rot, max_rot), resample=Image.BICUBIC, expand=False,
                   fillcolor=(rng.randint(0, 30), rng.randint(0, 30), rng.randint(0, 30)))

    # resolution (physical cards on stream: often 80-180 px; digital: can be sharper)
    if physical:
        short = rng.choice([70, 85, 100, 130, 170, 220] if strong else [90, 120, 160, 210])
    else:
        short = rng.choice([100, 130, 170, 220, 280] if strong else [150, 200, 260, 320])
    s = short / min(im.size)
    im = im.resize((max(8, int(im.size[0] * s)), max(8, int(im.size[1] * s))), Image.BILINEAR)

    # blur: physical may have focus issues on top of compression
    blur_p = 0.75 if physical else 0.5
    if rng.random() < blur_p:
        max_blur = 2.2 if (strong and physical) else (1.6 if strong else 1.1)
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, max_blur)))

    # motion blur (card being moved / camera shake)
    if physical and rng.random() < 0.25:
        im = _motion_blur(im, rng)

    # colour & exposure
    if physical:
        # physical cards often under mixed/warm lighting
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.55, 1.5))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.65, 1.4))
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.55, 1.4))
        # warm/cool tint from room lighting
        if rng.random() < 0.5:
            arr = np.asarray(im).astype(np.float32)
            tint = np.array([rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)])
            im = Image.fromarray(np.clip(arr * tint, 0, 255).astype(np.uint8))
    else:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.7, 1.3))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.75, 1.25))
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.7, 1.25))

    # holographic / foil glare (physical cards only)
    if physical and rng.random() < 0.55:
        im = _holo_glare(im, rng)
    elif not physical and rng.random() < 0.25:  # screen glare
        arr = np.asarray(im).astype(np.float32)
        gx = np.linspace(-1, 1, arr.shape[1])[None, :, None]
        gy = np.linspace(-1, 1, arr.shape[0])[:, None, None]
        cx, cy = rng.uniform(-1, 1), rng.uniform(-1, 1)
        glare = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) * rng.uniform(1.5, 5)) * rng.uniform(20, 70)
        im = Image.fromarray(np.clip(arr + glare, 0, 255).astype(np.uint8))

    # finger/hand shadow at edge (physical cards being held)
    if physical and rng.random() < 0.40:
        im = _finger_shadow(im, rng)

    # camera sensor noise (physical more grainy)
    if rng.random() < (0.6 if physical else 0.3):
        amount = rng.uniform(0.02, 0.07 if physical else 0.04)
        im = _add_noise(im, rng, amount)

    # JPEG compression
    quality = rng.randint(28 if (strong and physical) else 35, 82)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return squash(Image.open(io.BytesIO(buf.getvalue())))


def junk_images(paths, rng, n):
    """Things that are NOT a whole card: zoomed art patches, UI-ish gradients,
    noise, flat colours, and a card cut in half."""
    out = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.45:  # patch of a random card, 20-60 % of it
            im = Image.open(rng.choice(paths)).convert("RGB")
            w, h = im.size
            pw, ph = int(w * rng.uniform(0.2, 0.6)), int(h * rng.uniform(0.2, 0.6))
            x, y = rng.randint(0, w - pw), rng.randint(0, h - ph)
            out.append(video_like(im.crop((x, y, x + pw, y + ph)), rng, strong=True))
        elif kind < 0.6:  # half card
            im = Image.open(rng.choice(paths)).convert("RGB")
            w, h = im.size
            box = rng.choice([(0, 0, w, h // 2), (0, h // 2, w, h), (0, 0, w // 2, h), (w // 2, 0, w, h)])
            out.append(video_like(im.crop(box), rng, strong=True))
        elif kind < 0.75:  # gradient / flat
            a = np.random.RandomState(rng.randint(0, 1 << 30))
            c0, c1 = a.randint(0, 255, 3), a.randint(0, 255, 3)
            t = np.linspace(0, 1, ID_SIZE)[None, :, None]
            arr = (c0 * (1 - t) + c1 * t) * np.ones((ID_SIZE, 1, 1))
            out.append(Image.fromarray(arr.astype(np.uint8)))
        elif kind < 0.9:  # noise
            a = np.random.RandomState(rng.randint(0, 1 << 30))
            out.append(Image.fromarray(a.randint(0, 255, (ID_SIZE, ID_SIZE, 3)).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0, 6))))
        else:  # two cards' worth of clutter: a collage of patches
            canvas = Image.new("RGB", (ID_SIZE, ID_SIZE))
            for qx in (0, 1):
                for qy in (0, 1):
                    im = Image.open(rng.choice(paths)).convert("RGB")
                    w, h = im.size
                    canvas.paste(im.crop((0, 0, w, h)).resize((ID_SIZE // 2, ID_SIZE // 2)), (qx * ID_SIZE // 2, qy * ID_SIZE // 2))
            out.append(canvas)
    return out


# ------------------------------------------------------------------ model
def build_model(backbone):
    import torch
    import torch.nn as nn

    class Wrap(nn.Module):
        """image float [N,3,224,224] in 0..1 -> L2-normalised embedding."""

        def __init__(self, body, mean, std, kind):
            super().__init__()
            self.body, self.kind = body, kind
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, image):
            x = (image - self.mean) / self.std
            if self.kind == "hf":
                f = self.body(pixel_values=x).last_hidden_state[:, 0]  # CLS token
            else:
                f = self.body(x)
            return torch.nn.functional.normalize(f, dim=-1)

    imn = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if backbone == "dinov2-small":
        from transformers import Dinov2Model
        body = Dinov2Model.from_pretrained("facebook/dinov2-small")
        model = Wrap(body, *imn, "hf")
    elif backbone == "mobilenetv3":
        import timm
        body = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=True, num_classes=0)
        model = Wrap(body, *imn, "timm")
    else:
        sys.exit(f"unknown backbone {backbone}")
    return model.eval()


def export_onnx(model, path, fp16):
    import torch

    dummy = torch.rand(4, 3, ID_SIZE, ID_SIZE)
    with torch.no_grad():
        ref = model(dummy).numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(
        model, dummy, path, input_names=["image"], output_names=["embedding"],
        dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False,
    )
    if fp16:
        import onnx
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(onnx.load(path), keep_io_types=True)
        onnx.save(m, path)
    try:  # sanity check: ONNX must agree with PyTorch
        import onnxruntime as ort
        out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"image": dummy.numpy()})[0]
        err = float(np.abs(out - ref).max())
        print(f"  onnx vs torch max abs diff: {err:.5f}")
        if err > (0.02 if fp16 else 0.002):
            print("  ! difference is larger than expected; check the export", file=sys.stderr)
    except ImportError:
        print("  (install onnxruntime to verify the export)")
    print(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


class Embedder:
    def __init__(self, model):
        import torch
        self.torch = torch
        self.model = model
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev)

    def __call__(self, imgs, bs=64):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), bs):
                arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs[i:i + bs]]).transpose(0, 3, 1, 2)
                out.append(self.model(torch.from_numpy(arr).to(self.dev)).cpu().numpy())
        return np.concatenate(out)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def quantize(protos):
    """int8, one scale per row. The engine divides by 127 and re-normalises, so a
    per-row scale only buys precision (max |v| -> 127)."""
    scale = 127.0 / np.maximum(np.abs(protos).max(axis=1, keepdims=True), 1e-9)
    return np.clip(np.round(protos * scale), -127, 127).astype(np.int8)


def dequantize(q):
    return unit(q.astype(np.float32))


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--punk-records", required=True, help="path to a clone of buhbbl/punk-records")
    ap.add_argument("--lang", default="english", help="punk-records language folder (default: english)")
    ap.add_argument("--out", default=".", help="extension root (data/ and models/ are written inside)")
    ap.add_argument("--cache", default="./.card-cache", help="where card images are cached")
    ap.add_argument("--backbone", default="dinov2-small", choices=["dinov2-small", "mobilenetv3"])
    ap.add_argument("--aug", type=int, default=16, help="augmented views per card used for its prototype (default 16; more = better but slower)")
    ap.add_argument("--physical", action="store_true", help="bias augmentations heavily toward physical card simulation (glare, perspective, shadows)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only use the first N cards (for a quick test)")
    ap.add_argument("--fp16", action="store_true", help="store model weights as fp16 (half the size)")
    ap.add_argument("--metadata-only", action="store_true", help="only list cards and write faces.json without protos (no ML, no downloads)")
    ap.add_argument("--insecure", action="store_true", help="skip HTTPS certificate verification for image downloads (last resort)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    global SSL_CTX
    SSL_CTX = make_ssl_context(args.insecure)
    rng = random.Random(args.seed)

    print("1/6 reading card list")
    faces = load_cards(args.punk_records, args.lang)
    if args.limit:
        faces = faces[:args.limit]
    n_par = sum(1 for f in faces if "v" in f)
    print(f"  {len(faces)} printings ({len(faces) - n_par} base, {n_par} parallel/alt art)")
    if args.metadata_only:
        print(json.dumps({k: v for k, v in faces[0].items() if k != "_url"}, ensure_ascii=False, indent=1))
        return

    print("2/6 downloading images")
    paths = download_all(faces, args.cache, args.workers)
    keep = [i for i, p in enumerate(paths) if p]
    if len(keep) < len(faces):
        print(f"  ! {len(faces) - len(keep)} images failed and are skipped")
    faces = [faces[i] for i in keep]
    paths = [paths[i] for i in keep]

    print(f"3/6 exporting {args.backbone} to ONNX")
    model = build_model(args.backbone)
    export_onnx(model, os.path.join(args.out, "models", "embed.onnx"), args.fp16)
    emb = Embedder(model)

    physical_bias = 1.0 if args.physical else None  # None = random per augmentation
    print("4/6 computing prototypes")
    protos = []
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        # physical=None lets each augmentation randomly decide digital vs physical
        # with --physical flag we force physical=True for all augmentations
        views = [squash(im)] + [video_like(im, rng, physical=True if args.physical else None) for _ in range(args.aug)]
        v = emb(views)
        # the clean scan counts double: it is what the user compares against
        protos.append(unit(unit(v[1:]).mean(0) + 2 * v[0]))
        if (i + 1) % 250 == 0 or i + 1 == len(paths):
            print(f"  {i + 1}/{len(paths)}")
    protos = np.stack(protos)
    junk = unit(unit(emb(junk_images(paths, rng, 300))).mean(0))
    all_protos = np.concatenate([protos, junk[None]])
    q = quantize(all_protos)

    print("5/6 calibrating thresholds on held-out strong augmentations")
    P = dequantize(q)
    rng_test = random.Random(args.seed + 1)
    idx = rng_test.sample(range(len(paths)), min(600, len(paths)))
    test = [video_like(Image.open(paths[i]), rng_test, strong=True) for i in idx]
    te = emb(test)
    sims = te @ P.T
    sims_cards = sims[:, :-1]
    order = np.argsort(-sims_cards, axis=1)
    keys = np.array([f["b"] for f in faces])
    scores, margins, gaps, hit_face, hit_card = [], [], [], 0, 0
    for r, i in enumerate(idx):
        top = order[r]
        t1 = top[0]
        # runner-up among *different* cards (printings of one card don't compete)
        t2 = next(j for j in top[1:] if keys[j] != keys[t1])
        scores.append(sims_cards[r, t1])
        margins.append(sims_cards[r, t1] - sims_cards[r, t2])
        gaps.append(sims_cards[r, t1] - sims[r, -1])
        hit_face += int(t1 == i)
        hit_card += int(keys[t1] == keys[i])
    jt = emb(junk_images(paths, random.Random(args.seed + 2), 300))
    junk_top = (jt @ P[:-1].T).max(1)
    scores, margins, gaps = map(np.array, (scores, margins, gaps))
    thr_score = float(np.clip((np.percentile(junk_top, 95) + np.percentile(scores, 10)) / 2, 0.3, 0.85))
    thresholds = {
        "score": round(thr_score, 3),
        "junk": round(float(np.clip(np.percentile(gaps, 10) * 0.5, 0.0, 0.1)), 3),
        "margin": round(float(np.clip(np.percentile(margins, 10) * 0.5, 0.005, 0.08)), 3),
    }
    print(f"  simulated top-1: card {hit_card / len(idx):.1%} (exact printing {hit_face / len(idx):.1%})")
    print(f"  true score p10 {np.percentile(scores, 10):.3f} | non-card top score p95 {np.percentile(junk_top, 95):.3f}")
    print(f"  thresholds: {thresholds}")
    print("  (simulation, not real streams: expect lower accuracy on real video)")

    print("6/6 writing data/")
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    with open(os.path.join(args.out, "data", "protos.bin"), "wb") as f:
        f.write(q.tobytes())
    built = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "dim": int(all_protos.shape[1]), "img_base": "https://en.onepiece-cardgame.com/images/cardlist/card/",
        "built": built, "backbone": args.backbone, "thresholds": thresholds,
        "faces": [{k: v for k, v in f.items() if k != "_url"} for f in faces],
    }
    with open(os.path.join(args.out, "data", "faces.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(args.out, "data", "status.json"), "w", encoding="utf-8") as f:
        json.dump({"faces": len(faces), "built": built, "backbone": args.backbone}, f)
    print("done. Reload the extension in chrome://extensions.")


if __name__ == "__main__":
    main()
