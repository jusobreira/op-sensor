#!/usr/bin/env python3
"""Fine-tune um MobileNetV3 com ArcFace loss nas cartas de One Piece.

Substitui o embed.onnx genérico (DINOv2-small) por um modelo treinado
especificamente para distinguir cartas de One Piece — mesmo com brilho,
perspectiva e qualidade de vídeo baixa.

Tempo estimado: ~1-1.5h numa RTX 2070.

Setup:
    pip install torch torchvision timm onnx onnxruntime pillow numpy

Uso:
    python tools/finetune_arcface.py --cache ./.card-cache --out .

    # Para verificar o resultado antes de usar:
    python tools/finetune_arcface.py --cache ./.card-cache --out . --eval-only

Depois de terminar, rode o build_database.py normalmente:
    python tools/build_database.py --punk-records ./punk-records-main --out . \\
        --physical --aug 16 --cache ./.card-cache --backbone mobilenetv3-arcface
"""
import argparse
import glob
import io
import math
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ID_SIZE = 224   # deve coincidir com engine.js


# ------------------------------------------------------------------ augment
def augment(im, rng, strong=False):
    """Augmentação para treino: simula condições de vídeo ao vivo."""
    im = im.convert("RGB")
    w, h = im.size

    # crop aleatório (simula detector impreciso)
    m = 0.08 if strong else 0.05
    l = rng.uniform(-m, m) * w
    t = rng.uniform(-m, m) * h
    r = w + rng.uniform(-m, m) * w
    b = h + rng.uniform(-m, m) * h
    im = im.crop((int(l), int(t), int(r), int(b)))

    # perspectiva (carta segurada em ângulo)
    if rng.random() < (0.7 if strong else 0.4):
        strength = rng.uniform(0.04, 0.18 if strong else 0.12)
        im = _perspective_warp(im, rng, strength)

    # rotação
    max_rot = 15 if strong else 8
    im = im.rotate(rng.uniform(-max_rot, max_rot), resample=Image.BICUBIC,
                   expand=False,
                   fillcolor=(rng.randint(0, 30), rng.randint(0, 30), rng.randint(0, 30)))

    # resolução (vídeo de baixa qualidade)
    short = rng.choice([70, 90, 120, 160, 210] if strong else [100, 140, 190, 240])
    s = short / min(im.size)
    im = im.resize((max(8, int(im.size[0] * s)), max(8, int(im.size[1] * s))),
                   Image.BILINEAR)

    # blur (foco ruim, movimento)
    if rng.random() < (0.7 if strong else 0.45):
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 2.5 if strong else 1.5)))

    # motion blur
    if strong and rng.random() < 0.3:
        im = _motion_blur(im, rng)

    # cor e exposição
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.5, 1.5))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.6, 1.4))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.5, 1.4))

    # tint quente/frio (iluminação do ambiente)
    if rng.random() < 0.5:
        arr = np.asarray(im).astype(np.float32)
        tint = np.array([rng.uniform(0.85, 1.18), rng.uniform(0.85, 1.18),
                         rng.uniform(0.85, 1.18)])
        im = Image.fromarray(np.clip(arr * tint, 0, 255).astype(np.uint8))

    # brilho holográfico
    if rng.random() < 0.5:
        im = _holo_glare(im, rng)

    # sombra de dedo
    if rng.random() < 0.35:
        im = _finger_shadow(im, rng)

    # ruído de sensor
    if rng.random() < 0.6:
        arr = np.asarray(im).astype(np.float32)
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(
            0, rng.uniform(2, 12 if strong else 8), arr.shape).astype(np.float32)
        im = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    # JPEG compression
    quality = rng.randint(25 if strong else 35, 85)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    im = Image.open(io.BytesIO(buf.getvalue()))

    # redimensiona para 224x224
    return im.resize((ID_SIZE, ID_SIZE), Image.BILINEAR).convert("RGB")


def _perspective_warp(im, rng, strength):
    w, h = im.size
    pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts_dst = pts_src.copy()
    for i in range(4):
        pts_dst[i, 0] += rng.uniform(-strength, strength) * w
        pts_dst[i, 1] += rng.uniform(-strength, strength) * h
    # 8 coeficientes de transformação projetiva via numpy
    A = []
    b_vec = []
    for (xs, ys), (xd, yd) in zip(pts_src, pts_dst):
        A += [[xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys],
              [0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys]]
        b_vec += [xd, yd]
    A = np.array(A, dtype=np.float64)
    b_arr = np.array(b_vec, dtype=np.float64)
    try:
        coeffs = np.linalg.solve(A, b_arr)
    except np.linalg.LinAlgError:
        return im
    coeffs = np.append(coeffs, 1).reshape(3, 3)
    # aplica via PIL transform
    c = np.linalg.inv(coeffs).flatten()[:8]
    return im.transform((w, h), Image.PERSPECTIVE, c, Image.BICUBIC)


def _motion_blur(im, rng, max_px=6):
    px = rng.randint(2, max_px)
    if rng.random() < 0.5:
        k = [1/px]*px + [0]*(px-1)
        kernel = ImageFilter.Kernel((2*px-1, 1) if False else (px, 1),
                                    [1]*px, scale=px)
        # usa convolução manual simples
        arr = np.asarray(im).astype(np.float32)
        from scipy.ndimage import uniform_filter1d
        axis = 1 if rng.random() < 0.5 else 0
        arr = uniform_filter1d(arr, size=px, axis=axis)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return im


def _holo_glare(im, rng):
    arr = np.asarray(im).astype(np.float32)
    h, w = arr.shape[:2]
    angle = rng.uniform(0, math.pi)
    cx, cy = rng.uniform(0.1, 0.9)*w, rng.uniform(0.1, 0.9)*h
    streak_w = rng.uniform(0.05, 0.25) * min(w, h)
    intensity = rng.uniform(40, 130)
    tint = np.array([rng.uniform(0.8, 1.5), rng.uniform(0.8, 1.5),
                     rng.uniform(0.8, 1.5)], dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    dist = abs((xs - cx)*math.sin(angle) - (ys - cy)*math.cos(angle))
    mask = np.exp(-(dist / streak_w)**2)[:, :, None]
    arr = np.clip(arr + mask * intensity * tint, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _finger_shadow(im, rng):
    arr = np.asarray(im).astype(np.float32)
    h, w = arr.shape[:2]
    side = rng.randint(0, 3)
    strength = rng.uniform(0.3, 0.65)
    if side == 0:
        t = np.linspace(1, 0, h)[:, None, None]
        mask = (1 - np.exp(-t * rng.uniform(3, 8))) * strength
    elif side == 1:
        t = np.linspace(0, 1, h)[:, None, None]
        mask = (1 - np.exp(-(1 - t) * rng.uniform(3, 8))) * strength
        mask = np.flip(mask, axis=0)
    elif side == 2:
        t = np.linspace(1, 0, w)[None, :, None]
        mask = (1 - np.exp(-t * rng.uniform(3, 8))) * strength
    else:
        t = np.linspace(0, 1, w)[None, :, None]
        mask = (1 - np.exp(-(1 - t) * rng.uniform(3, 8))) * strength
        mask = np.flip(mask, axis=1)
    arr = arr * (1 - np.clip(mask, 0, 0.7))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ dataset
class CardDataset:
    """Carrega imagens do cache e aplica augmentação on-the-fly."""

    def __init__(self, paths, labels, augment_fn, views_per_card=2):
        self.paths = paths
        self.labels = labels
        self.augment_fn = augment_fn
        self.views = views_per_card
        self.rng = random.Random(42)

    def __len__(self):
        return len(self.paths) * self.views

    def __getitem__(self, idx):
        import torch
        card_idx = idx // self.views
        path = self.paths[card_idx]
        label = self.labels[card_idx]
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            im = Image.new("RGB", (ID_SIZE, ID_SIZE), (128, 128, 128))
        strong = (idx % self.views) > 0   # primeira view é suave, restantes são fortes
        aug = self.augment_fn(im, self.rng, strong=strong)
        arr = np.asarray(aug, dtype=np.float32) / 255.0
        # normalização ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        return x, label


# ------------------------------------------------------------------ ArcFace
class ArcFaceHead(object):
    """ArcFace loss (margin-based softmax para metric learning).

    Referência: Deng et al. (2019) ArcFace: Additive Angular Margin Loss

    Força o modelo a aprender discriminação angular entre classes:
    cartas visualmente parecidas ficam separadas no espaço de embedding.
    """
    pass


def arcface_loss(embeddings, labels, weight, s=16.0, m=0.3):
    """CosFace/ArcFace híbrido: margem aditiva no cosseno (mais estável que ArcFace puro).

    embeddings: [N, D] L2-normalizado
    labels:     [N] inteiros
    weight:     [num_classes, D] (centros das classes, normalizados internamente)
    s: scale — 16 é mais seguro para muitas classes (3000+)
    m: margem aditiva no cosseno (CosFace) — 0.3 converge bem
    """
    import torch
    import torch.nn.functional as F

    # cosine similarity entre embeddings e centros normalizados
    cos_theta = F.linear(embeddings, F.normalize(weight, dim=-1))  # [N, C]
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

    # CosFace: subtrai margem do cosseno da classe correta (mais estável que ArcFace)
    one_hot = torch.zeros_like(cos_theta)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    logits = cos_theta - one_hot * m  # [N, C]

    return F.cross_entropy(logits * s, labels)


# ------------------------------------------------------------------ modelo
def build_backbone(arch="resnet50"):
    """Backbone pré-treinado sem cabeça de classificação.

    arch: 'resnet50' (padrão, ~25M params, melhor qualidade)
          'mobilenetv3' (mais leve, ~5M params, mais rápido)
    """
    import timm, torch
    name = {
        "resnet50":     "resnet50.a1_in1k",
        "mobilenetv3":  "mobilenetv3_large_100.ra_in1k",
    }.get(arch, arch)
    model = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        feat_dim = model(dummy).shape[1]
    print(f"  backbone: {name}  feat_dim={feat_dim}")
    return model, feat_dim


class EmbedNet(object):
    pass


import torch
import torch.nn as nn


class EmbedModel(nn.Module):
    """Backbone + projeção linear → embedding L2-normalizado."""

    def __init__(self, backbone, in_dim, embed_dim=512):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x já vem normalizado pelo Dataset; aqui apenas extrai features
        f = self.backbone(x)
        f = self.proj(f)
        return nn.functional.normalize(f, dim=-1)

    def forward_raw(self, image):
        """Para exportar ONNX: recebe imagem 0..1 float."""
        x = (image - self.mean) / self.std
        return self.forward(x)


# ------------------------------------------------------------------ treino
def train(args):
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("AVISO: CUDA não encontrado, usando CPU — vai demorar muito mais!")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    # Encontra todas as imagens no cache
    print("Carregando lista de cartas do cache...")
    img_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    all_paths = []
    for ext in img_exts:
        all_paths.extend(glob.glob(os.path.join(args.cache, "**", ext), recursive=True))
    if not all_paths:
        sys.exit(f"Nenhuma imagem encontrada em {args.cache}. Rode o build_database.py primeiro.")

    # Agrupa por carta base (nome do arquivo sem extensão = ID da carta)
    # Formato esperado: OP01-001.jpg, OP01-001_p1.jpg (paralelo), etc.
    import re
    def card_key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        # OP01-001, OP01-001_p1, OP01-001_sec → mesma carta base OP01-001
        return re.sub(r'[_-](p\d+|sec|alt\d*|promo\d*)$', '', name, flags=re.I)

    from collections import defaultdict
    by_card = defaultdict(list)
    for p in all_paths:
        by_card[card_key(p)].append(p)

    # Filtra cartas com pelo menos 1 imagem e cria labels
    cards = sorted(by_card.keys())
    num_classes = len(cards)
    print(f"  {num_classes} cartas únicas, {len(all_paths)} imagens no cache")

    if num_classes < 10:
        sys.exit("Poucas cartas encontradas. Verifique o --cache.")

    card_to_label = {c: i for i, c in enumerate(cards)}
    paths, labels = [], []
    for card, imgs in by_card.items():
        lbl = card_to_label[card]
        for p in imgs:
            paths.append(p)
            labels.append(lbl)

    # Split treino/validação (90/10 por carta)
    rng = random.Random(args.seed)
    val_cards = set(rng.sample(cards, max(1, num_classes // 10)))
    tr_paths, tr_labels, va_paths, va_labels = [], [], [], []
    for p, l in zip(paths, labels):
        ck = card_key(p)
        if ck in val_cards:
            va_paths.append(p); va_labels.append(l)
        else:
            tr_paths.append(p); tr_labels.append(l)

    print(f"  treino: {len(tr_paths)} imgs | validação: {len(va_paths)} imgs")

    # Datasets e DataLoaders
    tr_ds = CardDataset(tr_paths, tr_labels, augment, views_per_card=args.views)
    va_ds = CardDataset(va_paths, va_labels, augment, views_per_card=1)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=(device == "cuda"))
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=(device == "cuda"))

    # Modelo
    import timm
    backbone, feat_dim = build_backbone(args.arch)
    model = EmbedModel(backbone, feat_dim, embed_dim=args.dim).to(device)

    # Centros ArcFace (aprendidos junto com o modelo)
    arc_weight = nn.Parameter(torch.randn(num_classes, args.dim, device=device))
    nn.init.xavier_uniform_(arc_weight.data)

    # Otimizador: todos os parâmetros com o mesmo LR (backbone precisa se adaptar)
    optim = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.proj.parameters(),     "lr": args.lr},
        {"params": [arc_weight],                "lr": args.lr},
    ], weight_decay=1e-4)

    # Scheduler: cosine annealing com LR mínimo de 1e-6
    total_steps = args.epochs * len(tr_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-6)

    # Warmup nas primeiras 3 épocas
    warmup_steps = 3 * len(tr_dl)

    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    ckpt_path = os.path.join(args.out, "models", "arcface_checkpoint.pt")
    best_acc = 0.0
    best_epoch = 0

    print(f"\nIniciando treino: {args.epochs} épocas, batch={args.batch}, dim={args.dim}")
    print(f"ArcFace: scale={args.scale}, margin={args.margin}\n")

    for epoch in range(1, args.epochs + 1):
        # --- treino ---
        model.train()
        tr_loss = 0.0
        t0 = time.time()
        for step, (x, y) in enumerate(tr_dl):
            x, y = x.to(device), y.to(device)

            # warmup linear
            global_step = (epoch - 1) * len(tr_dl) + step
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optim.param_groups:
                    pg["lr"] = pg["lr"] * lr_scale  # simplificado

            optim.zero_grad()
            emb = model(x)
            loss = arcface_loss(emb, y, arc_weight, s=args.scale, m=args.margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + [arc_weight], max_norm=5.0)
            optim.step()
            scheduler.step()
            tr_loss += loss.item()

        tr_loss /= len(tr_dl)
        elapsed = time.time() - t0

        # --- validação rápida: kNN dentro do conjunto de validação ---
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()

            va_embs, va_lbls = [], []
            with torch.no_grad():
                for x, y in va_dl:
                    e = model(x.to(device)).cpu()
                    va_embs.append(e)
                    va_lbls.extend(y.tolist())
            va_embs = torch.cat(va_embs)          # [V, D]
            va_lbls_t = torch.tensor(va_lbls)     # [V]

            # Para cada amostra, acha o vizinho mais próximo EXCLUINDO ela mesma
            sims = va_embs @ va_embs.T            # [V, V]
            sims.fill_diagonal_(-2.0)             # exclui auto-similaridade
            preds = sims.argmax(dim=1)            # índice do vizinho mais próximo
            acc = (va_lbls_t[preds] == va_lbls_t).float().mean().item()

            flag = " ← melhor!" if acc > best_acc else ""
            print(f"Época {epoch:3d}/{args.epochs}  loss={tr_loss:.4f}  "
                  f"val-top1={acc:.1%}  ({elapsed:.0f}s){flag}")

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "arc_weight": arc_weight.data,
                    "num_classes": num_classes,
                    "dim": args.dim,
                    "acc": acc,
                    "arch": args.arch,
                }, ckpt_path)
        else:
            print(f"Época {epoch:3d}/{args.epochs}  loss={tr_loss:.4f}  ({elapsed:.0f}s)")

    print(f"\nMelhor val-top1: {best_acc:.1%} na época {best_epoch}")
    print(f"Checkpoint salvo em: {ckpt_path}")
    return ckpt_path, best_acc


# ------------------------------------------------------------------ export ONNX
def export(args, ckpt_path):
    import torch

    print("\nExportando para ONNX...")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    import timm
    arch = ckpt.get("arch", "resnet50")
    backbone, feat_dim = build_backbone(arch)
    model = EmbedModel(backbone, feat_dim, embed_dim=ckpt["dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Wrapper para exportar: recebe imagem 0..1, devolve embedding normalizado
    class ExportWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, image):
            return self.inner.forward_raw(image)

    wrap = ExportWrap(model)
    dummy = torch.rand(4, 3, ID_SIZE, ID_SIZE)
    out_path = os.path.join(args.out, "models", "embed.onnx")

    with torch.no_grad():
        ref = wrap(dummy).numpy()

    torch.onnx.export(
        wrap, dummy, out_path,
        input_names=["image"], output_names=["embedding"],
        dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}},
        opset_version=17, dynamo=False,
    )

    # Verifica
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"image": dummy.numpy()})[0]
        err = float(np.abs(out - ref).max())
        print(f"  onnx vs torch max abs diff: {err:.5f}")
        if err > 0.002:
            print("  ! diferença maior que esperado; verifique a exportação")
    except ImportError:
        print("  (instale onnxruntime para verificar a exportação)")

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  embed.onnx salvo: {out_path} ({size_mb:.1f} MB)")
    print("\n✓ Pronto! Agora rode o build_database.py para gerar os protótipos:")
    print("  python tools/build_database.py --punk-records ./punk-records-main \\")
    print("      --out . --physical --aug 16 --cache ./.card-cache \\")
    print("      --backbone mobilenetv3-arcface")
    print("\n  (O build vai usar o embed.onnx que acabou de ser gerado)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Fine-tune ArcFace para OP Sensor")
    ap.add_argument("--cache",     required=True, help="Pasta .card-cache com as imagens")
    ap.add_argument("--out",       default=".",   help="Onde salvar o embed.onnx (padrão: .)")
    ap.add_argument("--epochs",    type=int, default=60, help="Épocas de treino (padrão: 60)")
    ap.add_argument("--batch",     type=int, default=64, help="Batch size (padrão: 64; reduza se der OOM)")
    ap.add_argument("--dim",       type=int, default=512, help="Dimensão do embedding (padrão: 512)")
    ap.add_argument("--lr",        type=float, default=1e-3, help="Learning rate base (padrão: 1e-3)")
    ap.add_argument("--scale",     type=float, default=16.0, help="CosFace scale s (padrão: 16)")
    ap.add_argument("--margin",    type=float, default=0.3,  help="CosFace margin m no cosseno (padrão: 0.3)")
    ap.add_argument("--views",     type=int, default=3, help="Augmentações por carta por época (padrão: 3)")
    ap.add_argument("--workers",   type=int, default=4, help="DataLoader workers (padrão: 4)")
    ap.add_argument("--eval-every",type=int, default=5, help="Avaliar a cada N épocas (padrão: 5)")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--arch",      default="resnet50", choices=["resnet50","mobilenetv3"],
                                   help="Backbone: resnet50 (padrão) ou mobilenetv3")
    ap.add_argument("--eval-only", action="store_true", help="Só avalia o checkpoint existente, não treina")
    ap.add_argument("--export-only",action="store_true", help="Só exporta o checkpoint para ONNX")
    args = ap.parse_args()

    # Verifica dependências
    try:
        import torch
        import timm
    except ImportError as e:
        sys.exit(f"Dependência faltando: {e}\nRode: pip install torch torchvision timm onnx onnxruntime")

    ckpt_path = os.path.join(args.out, "models", "arcface_checkpoint.pt")

    if args.export_only:
        if not os.path.exists(ckpt_path):
            sys.exit(f"Checkpoint não encontrado: {ckpt_path}")
        export(args, ckpt_path)
        return

    if args.eval_only:
        if not os.path.exists(ckpt_path):
            sys.exit(f"Checkpoint não encontrado: {ckpt_path}")
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"Checkpoint: época {ckpt['epoch']}, val-top1 = {ckpt['acc']:.1%}")
        export(args, ckpt_path)
        return

    print("=" * 60)
    print("OP Sensor — Fine-tuning ArcFace")
    print("=" * 60)
    ckpt_path, acc = train(args)
    export(args, ckpt_path)


if __name__ == "__main__":
    main()
