#!/usr/bin/env python3
"""
sd_run_motivation_collages.py
-----------------------------
SD-1.5 version of run_motivation_collages.py.

Motivation experiment: visualize how different ways of modifying the initial
SD-1.5 latent noise (in the DCT domain, on the K lowest radial frequencies)
change the final image. Builds 11 collages:

  Alpha:     A1, A2, A3      (DCT_low *= alpha)
  Base/c:    B1, B2, B3      (DCT_low += c)
  Affine v1: V1a, V1b        (alpha * DCT_low + c, scalar c)
  Affine v2: W1, W2, W3      (per-coefficient a_i, c_i)

Each image is labeled with its parameter value + ImageReward.

Run:
    python sd_run_motivation_collages.py
    python sd_run_motivation_collages.py --skip_ir   # faster, no ImageReward
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn, idctn

from sd_common import load_pipe, generate_from_latent, get_radial, DEVICE
from scorers import load_scorer, score_one

PROMPT = "a portrait of a man"
SEED = 42
K = 10
LH, LW = 64, 64
C = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default="results/sd_motivation1")
    p.add_argument("--prompt", type=str, default=PROMPT,
                   help="text prompt (default: portrait of a man)")
    p.add_argument("--scorer", type=str, default="hpsv3",
                   help="hpsv3 | imagereward | none  (label under each image)")
    p.add_argument("--skip_ir", action="store_true",
                   help="alias for --scorer none (no scoring, faster)")
    p.add_argument("--thumb", type=int, default=256)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# DCT modification
# ──────────────────────────────────────────────────────────────

def get_low_idx(k):
    r = get_radial(LH, LW)
    return np.argsort(r)[:k]


def apply_mod(z_np, method, params, low_idx):
    """Apply a DCT-domain modification and renormalize per-channel norm."""
    z_mod = z_np.copy().astype(np.float64)
    orig_norms = np.linalg.norm(z_mod.reshape(C, -1), axis=1)
    for ch in range(C):
        flat = dctn(z_mod[ch], norm="ortho").flatten()
        if method == "alpha":
            flat[low_idx] *= params["alpha"]
        elif method == "base":
            flat[low_idx] += params["c"]
        elif method == "affine_v1":
            flat[low_idx] = params["alpha"] * flat[low_idx] + params["c"]
        elif method == "affine_v2":
            flat[low_idx] = params["a"] * flat[low_idx] + params["c"]
        z_mod[ch] = idctn(flat.reshape(LH, LW), norm="ortho")
    new_norms = np.linalg.norm(z_mod.reshape(C, -1), axis=1)
    for ch in range(C):
        if new_norms[ch] > 1e-8:
            z_mod[ch] *= orig_norms[ch] / new_norms[ch]
    return z_mod.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Collage
# ──────────────────────────────────────────────────────────────

def load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_collage(images, labels, title, thumb, n_cols=None):
    n = len(images)
    if n_cols is None:
        n_cols = n
    n_rows = (n + n_cols - 1) // n_cols

    pad = 10
    label_h = 36
    title_h = 48
    cell_w = thumb + pad
    cell_h = thumb + label_h + pad

    canvas_w = cell_w * n_cols + pad
    canvas_h = cell_h * n_rows + title_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    f_title = load_font(18)
    f_label = load_font(13)

    draw.text((pad, 8), title, fill=(30, 30, 30), font=f_title)

    for i, (img, lbl) in enumerate(zip(images, labels)):
        row = i // n_cols
        col = i % n_cols
        x = pad + col * cell_w
        y = title_h + row * cell_h
        canvas.paste(img.resize((thumb, thumb), Image.LANCZOS), (x, y))
        draw.text((x, y + thumb + 4), lbl, fill=(60, 60, 60), font=f_label)

    return canvas


# ──────────────────────────────────────────────────────────────
# All collages
# ──────────────────────────────────────────────────────────────

def run_all(pipe, scorer, z_base_np, low_idx, save_dir, thumb):

    def gen_image(params_dict, method):
        z_mod = apply_mod(z_base_np, method, params_dict, low_idx)
        img = generate_from_latent(pipe, z_mod, PROMPT)
        return img, score_one(scorer, img, PROMPT)

    def save(collage, name):
        collage.save(save_dir / f"{name}.png")
        print(f"  ok {name}.png")

    # ── A1: Alpha sweep ──────────────────────────────────────
    print("\n[A1] Alpha sweep...")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0]
    imgs, lbls = [], []
    for a in alphas:
        img, ir = gen_image({"alpha": a}, "alpha")
        imgs.append(img)
        lbls.append(f"a={a:.2f}\ns={ir:.3f}")
    save(make_collage(imgs, lbls,
        "A1 - Alpha sweep: DCT_low *= alpha", thumb), "A1_alpha_sweep")

    # ── A2: Alpha + IR ───────────────────────────────────────
    lbls2 = [f"a={alphas[i]:.2f}  s={score_one(scorer, imgs[i], PROMPT):.3f}"
             if scorer is not None else f"a={alphas[i]:.2f}"
             for i in range(len(imgs))]
    save(make_collage(imgs, lbls2,
        "A2 - Alpha sweep + ImageReward", thumb), "A2_alpha_with_ir")

    # ── A3: Alpha low vs high ────────────────────────────────
    low_a = [a for a in alphas if a <= 1.0]
    high_a = [a for a in alphas if a >= 1.0]
    imgs_low = [imgs[alphas.index(a)] for a in low_a]
    imgs_high = [imgs[alphas.index(a)] for a in high_a]
    row1 = make_collage(imgs_low, [f"a={a}" for a in low_a],
                        "A3 - Alpha LOW (attenuate low freq)", thumb)
    row2 = make_collage(imgs_high, [f"a={a}" for a in high_a],
                        "        HIGH (boost low freq)", thumb)
    combined = Image.new("RGB",
        (max(row1.width, row2.width), row1.height + row2.height + 8),
        (245, 245, 245))
    combined.paste(row1, (0, 0))
    combined.paste(row2, (0, row1.height + 8))
    save(combined, "A3_alpha_low_vs_high")

    # ── B1: Base DC only ─────────────────────────────────────
    print("\n[B1] Base - DC only sweep...")
    imgs_b1, lbls_b1 = [], []
    for t in [-3, -2, -1, 0, 1, 2, 3]:
        c = np.zeros(K); c[0] = t
        img, ir = gen_image({"c": c}, "base")
        imgs_b1.append(img)
        lbls_b1.append(f"c[0]={t}\ns={ir:.3f}")
    save(make_collage(imgs_b1, lbls_b1,
        "B1 - Base: shift only DC (coef 0)", thumb), "B1_base_dc_only")

    # ── B2: Base random direction ────────────────────────────
    print("\n[B2] Base - random direction sweep...")
    rng = np.random.RandomState(0)
    c_rand = rng.randn(K).astype(np.float32)
    c_rand /= np.linalg.norm(c_rand)
    imgs_b2, lbls_b2 = [], []
    for t in [-2, -1, 0, 1, 2]:
        img, ir = gen_image({"c": t * c_rand}, "base")
        imgs_b2.append(img)
        lbls_b2.append(f"t={t}\ns={ir:.3f}")
    save(make_collage(imgs_b2, lbls_b2,
        "B2 - Base: random direction  c = t * c_rand", thumb), "B2_base_random_dir")

    # ── B3: Base per-coefficient ─────────────────────────────
    print("\n[B3] Base - per-coefficient...")
    imgs_b3, lbls_b3 = [], []
    for ki in range(K):
        for t in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
            c = np.zeros(K); c[ki] = t
            img, ir = gen_image({"c": c}, "base")
            imgs_b3.append(img)
            lbls_b3.append(f"c[{ki}]={t}\ns={ir:.3f}")
    save(make_collage(imgs_b3, lbls_b3,
        "B3 - Base: each coef separately  (rows=coef, cols=t)",
        thumb, n_cols=3), "B3_base_per_coef")

    # ── V1a: Affine v1 grid ──────────────────────────────────
    print("\n[V1a] Affine v1 - grid alpha x c...")
    imgs_v1a, lbls_v1a = [], []
    t_c_v1 = [-1, 0, 1]
    for a in [0.5, 1.0, 1.5, 2.0]:
        for t in t_c_v1:
            c = np.full(K, t, dtype=np.float32)
            img, ir = gen_image({"alpha": a, "c": c}, "affine_v1")
            imgs_v1a.append(img)
            lbls_v1a.append(f"a={a} c={t}\ns={ir:.3f}")
    save(make_collage(imgs_v1a, lbls_v1a,
        "V1a - Affine v1 grid  (rows=alpha, cols=c)  alpha*DCT_low+c",
        thumb, n_cols=len(t_c_v1)), "V1a_affine_v1_grid")

    # ── V1b: Affine v1 two rows ──────────────────────────────
    print("\n[V1b] Affine v1 - two rows...")
    imgs_v1b, lbls_v1b = [], []
    for a in [0.25, 0.5, 1.0, 1.5, 2.0]:
        img, ir = gen_image({"alpha": a, "c": np.zeros(K)}, "affine_v1")
        imgs_v1b.append(img)
        lbls_v1b.append(f"a={a} c=0\ns={ir:.3f}")
    for t in [-2, -1, 0, 1, 2]:
        c = np.full(K, t, dtype=np.float32)
        img, ir = gen_image({"alpha": 1.0, "c": c}, "affine_v1")
        imgs_v1b.append(img)
        lbls_v1b.append(f"a=1 c={t}\ns={ir:.3f}")
    save(make_collage(imgs_v1b, lbls_v1b,
        "V1b - Affine v1: row1 fix c=0 vary alpha | row2 fix alpha=1 vary c",
        thumb, n_cols=5), "V1b_affine_v1_two_rows")

    # ── W1: Affine v2 a_0 ────────────────────────────────────
    print("\n[W1] Affine v2 - a_0 sweep...")
    imgs_w1, lbls_w1 = [], []
    for a0 in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        a = np.ones(K); a[0] = a0
        img, ir = gen_image({"a": a, "c": np.zeros(K)}, "affine_v2")
        imgs_w1.append(img)
        lbls_w1.append(f"a[0]={a0}\ns={ir:.3f}")
    save(make_collage(imgs_w1, lbls_w1,
        "W1 - Affine v2: vary only a[0] (DC)", thumb), "W1_affine_v2_a0")

    # ── W2: Affine v2 grid a_0 x a_1 ─────────────────────────
    print("\n[W2] Affine v2 - grid a_0 x a_1...")
    a01_vals = [0.5, 1.0, 1.5, 2.0]
    imgs_w2, lbls_w2 = [], []
    for a0 in a01_vals:
        for a1 in a01_vals:
            a = np.ones(K); a[0] = a0; a[1] = a1
            img, ir = gen_image({"a": a, "c": np.zeros(K)}, "affine_v2")
            imgs_w2.append(img)
            lbls_w2.append(f"a0={a0}\na1={a1} s={ir:.3f}")
    save(make_collage(imgs_w2, lbls_w2,
        "W2 - Affine v2 grid a[0] x a[1]  (rows=a0, cols=a1)",
        thumb, n_cols=len(a01_vals)), "W2_affine_v2_grid")

    # ── W3: Affine v2 per a_i ────────────────────────────────
    print("\n[W3] Affine v2 - per a_i...")
    a_sweep = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    imgs_w3, lbls_w3 = [], []
    for ki in range(K):
        for av in a_sweep:
            a = np.ones(K); a[ki] = av
            img, ir = gen_image({"a": a, "c": np.zeros(K)}, "affine_v2")
            imgs_w3.append(img)
            lbls_w3.append(f"a[{ki}]={av}\ns={ir:.3f}")
    save(make_collage(imgs_w3, lbls_w3,
        "W3 - Affine v2: each a_i separately  (rows=coef, cols=a)",
        thumb, n_cols=len(a_sweep)), "W3_affine_v2_per_ai")


def main():
    args = parse_args()
    global PROMPT
    PROMPT = args.prompt
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SD-1.5 motivation experiment - 11 collages")
    print(f"PROMPT: {PROMPT}   SEED={SEED}  K={K}  thumb={args.thumb}")
    print(f"scorer={'none' if args.skip_ir else args.scorer}")
    print("=" * 60)

    print("\n[1/3] Loading SD-1.5 ...")
    pipe = load_pipe()

    scorer_name = "none" if args.skip_ir else args.scorer
    print(f"[2/3] Loading scorer: {scorer_name} ...")
    scorer = load_scorer(scorer_name, device=DEVICE)

    print("[3/3] Preparing base noise...")
    g = torch.Generator("cpu").manual_seed(SEED)
    z_base_np = torch.randn(C, LH, LW, generator=g).numpy().astype(np.float32)
    low_idx = get_low_idx(K)

    print(f"\nLatent shape: ({C}, {LH}, {LW})")
    print(f"Low freq idx: first {K} of {LH*LW} coefficients\n")

    t0 = time.time()
    run_all(pipe, scorer, z_base_np, low_idx, save_dir, args.thumb)

    print(f"\n{'='*60}")
    print(f"Done in {(time.time()-t0)/60:.1f} min  ->  {save_dir}/")


if __name__ == "__main__":
    main()
