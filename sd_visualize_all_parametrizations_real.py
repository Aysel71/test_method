#!/usr/bin/env python3
"""
SD-1.5 version of visualize_all_parametrizations_real.py.

Tests all 8 DCT parametrizations of the initial SD-1.5 latent noise on 4
prompt types. The parametrization math is identical to the FLUX version
(see visualize_all_parametrizations_real.py); only the generator is SD-1.5.

AMPLITUDE-ONLY methods (preserve composition):
  1. dct_affine, 2. power_law, 3. log_bands, 4. chebyshev, 5. bspline, 6. rbf
SIGNED methods (can change composition):
  7. dct_affine_signed, 8. chebyshev_phase

Usage:
  python sd_visualize_all_parametrizations_real.py --save_dir results/sd_all_parametrizations
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn, idctn

from sd_common import (
    load_pipe, get_base_latent, generate_from_latent,
    get_radial, get_low_idx, renormalize, SEED,
)
from scorers import load_scorer, score_one


PROMPTS = [
    ("portrait",  "a portrait of a woman in golden light, "
                  "soft focus, cinematic lighting"),
    ("landscape", "a snowy mountain landscape at sunrise, "
                  "dramatic clouds, ultra-detailed"),
    ("object",    "a cup of coffee on a wooden table, "
                  "morning light, cozy atmosphere"),
    ("fantasy",   "a dragon flying over a medieval castle, "
                  "epic fantasy, thunderstorm"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default="results/sd_all_parametrizations")
    p.add_argument("--prompts",  nargs="+", default=None)
    p.add_argument("--methods",  nargs="+", default=None)
    p.add_argument("--scorer",   type=str, default="none",
                   help="none | hpsv3 | imagereward  (label under each image)")
    return p.parse_args()


# =============================================================================
# Parametrization methods (identical to FLUX version, operate on (C, H, W))
# =============================================================================

K_AFFINE = 5
CHEB_M = 4


def apply_dct_affine(z_np, theta, low_idx):
    a_p, c_p = theta[:K_AFFINE], theta[K_AFFINE:]
    _, H, W = z_np.shape
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat[low_idx] = a_p * flat[low_idx] + c_p
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


def apply_power_law(z_np, theta, r_flat):
    A, alpha, mu_dc = theta
    _, H, W = z_np.shape
    a_curve = A * np.power(r_flat + 1, -alpha / 2)
    a_matrix = a_curve.reshape(H, W)
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho")
        flat = a_matrix * flat
        flat[0, 0] += mu_dc
        z_new[ch] = idctn(flat, norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


def apply_log_bands(z_np, theta, r_flat):
    a_bands, c_bands = theta[:5], theta[5:]
    _, H, W = z_np.shape
    log_r_max = np.log(r_flat.max() + 1.0)
    log_r = np.log(r_flat + 1.0)
    a_matrix = np.ones_like(r_flat)
    c_matrix = np.zeros_like(r_flat)
    for i, (a_val, c_val) in enumerate(zip(a_bands, c_bands)):
        band_start = i * log_r_max / 5
        band_end = (i + 1) * log_r_max / 5
        if i == 4:
            mask = (log_r >= band_start) & (log_r <= band_end)
        else:
            mask = (log_r >= band_start) & (log_r < band_end)
        a_matrix[mask] = a_val
        c_matrix[mask] = c_val
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat = a_matrix * flat + c_matrix
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


def get_cheb_basis(r_flat):
    log_r = np.log(r_flat + 1.0)
    log_r_max = np.log(r_flat.max() + 1.0)
    tau = 2.0 * log_r / log_r_max - 1.0
    T = np.zeros((CHEB_M, len(tau)))
    T[0] = 1.0
    if CHEB_M > 1:
        T[1] = tau
    for m in range(2, CHEB_M):
        T[m] = 2 * tau * T[m - 1] - T[m - 2]
    return T


def apply_chebyshev(z_np, theta, r_flat):
    c_coeffs, mu_dc = theta[:CHEB_M], theta[CHEB_M]
    T = get_cheb_basis(r_flat)
    log_a = np.zeros(T.shape[1])
    for m in range(CHEB_M):
        log_a += c_coeffs[m] * T[m]
    a_curve = np.exp(log_a)
    _, H, W = z_np.shape
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat = a_curve * flat
        flat[0] += mu_dc
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


def apply_bspline(z_np, theta, r_flat):
    y_knots, mu_dc = theta[:6], theta[6]
    log_r_max = np.log(r_flat.max() + 1.0)
    log_r_norm = np.log(r_flat + 1.0) / log_r_max
    knot_pos = np.linspace(0, 1, 6)
    log_a = np.interp(log_r_norm, knot_pos, y_knots)
    a_curve = np.exp(log_a)
    _, H, W = z_np.shape
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat = a_curve * flat
        flat[0] += mu_dc
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


def apply_rbf(z_np, theta, r_flat):
    a0, b, w1, w2, w3, mu_dc = theta[:6]
    log_r = np.log(r_flat + 1.0)
    log_r_max = np.log(r_flat.max() + 1.0)
    centers = [0.2 * log_r_max, 0.5 * log_r_max, 0.8 * log_r_max]
    sigma = log_r_max / 7.5
    log_a = a0 + b * log_r
    for w, center in zip([w1, w2, w3], centers):
        gaussian = np.exp(-0.5 * ((log_r - center) / sigma) ** 2)
        log_a += w * gaussian
    a_curve = np.exp(log_a)
    _, H, W = z_np.shape
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat = a_curve * flat
        flat[0] += mu_dc
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


def apply_dct_affine_signed(z_np, theta, low_idx):
    return apply_dct_affine(z_np, theta, low_idx)


def apply_chebyshev_phase(z_np, theta, r_flat):
    T = get_cheb_basis(r_flat)
    raw = np.zeros(T.shape[1])
    for m in range(CHEB_M):
        raw += theta[m] * T[m]
    s_curve = np.tanh(raw)
    _, H, W = z_np.shape
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat = flat * s_curve
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


# =============================================================================
# Collage
# =============================================================================

def load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_collage(title, prompt, rows, save_path, subtitle=""):
    thumb = 200
    pad = 5
    label_w = 170
    header_h = 75
    footer_h = 10
    n_rows = len(rows)
    n_cols = max(len(items) for _, items in rows)
    W_total = label_w + n_cols * (thumb + pad) + pad
    H_total = header_h + n_rows * (thumb + pad) + footer_h

    canvas = Image.new("RGB", (W_total, H_total), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font_big = load_font(14)
    font_med = load_font(11)
    font_sm = load_font(9)

    draw.text((pad, 6), title, fill="#1E1E1E", font=font_big)
    draw.text((pad, 26), prompt[:110], fill="#505050", font=font_med)
    draw.text((pad, 45), subtitle, fill="#757575", font=font_sm)

    for row_i, (row_label, col_items) in enumerate(rows):
        y = header_h + row_i * (thumb + pad)
        draw.text((pad, y + thumb // 2 - 10),
                  row_label, fill="#1E1E1E", font=font_med)
        for col_j, item in enumerate(col_items):
            val, img = item[0], item[1]
            score = item[2] if len(item) > 2 else None
            x = label_w + col_j * (thumb + pad)
            canvas.paste(img.resize((thumb, thumb), Image.LANCZOS), (x, y))
            val_str = f"{val:+.2f}" if isinstance(val, (int, float)) else str(val)
            if score is not None:
                val_str = f"{val_str}\nHPS={score:+.3f}"
            draw.text((x + 4, y + 4), val_str,
                      fill="#FFFFFF", font=font_sm,
                      stroke_width=2, stroke_fill="#000000")
    canvas.save(save_path, quality=92)


# =============================================================================
# Method configurations (thresholds to tune per model)
# =============================================================================

METHODS = {
    'dct_affine': {
        'func': apply_dct_affine,
        'test_values': [0.0, 0.25, 0.5, 0.57, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.3],
        'base_theta': lambda val: [val, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'a_1',
        'needs_low_idx': True,
        'title': 'DCT-affine (amplitude only)',
        'subtitle': 'HPSv3-calibrated a_1 in [0, 2.3]; best ~0.57 (HPS 8.40 > base 8.05)'
    },
    'power_law': {
        'func': apply_power_law,
        'test_values': [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0],
        'base_theta': lambda val: [1.0, val, 0.0],
        'param_name': 'alpha',
        'needs_low_idx': False,
        'title': 'Power-law (HPSv3-fragile)',
        'subtitle': 'a(r)=A*(r+1)^(-alpha/2); HPSv3 likes only alpha~0 - any slope degrades'
    },
    'log_bands': {
        'func': apply_log_bands,
        'test_values': [0.0, 0.25, 0.5, 0.57, 0.75, 1.0, 1.15, 1.3, 1.43],
        'base_theta': lambda val: [val, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'a_band1',
        'needs_low_idx': False,
        'title': 'Log-bands',
        'subtitle': 'HPSv3-calibrated low band in [0, 1.43]; best ~0.57 (HPS 8.44 > base 7.91)'
    },
    'chebyshev': {
        'func': apply_chebyshev,
        'test_values': [-1.2, -0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8, 1.2],
        'base_theta': lambda val: [val, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'c_0',
        'needs_low_idx': False,
        'title': 'Chebyshev (amplitude, weak knob)',
        'subtitle': 'a(r)=exp(sum c_m T_m); HPSv3 flat over [-1.2,1.2] - little effect'
    },
    'bspline': {
        'func': apply_bspline,
        'test_values': [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2],
        'base_theta': lambda val: [val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'y_1',
        'needs_low_idx': False,
        'title': 'B-spline',
        'subtitle': 'HPSv3-calibrated first knot in [-1.2, 0]; best -1.2 (HPS 8.47 > base 8.38)'
    },
    'rbf': {
        'func': apply_rbf,
        'test_values': [-0.4, -0.34, -0.2, -0.1, 0.0, 0.1, 0.17, 0.25, 0.35, 0.51],
        'base_theta': lambda val: [0.0, 0.0, val, 0.0, 0.0, 0.0],
        'param_name': 'w_1',
        'needs_low_idx': False,
        'title': 'RBF (best method)',
        'subtitle': 'HPSv3-calibrated low bump w in [-0.34, 0.51]; best ~0.17 (HPS 8.54, top)'
    },
    'dct_affine_signed': {
        'func': apply_dct_affine_signed,
        'test_values': [-2.0, -1.0, 0.0, 0.5, 0.857, 1.0, 1.25, 1.5, 2.0],
        'base_theta': lambda val: [val, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'a_1',
        'needs_low_idx': True,
        'title': 'DCT-affine SIGNED (HPSv3-fragile)',
        'subtitle': 'HPSv3 survives only a_1 ~[0.86, 1.0]; sign flip (a<0) destroys the image'
    },
    'chebyshev_phase': {
        'func': apply_chebyshev_phase,
        'test_values': [-5.0, -2.0, 0.0, 2.14, 3.0, 5.0, 8.0, 11.0, 15.0],
        'base_theta': lambda val: [val, 0.0, 0.0, 0.0],
        'param_name': 'c_0',
        'needs_low_idx': False,
        'title': 'Chebyshev PHASE (HPSv3-fragile)',
        'subtitle': 's(r)=tanh(sum c_m T_m); HPSv3 safe only c_0>=2.14 (tanh~+1); flip destroys'
    }
}


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    prompts_to_run = PROMPTS
    if args.prompts:
        prompts_to_run = [p for p in PROMPTS if p[0] in args.prompts]

    methods_to_run = list(METHODS.keys())
    if args.methods:
        methods_to_run = [m for m in methods_to_run if m in args.methods]

    total = len(prompts_to_run) * sum(len(METHODS[m]['test_values']) for m in methods_to_run)
    print("=" * 70)
    print(f"SD-1.5  |  Prompts: {len(prompts_to_run)}  Methods: {len(methods_to_run)}")
    print(f"Total gens: ~{total}")
    print("=" * 70)

    print("\nLoading SD-1.5 ...")
    pipe = load_pipe()
    scorer = load_scorer(args.scorer, device="cuda")

    r = get_radial()
    low_idx = get_low_idx(r, K_AFFINE)
    z_base = get_base_latent(SEED)
    print(f"  latent shape = {z_base.shape}, sigma = {np.std(z_base):.3f}\n")

    for p_label, p_text in prompts_to_run:
        print(f"\n{'='*70}\nPrompt: [{p_label}] {p_text[:60]}\n{'='*70}")

        baseline_img = generate_from_latent(pipe, z_base, p_text)
        baseline_img.save(save_dir / f"{p_label}_baseline.png")

        for method_name in methods_to_run:
            print(f"[{p_label}] {method_name}")
            mc = METHODS[method_name]
            col_items = []
            for val in mc['test_values']:
                theta = mc['base_theta'](val)
                if mc['needs_low_idx']:
                    z_mod = mc['func'](z_base, theta, low_idx)
                else:
                    z_mod = mc['func'](z_base, theta, r)
                img = generate_from_latent(pipe, z_mod, p_text)
                if scorer is not None:
                    col_items.append((val, img, score_one(scorer, img, p_text)))
                else:
                    col_items.append((val, img))

            rows = [(mc['param_name'], col_items)]
            make_collage(f"{mc['title']} [{p_label}]", p_text, rows,
                         save_dir / f"{p_label}_{method_name}.png", mc['subtitle'])

        print(f"  [{p_label}] done.")

    print(f"\nAll saved to {save_dir}/")


if __name__ == "__main__":
    main()
