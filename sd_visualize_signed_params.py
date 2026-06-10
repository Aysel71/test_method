#!/usr/bin/env python3
"""
SD-1.5 version of visualize_signed_params.py.

Two signed parametrizations of the initial SD-1.5 latent noise:
  1. DCT-affine-signed: f_new[i] = a_i * f_old[i] + c_i, a_i in [-2, +2]
     (a_i < 0 flips the sign of low-freq coefficient i)
  2. Chebyshev-phase: s(r) = tanh(sum c_m T_m(tau(r))) in (-1, +1)
     one smooth sign curve over all frequencies

Usage:
  python sd_visualize_signed_params.py --prompt "a dragon flying over a medieval castle"
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str,
                   default="a dragon flying over a medieval castle, "
                           "epic fantasy, thunderstorm")
    p.add_argument("--save_dir", type=str, default="results/sd_signed_params")
    return p.parse_args()


# =============================================================================
# Method 1: DCT-affine-signed
# =============================================================================

K_AFFINE = 5


def apply_dct_affine_signed(z_np, theta, low_idx):
    a_p, c_p = theta[:K_AFFINE], theta[K_AFFINE:]
    _, H, W = z_np.shape
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat[low_idx] = a_p * flat[low_idx] + c_p
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


# =============================================================================
# Method 2: Chebyshev-phase
# =============================================================================

CHEB_M = 4


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


def apply_chebyshev_phase(z_np, theta, r_flat):
    c_m = theta
    T = get_cheb_basis(r_flat)
    raw = np.zeros(T.shape[1])
    for m in range(CHEB_M):
        raw += c_m[m] * T[m]
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


def make_collage(method_name, rows_imgs, prompt, save_path, extra_info=""):
    thumb = 220
    pad = 6
    label_w = 160
    header_h = 80
    footer_h = 10
    n_rows = len(rows_imgs)
    n_cols = max(len(items) for _, items in rows_imgs)
    W_total = label_w + n_cols * (thumb + pad) + pad
    H_total = header_h + n_rows * (thumb + pad) + footer_h

    canvas = Image.new("RGB", (W_total, H_total), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(14)
    font_med = load_font(11)
    font_sm = load_font(9)

    draw.text((pad, 6), f"Signed parametrization: {method_name}",
              fill="#1E1E1E", font=font_title)
    draw.text((pad, 26), prompt[:110], fill="#505050", font=font_med)
    draw.text((pad, 45), extra_info, fill="#757575", font=font_sm)

    for row_i, (row_label, col_items) in enumerate(rows_imgs):
        y = header_h + row_i * (thumb + pad)
        draw.text((pad, y + thumb // 2 - 10),
                  row_label, fill="#1E1E1E", font=font_med)
        for col_j, (val, img) in enumerate(col_items):
            x = label_w + col_j * (thumb + pad)
            canvas.paste(img.resize((thumb, thumb), Image.LANCZOS), (x, y))
            val_str = (f"{val:+.3f}" if isinstance(val, (int, float))
                       else str(val))
            draw.text((x + 4, y + 4), val_str,
                      fill="#FFFFFF", font=font_sm,
                      stroke_width=2, stroke_fill="#000000")
    canvas.save(save_path, quality=92)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SD-1.5  |  Visualize SIGNED parametrizations")
    print(f"  Prompt: {args.prompt}")
    print("=" * 70)

    print("\nLoading SD-1.5 ...")
    pipe = load_pipe()

    r = get_radial()
    low_idx = get_low_idx(r, K_AFFINE)
    z_base = get_base_latent(SEED)
    print(f"  latent shape = {z_base.shape}, sigma = {np.std(z_base):.3f}")

    baseline_img = generate_from_latent(pipe, z_base, args.prompt)
    baseline_img.save(save_dir / "baseline.png")

    # ---- Method 1: DCT-affine-signed ----------------------------------------
    print("\n[DCT-affine-signed]")
    rows_imgs = []

    col = []
    for a_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        theta = np.array([1.0] * 5 + [0.0] * 5)
        theta[0] = a_val
        z_mod = apply_dct_affine_signed(z_base, theta, low_idx)
        col.append((a_val, generate_from_latent(pipe, z_mod, args.prompt)))
    rows_imgs.append(("a_1 (lowest freq)", col))

    col = []
    for a_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        theta = np.array([1.0] * 5 + [0.0] * 5)
        theta[2] = a_val
        z_mod = apply_dct_affine_signed(z_base, theta, low_idx)
        col.append((a_val, generate_from_latent(pipe, z_mod, args.prompt)))
    rows_imgs.append(("a_3 (mid low freq)", col))

    col = []
    for a_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        theta = np.array([a_val] * 5 + [0.0] * 5)
        z_mod = apply_dct_affine_signed(z_base, theta, low_idx)
        col.append((a_val, generate_from_latent(pipe, z_mod, args.prompt)))
    rows_imgs.append(("all 5 a_i same value", col))

    make_collage("DCT-affine-signed", rows_imgs, args.prompt,
                 save_dir / "dct_affine_signed.png",
                 "a_i in [-2, +2]; negative a flips the sign of coef i")

    # ---- Method 2: Chebyshev-phase ------------------------------------------
    print("\n[Chebyshev-phase]  s(r) = tanh(sum c_m T_m)")
    rows_imgs = []
    NEUTRAL_C0 = 5.0

    col = []
    for c_val in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        theta = np.array([c_val, 0.0, 0.0, 0.0])
        z_mod = apply_chebyshev_phase(z_base, theta, r)
        col.append((c_val, generate_from_latent(pipe, z_mod, args.prompt)))
    rows_imgs.append(("c_0 (global sign)", col))

    col = []
    for c_val in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        theta = np.array([NEUTRAL_C0, c_val, 0.0, 0.0])
        z_mod = apply_chebyshev_phase(z_base, theta, r)
        col.append((c_val, generate_from_latent(pipe, z_mod, args.prompt)))
    rows_imgs.append(("c_1 (low->high tilt)", col))

    col = []
    for c_val in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        theta = np.array([NEUTRAL_C0, 0.0, c_val, 0.0])
        z_mod = apply_chebyshev_phase(z_base, theta, r)
        col.append((c_val, generate_from_latent(pipe, z_mod, args.prompt)))
    rows_imgs.append(("c_2 (curvature)", col))

    make_collage("Chebyshev-phase", rows_imgs, args.prompt,
                 save_dir / "chebyshev_phase.png",
                 "s(r) = tanh(sum c_m * T_m(tau(r))) in (-1, +1); "
                 "negative s flips that freq band")

    # ---- Bonus: sign-curve plot ---------------------------------------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    r_plot = np.linspace(0, r.max(), 200)
    T_plot = get_cheb_basis(r_plot)

    ax = axes[0]
    for c0 in [-5, -1, 0, 1, 5]:
        ax.plot(r_plot, np.tanh(c0 * T_plot[0]), lw=2, label=f"c_0={c0:+.0f}")
    ax.axhline(0, color="gray", ls=":"); ax.axhline(+1, color="gray", ls=":")
    ax.axhline(-1, color="gray", ls=":")
    ax.set_title("c_0 only: global sign control")
    ax.set_xlabel("r"); ax.set_ylabel("s(r)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for c1 in [-10, -5, 0, 5, 10]:
        ax.plot(r_plot, np.tanh(NEUTRAL_C0 * T_plot[0] + c1 * T_plot[1]),
                lw=2, label=f"c_1={c1:+.0f}")
    ax.axhline(0, color="gray", ls=":"); ax.axhline(+1, color="gray", ls=":")
    ax.axhline(-1, color="gray", ls=":")
    ax.set_title(f"c_1 varying (c_0={NEUTRAL_C0}): tilt low<->high")
    ax.set_xlabel("r"); ax.set_ylabel("s(r)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for c2 in [-10, -5, 0, 5, 10]:
        ax.plot(r_plot, np.tanh(NEUTRAL_C0 * T_plot[0] + c2 * T_plot[2]),
                lw=2, label=f"c_2={c2:+.0f}")
    ax.axhline(0, color="gray", ls=":"); ax.axhline(+1, color="gray", ls=":")
    ax.axhline(-1, color="gray", ls=":")
    ax.set_title(f"c_2 varying (c_0={NEUTRAL_C0}): curvature")
    ax.set_xlabel("r"); ax.set_ylabel("s(r)"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Chebyshev-phase sign curves s(r) for different theta",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "cheb_phase_curves.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nAll saved to {save_dir}/")


if __name__ == "__main__":
    main()
