#!/usr/bin/env python3
"""
Visualize effect of SIGNED parametrizations that can flip low-frequency
coefficients.

Two methods tested:

  1. DCT-affine-signed: simple extension of baseline DCT-affine where
     a_i can now be NEGATIVE:
       f_new[i] = a_i * f_old[i] + c_i   with a_i in [-2, +2]
     When a_i < 0, the sign of the i-th low-frequency coefficient flips.

  2. Chebyshev-phase: sign is parametrized by a smooth curve over
     frequency:
       s(r) = tanh(sum c_m * T_m(tau(r)))      in (-1, +1)
       f_new[u,v] = s(r_{u,v}) * f_old[u,v]
     One smooth sign curve governs ALL frequencies.

For each method we fix all parameters to neutral and vary one at a time
to see its effect on the image, especially composition.

Usage:
  python visualize_signed_params.py --prompt "dragon over castle"
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn, idctn

from diffusers import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor


# =============================================================================
# Config
# =============================================================================

FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16
HEIGHT = 1024
WIDTH = 1024
STEPS = 4
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str,
                   default="a dragon flying over a medieval castle, "
                           "epic fantasy, thunderstorm")
    p.add_argument("--save_dir", type=str, default="results/signed_params")
    return p.parse_args()


# =============================================================================
# FLUX generation
# =============================================================================

@torch.no_grad()
def get_base_latent(pipe, seed):
    gen = torch.Generator("cpu").manual_seed(seed)
    n_channels = pipe.transformer.config.in_channels // 4
    return randn_tensor(
        (1, n_channels, HEIGHT // 8, WIDTH // 8),
        generator=gen, device=pipe.device, dtype=pipe.dtype,
    )


@torch.no_grad()
def generate_from_latent(pipe, z_np, prompt_emb, pooled_emb, txt_ids):
    lat = torch.tensor(z_np[None], device=DEVICE, dtype=DTYPE)
    pipe.scheduler.set_timesteps(STEPS, device=DEVICE)
    img_ids = pipe._prepare_latent_image_ids(
        1, HEIGHT // 16, WIDTH // 16, DEVICE, DTYPE)

    for t in pipe.scheduler.timesteps:
        b, c, H, W = lat.shape
        lp = lat.view(b, c, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
        lp = lp.reshape(b, (H // 2) * (W // 2), c * 4)
        vp = pipe.transformer(
            hidden_states=lp,
            timestep=t.expand(1) / 1000,
            guidance=None,
            encoder_hidden_states=prompt_emb,
            pooled_projections=pooled_emb,
            txt_ids=txt_ids,
            img_ids=img_ids,
        )[0]
        b2, _, _ = vp.shape
        c2 = vp.shape[-1] // 4
        vp = vp.view(b2, HEIGHT // 16, WIDTH // 16, c2, 2, 2)
        vp = vp.permute(0, 3, 1, 4, 2, 5).reshape(
            b2, c2, HEIGHT // 8, WIDTH // 8)
        lat = pipe.scheduler.step(vp, t, lat, return_dict=False)[0]

    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((img[0] * 255).round().astype("uint8"))


# =============================================================================
# Utilities
# =============================================================================

def get_radial(H, W):
    fy = np.arange(H, dtype=np.float32)
    fx = np.arange(W, dtype=np.float32)
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).flatten()


def get_low_idx(r, K):
    return np.argsort(r)[:K]


def renormalize(z_new, z_orig):
    for ch in range(z_new.shape[0]):
        scale = np.linalg.norm(z_orig[ch]) / (
            np.linalg.norm(z_new[ch]) + 1e-9)
        z_new[ch] *= scale
    return z_new


# =============================================================================
# Method 1: DCT-affine-signed (a_i can be negative)
# =============================================================================

K_AFFINE = 5

def apply_dct_affine_signed(z_np, theta, low_idx, sigma):
    """
    Same as baseline DCT-affine but a_i in [-2, +2] instead of [0, 2].
    theta = [a_1, ..., a_K, c_1, ..., c_K]
    """
    a_p, c_p = theta[:K_AFFINE], theta[K_AFFINE:]
    _, H, W = z_np.shape
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat[low_idx] = a_p * flat[low_idx] + c_p
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


# =============================================================================
# Method 2: Chebyshev-phase (smooth sign curve)
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
        T[m] = 2 * tau * T[m-1] - T[m-2]
    return T


def apply_chebyshev_phase(z_np, theta, r_flat, sigma):
    """
    s(r) = tanh(sum c_m * T_m(tau(r)))  in (-1, +1)
    f_new[u,v] = s(r) * f[u,v]

    theta = [c_0, c_1, c_2, c_3]   (no mu_DC here -- we focus on sign)
    """
    c_m = theta
    T = get_cheb_basis(r_flat)
    raw = np.zeros(T.shape[1])
    for m in range(CHEB_M):
        raw += c_m[m] * T[m]
    # use tanh to squash to [-1, +1]
    s_curve = np.tanh(raw)

    _, H, W = z_np.shape
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat = flat * s_curve
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


# =============================================================================
# Test configs
# =============================================================================

# For DCT-affine-signed: show what happens with a_i at neg / 0 / pos
# and with c_i varying.
def test_dct_signed():
    # baseline: all a=1, all c=0
    rows = []

    # Row 1: vary a_1 from -2 to +2
    rows.append(("a_1", [(v, "a1") for v in [-2.0, -1.0, 0.0, 1.0, 2.0]]))
    # Row 2: vary a_3 from -2 to +2
    rows.append(("a_3", [(v, "a3") for v in [-2.0, -1.0, 0.0, 1.0, 2.0]]))
    # Row 3: all a flipped to negative, varying magnitude
    rows.append(("all a_i flipped",
                 [(v, "all_a") for v in [-2.0, -1.5, -1.0, -0.5, +0.5]]))
    return rows


def test_cheb_phase():
    # Neutral is c_0=something positive (so s(r) ~ +1 = identity)
    # Interesting: what happens with c_0 = -10 (s(r) ~ -1 everywhere)
    rows = []
    # Row 1: vary c_0 only (global sign control)
    rows.append(("c_0 (global)", [(v, "c0") for v in [-5.0, -1.0, 0.0, 1.0, 5.0]]))
    # Row 2: vary c_1 only (tilt from low to high)
    rows.append(("c_1 (tilt)", [(v, "c1") for v in [-5.0, -2.0, 0.0, 2.0, 5.0]]))
    # Row 3: vary c_2 only (curvature)
    rows.append(("c_2 (curve)", [(v, "c2") for v in [-5.0, -2.0, 0.0, 2.0, 5.0]]))
    return rows


# =============================================================================
# Main
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
    print("Visualize SIGNED parametrizations")
    print("=" * 70)
    print(f"  Prompt: {args.prompt}")
    total_gens = 15 + 15  # 3 rows * 5 cols for each method
    print(f"  Total generations: {total_gens}  "
          f"({total_gens*10/60:.0f} min)")
    print("=" * 70)

    print("\nLoading FLUX.1-schnell...")
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL, torch_dtype=DTYPE).to(DEVICE)

    with torch.no_grad():
        prompt_emb, pooled_emb, txt_ids = pipe.encode_prompt(
            prompt=args.prompt, prompt_2=None,
            device=DEVICE, num_images_per_prompt=1)

    r = get_radial(HEIGHT // 8, WIDTH // 8)
    low_idx = get_low_idx(r, K_AFFINE)
    z_base = get_base_latent(pipe, SEED)[0].float().cpu().numpy()
    sigma = float(np.std(z_base))
    print(f"  sigma = {sigma:.3f}")

    # Save baseline
    baseline_img = generate_from_latent(
        pipe, z_base, prompt_emb, pooled_emb, txt_ids)
    baseline_img.save(save_dir / "baseline.png")

    # ========================================================================
    # Method 1: DCT-affine-signed
    # ========================================================================
    print("\n[DCT-affine-signed]  (baseline has a=1, c=0)")
    rows_imgs = []

    # Row 1: vary a_1 only
    print("  Row 1: varying a_1")
    col = []
    for a_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        theta = np.array([1.0]*5 + [0.0]*5)  # neutral
        theta[0] = a_val
        z_mod = apply_dct_affine_signed(z_base, theta, low_idx, sigma)
        img = generate_from_latent(pipe, z_mod, prompt_emb, pooled_emb, txt_ids)
        col.append((a_val, img))
        print(f"    a_1={a_val:+.1f} done")
    rows_imgs.append(("a_1 (lowest freq)", col))

    # Row 2: vary a_3 only
    print("  Row 2: varying a_3")
    col = []
    for a_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        theta = np.array([1.0]*5 + [0.0]*5)
        theta[2] = a_val
        z_mod = apply_dct_affine_signed(z_base, theta, low_idx, sigma)
        img = generate_from_latent(pipe, z_mod, prompt_emb, pooled_emb, txt_ids)
        col.append((a_val, img))
        print(f"    a_3={a_val:+.1f} done")
    rows_imgs.append(("a_3 (mid low freq)", col))

    # Row 3: all a_i flipped to same negative value (and some positive variants)
    print("  Row 3: all a_i set to same value")
    col = []
    for a_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        theta = np.array([a_val]*5 + [0.0]*5)  # ALL a the same
        z_mod = apply_dct_affine_signed(z_base, theta, low_idx, sigma)
        img = generate_from_latent(pipe, z_mod, prompt_emb, pooled_emb, txt_ids)
        col.append((a_val, img))
        print(f"    all a_i={a_val:+.1f} done")
    rows_imgs.append(("all 5 a_i same value", col))

    make_collage("DCT-affine-signed", rows_imgs, args.prompt,
                 save_dir / "dct_affine_signed.png",
                 "a_i in [-2, +2]; negative a flips the sign of coef i")

    # ========================================================================
    # Method 2: Chebyshev-phase
    # ========================================================================
    print("\n[Chebyshev-phase]  s(r) = tanh(sum c_m T_m)")
    rows_imgs = []

    # Neutral: c_0 large positive so tanh saturates to +1 (identity)
    NEUTRAL_C0 = 5.0

    # Row 1: vary c_0 (global sign)
    print("  Row 1: varying c_0 (global sign)")
    col = []
    for c_val in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        theta = np.array([c_val, 0.0, 0.0, 0.0])  # only c_0 nonzero
        z_mod = apply_chebyshev_phase(z_base, theta, r, sigma)
        img = generate_from_latent(pipe, z_mod, prompt_emb, pooled_emb, txt_ids)
        col.append((c_val, img))
        s_min = np.tanh(c_val - abs(0))
        print(f"    c_0={c_val:+.1f}  s(r)≈{np.tanh(c_val):+.2f}  done")
    rows_imgs.append(("c_0 (global sign)", col))

    # Row 2: vary c_1 (tilt low->high)
    # Keep c_0 at neutral +5 so baseline s(r)~+1
    print("  Row 2: varying c_1 (tilt) with c_0=+5")
    col = []
    for c_val in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        theta = np.array([NEUTRAL_C0, c_val, 0.0, 0.0])
        z_mod = apply_chebyshev_phase(z_base, theta, r, sigma)
        img = generate_from_latent(pipe, z_mod, prompt_emb, pooled_emb, txt_ids)
        col.append((c_val, img))
        print(f"    c_1={c_val:+.1f}  done")
    rows_imgs.append(("c_1 (low->high tilt)", col))

    # Row 3: vary c_2 (curvature)
    print("  Row 3: varying c_2 (curvature) with c_0=+5")
    col = []
    for c_val in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        theta = np.array([NEUTRAL_C0, 0.0, c_val, 0.0])
        z_mod = apply_chebyshev_phase(z_base, theta, r, sigma)
        img = generate_from_latent(pipe, z_mod, prompt_emb, pooled_emb, txt_ids)
        col.append((c_val, img))
        print(f"    c_2={c_val:+.1f}  done")
    rows_imgs.append(("c_2 (curvature)", col))

    make_collage("Chebyshev-phase", rows_imgs, args.prompt,
                 save_dir / "chebyshev_phase.png",
                 "s(r) = tanh(sum c_m * T_m(tau(r)))  in (-1, +1); "
                 "negative s flips that freq band")

    # ========================================================================
    # Bonus: sign-curve plot (for understanding)
    # ========================================================================
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    r_plot = np.linspace(0, r.max(), 200)
    T_plot = get_cheb_basis(r_plot)

    # c_0 only
    ax = axes[0]
    for c0 in [-5, -1, 0, 1, 5]:
        s = np.tanh(c0 * T_plot[0])
        ax.plot(r_plot, s, lw=2, label=f"c_0={c0:+.0f}")
    ax.axhline(0, color="gray", ls=":")
    ax.axhline(+1, color="gray", ls=":"); ax.axhline(-1, color="gray", ls=":")
    ax.set_title("c_0 only: global sign control")
    ax.set_xlabel("r"); ax.set_ylabel("s(r)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # c_1 (with c_0=+5)
    ax = axes[1]
    for c1 in [-10, -5, 0, 5, 10]:
        s = np.tanh(NEUTRAL_C0 * T_plot[0] + c1 * T_plot[1])
        ax.plot(r_plot, s, lw=2, label=f"c_1={c1:+.0f}")
    ax.axhline(0, color="gray", ls=":")
    ax.axhline(+1, color="gray", ls=":"); ax.axhline(-1, color="gray", ls=":")
    ax.set_title(f"c_1 varying (c_0={NEUTRAL_C0}): tilt low<->high")
    ax.set_xlabel("r"); ax.set_ylabel("s(r)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # c_2 (with c_0=+5)
    ax = axes[2]
    for c2 in [-10, -5, 0, 5, 10]:
        s = np.tanh(NEUTRAL_C0 * T_plot[0] + c2 * T_plot[2])
        ax.plot(r_plot, s, lw=2, label=f"c_2={c2:+.0f}")
    ax.axhline(0, color="gray", ls=":")
    ax.axhline(+1, color="gray", ls=":"); ax.axhline(-1, color="gray", ls=":")
    ax.set_title(f"c_2 varying (c_0={NEUTRAL_C0}): curvature")
    ax.set_xlabel("r"); ax.set_ylabel("s(r)")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Chebyshev-phase sign curves s(r) for different theta",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "cheb_phase_curves.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nAll saved to {save_dir}/")


if __name__ == "__main__":
    main()