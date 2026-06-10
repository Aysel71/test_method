#!/usr/bin/env python3
"""
Comprehensive visualization of ALL 8 DCT parametrizations on REAL FLUX images.

Tests all parametrization methods on 4 different prompt types:
  - portrait (face-centric, likely small changes)
  - landscape (lots of structure)  
  - object still-life (sensitive to arrangement)
  - fantasy scene (complex composition)

AMPLITUDE-ONLY methods (preserve composition):
  1. DCT-affine - modify 5 lowest frequencies directly
  2. Power-law - global spectral slope A*(r+1)^(-α/2)
  3. Log-bands - 5 logarithmic frequency bands
  4. Chebyshev - polynomial amplitude envelope
  5. B-spline - spline through 6 knots
  6. RBF - linear base + 3 Gaussian bumps

SIGNED methods (can change composition):
  7. DCT-affine-signed with a_i in [-3, +3]
  8. Chebyshev-phase with c_m in [-15, +15]

Usage:
  python visualize_all_parametrizations_real.py --save_dir results/all_parametrizations
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn, idctn

from diffusers import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor


FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16
HEIGHT = 1024
WIDTH = 1024
STEPS = 4
SEED = 42

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
    p.add_argument("--save_dir", type=str, default="results/all_parametrizations")
    p.add_argument("--prompts",  nargs="+", default=None,
                   help="which prompt labels to run (default: all)")
    p.add_argument("--methods", nargs="+", default=None,
                   help="which methods to run (default: all)")
    return p.parse_args()


# =============================================================================
# FLUX
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
            timestep=t.expand(1) / 1000, guidance=None,
            encoder_hidden_states=prompt_emb, pooled_projections=pooled_emb,
            txt_ids=txt_ids, img_ids=img_ids,
        )[0]
        b2 = vp.shape[0]; c2 = vp.shape[-1] // 4
        vp = vp.view(b2, HEIGHT // 16, WIDTH // 16, c2, 2, 2)
        vp = vp.permute(0, 3, 1, 4, 2, 5).reshape(
            b2, c2, HEIGHT // 8, WIDTH // 8)
        lat = pipe.scheduler.step(vp, t, lat, return_dict=False)[0]
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((img[0] * 255).round().astype("uint8"))


def get_radial(H, W):
    fy = np.arange(H, dtype=np.float32)
    fx = np.arange(W, dtype=np.float32)
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).flatten()


def renormalize(z_new, z_orig):
    for ch in range(z_new.shape[0]):
        scale = np.linalg.norm(z_orig[ch]) / (
            np.linalg.norm(z_new[ch]) + 1e-9)
        z_new[ch] *= scale
    return z_new


# =============================================================================
# Parametrization methods
# =============================================================================

K_AFFINE = 5
CHEB_M = 4

# 1. DCT-affine (amplitude-only, a_i >= 0)
def apply_dct_affine(z_np, theta, low_idx):
    """Original DCT-affine with a_i >= 0."""
    a_p, c_p = theta[:K_AFFINE], theta[K_AFFINE:]
    _, H, W = z_np.shape
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho").flatten()
        flat[low_idx] = a_p * flat[low_idx] + c_p
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


# 2. Power-law
def apply_power_law(z_np, theta, r_flat):
    """A * (r+1)^(-alpha/2)"""
    A, alpha, mu_dc = theta
    _, H, W = z_np.shape
    
    # Build amplitude matrix
    a_curve = A * np.power(r_flat + 1, -alpha/2)
    a_matrix = a_curve.reshape(H, W)
    
    z_new = z_np.copy().astype(np.float64)
    for ch in range(z_np.shape[0]):
        flat = dctn(z_new[ch], norm="ortho")
        flat = a_matrix * flat
        flat[0, 0] += mu_dc
        z_new[ch] = idctn(flat, norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


# 3. Log-bands
def apply_log_bands(z_np, theta, r_flat):
    """5 log-spaced bands."""
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


# 4. Chebyshev (amplitude)
def get_cheb_basis(r_flat):
    log_r = np.log(r_flat + 1.0)
    log_r_max = np.log(r_flat.max() + 1.0)
    tau = 2.0 * log_r / log_r_max - 1.0
    T = np.zeros((CHEB_M, len(tau)))
    T[0] = 1.0
    if CHEB_M > 1: T[1] = tau
    for m in range(2, CHEB_M):
        T[m] = 2 * tau * T[m-1] - T[m-2]
    return T


def apply_chebyshev(z_np, theta, r_flat):
    """exp(sum c_m T_m) - amplitude only."""
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
        flat[0] += mu_dc  # DC offset
        z_new[ch] = idctn(flat.reshape(H, W), norm="ortho")
    return renormalize(z_new, z_np).astype(np.float32)


# 5. B-spline (simplified as piecewise linear)
def apply_bspline(z_np, theta, r_flat):
    """Piecewise linear spline through 6 knots."""
    y_knots, mu_dc = theta[:6], theta[6]
    
    log_r_max = np.log(r_flat.max() + 1.0)
    log_r_norm = np.log(r_flat + 1.0) / log_r_max  # [0, 1]
    
    # 6 knots at positions 0, 0.2, 0.4, 0.6, 0.8, 1.0
    knot_pos = np.linspace(0, 1, 6)
    
    # Linear interpolation
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


# 6. RBF (simplified)
def apply_rbf(z_np, theta, r_flat):
    """a_0 + b*log(r) + sum w_j * gaussian_j."""
    a0, b, w1, w2, w3, mu_dc = theta[:6]
    
    log_r = np.log(r_flat + 1.0)
    log_r_max = np.log(r_flat.max() + 1.0)
    
    # 3 Gaussian centers at 20%, 50%, 80%
    centers = [0.2 * log_r_max, 0.5 * log_r_max, 0.8 * log_r_max]
    sigma = log_r_max / 7.5
    
    # Base + bumps
    log_a = a0 + b * log_r
    for j, (w, center) in enumerate(zip([w1, w2, w3], centers)):
        gaussian = np.exp(-0.5 * ((log_r - center) / sigma)**2)
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


# 7. DCT-affine-signed
def apply_dct_affine_signed(z_np, theta, low_idx):
    """DCT-affine with a_i can be negative."""
    # Same function, different bounds
    return apply_dct_affine(z_np, theta, low_idx)


# 8. Chebyshev-phase
def apply_chebyshev_phase(z_np, theta, r_flat):
    """tanh(sum c_m T_m) - can flip signs."""
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
        for col_j, (val, img) in enumerate(col_items):
            x = label_w + col_j * (thumb + pad)
            canvas.paste(img.resize((thumb, thumb), Image.LANCZOS), (x, y))
            val_str = f"{val:+.2f}" if isinstance(val, (int, float)) else str(val)
            draw.text((x + 4, y + 4), val_str,
                      fill="#FFFFFF", font=font_sm,
                      stroke_width=2, stroke_fill="#000000")
    canvas.save(save_path, quality=92)


# =============================================================================
# Method configurations with EXPANDED RANGES for extreme experiments
# =============================================================================

METHODS = {
    'dct_affine': {
        'func': apply_dct_affine,
        'test_values': [0.0, 0.3, 0.7, 1.0, 1.5, 2.5, 4.0],  # expanded upper bound
        'base_theta': lambda val: [val, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'a_1',
        'needs_low_idx': True,
        'title': 'DCT-affine (amplitude only)',
        'subtitle': 'a_i ≥ 0; modifies 5 lowest frequencies - EXPANDED RANGE'
    },
    'power_law': {
        'func': apply_power_law,
        'test_values': [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],  # much wider alpha range
        'base_theta': lambda val: [1.0, val, 0.0],
        'param_name': 'α',
        'needs_low_idx': False,
        'title': 'Power-law WIDE',
        'subtitle': 'a(r) = A*(r+1)^(-α/2); α∈[-2,+2] extreme spectral slopes'
    },
    'log_bands': {
        'func': apply_log_bands,
        'test_values': [0.1, 0.5, 0.8, 1.0, 1.5, 2.5, 4.0],  # extreme band boosts
        'base_theta': lambda val: [val, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'a_band1',
        'needs_low_idx': False,
        'title': 'Log-bands EXTREME',
        'subtitle': '5 logarithmic frequency bands; low band ∈[0.1,4.0]'
    },
    'chebyshev': {
        'func': apply_chebyshev,
        'test_values': [-0.8, -0.4, -0.1, 0.0, 0.1, 0.4, 0.8],  # wider polynomial range
        'base_theta': lambda val: [val, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'c_0',
        'needs_low_idx': False,
        'title': 'Chebyshev (amplitude) WIDE',
        'subtitle': 'a(r) = exp(Σc_m T_m); c_0 ∈[-0.8,+0.8] polynomial envelope'
    },
    'bspline': {
        'func': apply_bspline,
        'test_values': [-0.8, -0.4, -0.1, 0.0, 0.1, 0.4, 0.8],  # wider spline range
        'base_theta': lambda val: [val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'y_1',
        'needs_low_idx': False,
        'title': 'B-spline WIDE',
        'subtitle': 'spline through 6 knots; first knot ∈[-0.8,+0.8]'
    },
    'rbf': {
        'func': apply_rbf,
        'test_values': [-0.8, -0.4, -0.1, 0.0, 0.1, 0.4, 0.8],  # wider RBF weights
        'base_theta': lambda val: [0.0, 0.0, val, 0.0, 0.0, 0.0],
        'param_name': 'w_1',
        'needs_low_idx': False,
        'title': 'RBF WIDE',
        'subtitle': 'linear base + 3 Gaussian bumps; low bump w∈[-0.8,+0.8]'
    },
    'dct_affine_signed': {
        'func': apply_dct_affine_signed,
        'test_values': [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0],  # MUCH wider signed range
        'base_theta': lambda val: [val, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'param_name': 'a_1',
        'needs_low_idx': True,
        'title': '★ DCT-affine SIGNED EXTREME',
        'subtitle': 'a_i ∈ [-5,+5]; negative = sign flip → EXTREME composition change'
    },
    'chebyshev_phase': {
        'func': apply_chebyshev_phase,
        'test_values': [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0],  # EXTREME signed range
        'base_theta': lambda val: [val, 0.0, 0.0, 0.0],
        'param_name': 'c_0',
        'needs_low_idx': False,
        'title': '★ Chebyshev PHASE EXTREME',
        'subtitle': 's(r) = tanh(Σc_m T_m); c_0 ∈[-15,+15] EXTREME signed curve'
    }
}


# =============================================================================
# Main
# =============================================================================

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

    total = len(prompts_to_run) * len(methods_to_run) * 7  # 7 values per method now
    print("=" * 70)
    print(f"Prompts to test: {len(prompts_to_run)}")
    print(f"Methods to test: {len(methods_to_run)}")
    print(f"Total gens: ~{total}  (~{total * 10 / 60:.0f} min)")
    print(f"🔥 EXTREME RANGES - wider parameter boundaries!")
    print("=" * 70)

    print("\nLoading FLUX.1-schnell...")
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL, torch_dtype=DTYPE).to(DEVICE)

    r = get_radial(HEIGHT // 8, WIDTH // 8)
    low_idx = np.argsort(r)[:K_AFFINE]
    z_base = get_base_latent(pipe, SEED)[0].float().cpu().numpy()
    sigma = float(np.std(z_base))
    print(f"  sigma = {sigma:.3f}\n")

    for p_label, p_text in prompts_to_run:
        print(f"\n{'='*70}")
        print(f"Prompt: [{p_label}] {p_text[:60]}")
        print("=" * 70)

        with torch.no_grad():
            prompt_emb, pooled_emb, txt_ids = pipe.encode_prompt(
                prompt=p_text, prompt_2=None,
                device=DEVICE, num_images_per_prompt=1)

        # Save baseline
        baseline_img = generate_from_latent(
            pipe, z_base, prompt_emb, pooled_emb, txt_ids)
        baseline_img.save(save_dir / f"{p_label}_baseline.png")

        for method_name in methods_to_run:
            print(f"\n[{p_label}] {method_name}")
            
            method_config = METHODS[method_name]
            method_func = method_config['func']
            test_values = method_config['test_values']
            
            # Generate images for this method
            col_items = []
            for val in test_values:
                theta = method_config['base_theta'](val)
                
                # Apply parametrization
                if method_config['needs_low_idx']:
                    z_mod = method_func(z_base, theta, low_idx)
                else:
                    z_mod = method_func(z_base, theta, r)
                
                # Generate image
                img = generate_from_latent(
                    pipe, z_mod, prompt_emb, pooled_emb, txt_ids)
                col_items.append((val, img))
            
            # Create collage for this method
            rows = [(method_config['param_name'], col_items)]
            make_collage(
                f"{method_config['title']} [{p_label}]",
                p_text, rows,
                save_dir / f"{p_label}_{method_name}.png",
                method_config['subtitle']
            )

        print(f"\n  [{p_label}] All methods completed.")

    # Create summary
    summary = f"""# All 8 Parametrizations Comparison Results

Generated comprehensive comparison of DCT parametrization methods on real FLUX images.

## Files created:
- `{{prompt}}_baseline.png` - original FLUX outputs
- `{{prompt}}_{{method}}.png` - parameter sweeps for each method

## Amplitude-only methods (should preserve composition):
1. **dct_affine** - 5 lowest frequencies, a_i ≥ 0
2. **power_law** - global spectral slope 
3. **log_bands** - 5 frequency bands
4. **chebyshev** - polynomial amplitude envelope
5. **bspline** - spline through knots
6. **rbf** - linear + Gaussian bumps

## Signed methods (can change composition): ★
7. **dct_affine_signed** - can flip coefficient signs
8. **chebyshev_phase** - signed polynomial curve

## Key observations to look for:
1. **Amplitude methods (1-6)**: texture/lighting changes only - EXTREME ranges may show limits
2. **Signed methods (7-8)**: potential composition changes - EXTREME values may break/create new modes  
3. **Cross-prompt sensitivity**: which prompts show biggest effects at extreme values?
4. **Quality degradation**: where do methods break down? Artifacts, unrealistic images?
5. **🔥 NEW: Artistic effects** - extreme parameters may create stylistic transformations

## EXTREME PARAMETER RANGES:
- **dct_affine**: a_1 ∈ [0, 4.0] (was [0, 2.0])
- **power_law**: α ∈ [-2.0, +2.0] (was [-1.0, +1.0]) 
- **dct_affine_signed**: a_1 ∈ [-5.0, +5.0] (was [-3.0, +3.0]) ⭐
- **chebyshev_phase**: c_0 ∈ [-15.0, +15.0] (was [-10.0, +10.0]) ⭐

Generated for {len(prompts_to_run)} prompts × {len(methods_to_run)} methods × 7 values = {total} total images.
"""

    with open(save_dir / "README.md", 'w') as f:
        f.write(summary)

    print(f"\n\nAll saved to {save_dir}/")
    print("🔍 Look for differences between amplitude (1-6) vs signed (7-8) methods!")


if __name__ == "__main__":
    main()