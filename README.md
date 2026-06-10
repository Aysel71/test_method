# test_method — DCT noise parametrization experiments

Noise-modification experiments: vary DCT alpha coefficients in front of the
low-frequency coefficients, build grids in several ways, and sweep the
transformation itself. Companion to the `fabric_implementation` model repo.

There are two sets of scripts:

- **`sd_*.py` — SD-1.5 (`runwayml/stable-diffusion-v1-5`), this is what we run on.**
  512×512, latent `(4, 64, 64)`, DDIM, modified noise handed to the pipeline via
  `latents=`. Shared SD helpers live in `sd_common.py`.
- The original FLUX scripts (no prefix) are kept for reference.

## SD-1.5 scripts (primary)

| Script | What it does |
|---|---|
| `sd_run_motivation_collages.py` | Motivation grids: `alpha`, `base/c`, `affine_v1`, `affine_v2` sweeps + ImageReward under each image (11 collages A1–W3) |
| `sd_visualize_all_parametrizations_real.py` | All 8 DCT parametrizations (dct_affine, power_law, log_bands, chebyshev, bspline, rbf, dct_affine_signed, chebyshev_phase) over 4 prompt types |
| `sd_visualize_signed_params.py` | Signed parametrizations that flip low-freq coefficients (dct_affine_signed, chebyshev_phase) + sign-curve plots |

```bash
python sd_run_motivation_collages.py --save_dir results/sd_motivation1
python sd_run_motivation_collages.py --skip_ir                 # faster, no ImageReward
python sd_visualize_all_parametrizations_real.py --save_dir results/sd_all_parametrizations
python sd_visualize_all_parametrizations_real.py --prompts portrait --methods dct_affine
python sd_visualize_signed_params.py --prompt "a dragon flying over a medieval castle"
```

SD knobs (model, steps, CFG, image size, seed) live at the top of `sd_common.py`.

## FLUX scripts (reference)

| Script | Model | What it does |
|---|---|---|
| `run_motivation_collages.py` | FLUX.1-dev (28 steps, CFG 3.5) | Motivation grids: `alpha`, `base/c`, `affine_v1`, `affine_v2` sweeps + ImageReward under each image (11 collages: A1–A3, B1–B3, V1a/V1b, W1–W3) |
| `visualize_all_parametrizations_real.py` | FLUX.1-schnell (4 steps) | All 8 DCT parametrizations (dct_affine, power_law, log_bands, chebyshev, bspline, rbf, dct_affine_signed, chebyshev_phase) over 4 prompt types, EXPANDED ranges |
| `visualize_signed_params.py` | FLUX.1-schnell (4 steps) | Signed parametrizations that can flip low-freq coefficients (dct_affine_signed, chebyshev_phase) + sign-curve plots |

All three modify the **initial latent noise** in the DCT domain on the `K`
lowest radial frequencies, renormalize per-channel norm, then run the FLUX
sampler and lay the results out in labeled collages.

## Setup

```bash
pip install -r requirements.txt
# FLUX.1-dev is gated; log in to Hugging Face for run_motivation_collages.py:
huggingface-cli login
```

## Run

```bash
# 1) motivation collages (alpha / base / affine grids, with ImageReward)
python run_motivation_collages.py --save_dir results/motivation1
python run_motivation_collages.py --skip_ir          # faster, no ImageReward

# 2) all 8 parametrizations on real FLUX images
python visualize_all_parametrizations_real.py --save_dir results/all_parametrizations
python visualize_all_parametrizations_real.py --prompts portrait --methods dct_affine

# 3) signed parametrizations
python visualize_signed_params.py --prompt "a dragon flying over a medieval castle"
```

## Notes on thresholds

Parameter ranges are the "thresholds" to tune per model. They live inline:

- `*_run_motivation_collages.py`: the `alphas`, `t` value lists, `a_sweep`, etc. in `run_all()`.
- `*_visualize_all_parametrizations_real.py`: the `test_values` field of each entry in the `METHODS` dict.
- `*_visualize_signed_params.py`: the value lists in the per-row loops in `main()`.

For SD: `SEED`, `STEPS`, `CFG`, `H/W` are in `sd_common.py`; `K` is at the top of
`sd_run_motivation_collages.py` / `K_AFFINE` in the visualize scripts.
