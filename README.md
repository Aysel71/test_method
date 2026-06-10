# test_method — DCT noise parametrization experiments

Noise-modification experiments on FLUX: vary DCT alpha coefficients in front of
the low-frequency coefficients, build grids in several ways, and sweep the
transformation itself. Companion to the `fabric_implementation` model repo.

## Scripts

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

- `run_motivation_collages.py`: the `alphas`, `t_vals`, `a_sweep`, etc. lists in `run_all()`.
- `visualize_all_parametrizations_real.py`: the `test_values` field of each entry in the `METHODS` dict.
- `visualize_signed_params.py`: the value lists in the per-row loops in `main()`.

`SEED`, `STEPS`, `H/W`, and `K` are constants at the top of each file.
