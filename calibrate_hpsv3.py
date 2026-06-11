#!/usr/bin/env python3
"""
calibrate_hpsv3.py
------------------
Pick parameter boundaries ("thresholds") for each DCT parametrization on
SD-1.5, scored by HPSv3.

For every method it sweeps its parameter over a dense, deliberately-too-wide
grid, generates an SD-1.5 image per (value, prompt, seed), scores each with
HPSv3, averages over prompts/seeds, and then:

  - saves a per-method CSV (value, mean_hps, std_hps)
  - saves a per-method plot (HPSv3 vs value, neutral + suggested range marked)
  - suggests a contiguous [lo, hi] range = the widest interval around the
    neutral value where mean HPSv3 stays within `--margin` of the neutral
    (identity) baseline, and reports the global argmax value.
  - writes suggested_test_values.json you can paste back into the collage scripts.

This is the script to run on the GPU server. Locally we then read the CSV/JSON
and update test_values.

Run:
    python calibrate_hpsv3.py --save_dir results/calib_hpsv3 \
        --n_values 15 --n_seeds 1 --margin 0.5
    # subset:
    python calibrate_hpsv3.py --methods dct_affine power_law --prompts portrait object
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sd_common import (
    load_pipe, get_base_latent, generate_from_latent,
    get_radial, get_low_idx,
)
from scorers import load_scorer, score_one
from sd_visualize_all_parametrizations_real import METHODS, K_AFFINE


# Calibration prompts (small, diverse). Override with --prompts_file.
PROMPTS = {
    "portrait":  "a portrait of a woman in golden light, soft focus, cinematic lighting",
    "landscape": "a snowy mountain landscape at sunrise, dramatic clouds, ultra-detailed",
    "object":    "a cup of coffee on a wooden table, morning light, cozy atmosphere",
    "fantasy":   "a dragon flying over a medieval castle, epic fantasy, thunderstorm",
}


# Per-method: (sweep_lo, sweep_hi, neutral_value). The sweep is intentionally
# wider than the current collage test_values so we can see where HPSv3 breaks.
CALIB = {
    "dct_affine":        (0.0,   4.0,  1.0),
    "power_law":         (-3.0,  3.0,  0.0),
    "log_bands":         (0.0,   4.0,  1.0),
    "chebyshev":         (-1.2,  1.2,  0.0),
    "bspline":           (-1.2,  1.2,  0.0),
    "rbf":               (-1.2,  1.2,  0.0),
    "dct_affine_signed": (-6.0,  6.0,  1.0),
    "chebyshev_phase":   (-15.0, 15.0, 5.0),   # neutral c0=+5 => tanh~+1 (identity)
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default="results/calib_hpsv3")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--prompts", nargs="+", default=None,
                   help="which prompt keys to average over (default: all)")
    p.add_argument("--prompts_file", type=str, default=None,
                   help="one prompt per line; overrides the built-in set")
    p.add_argument("--n_values", type=int, default=15,
                   help="number of sweep points per method")
    p.add_argument("--n_seeds", type=int, default=1,
                   help="base-noise seeds to average over (42, 43, ...)")
    p.add_argument("--margin", type=float, default=0.5,
                   help="HPSv3 drop (abs) below neutral still counted as 'valid'")
    p.add_argument("--scorer", type=str, default="hpsv3",
                   help="hpsv3 | imagereward")
    return p.parse_args()


def suggest_range(values, mean_hps, neutral_val, margin):
    """Widest contiguous interval around neutral where mean_hps >= baseline-margin."""
    n_idx = int(np.argmin(np.abs(values - neutral_val)))
    baseline = mean_hps[n_idx]
    thr = baseline - margin
    lo_i = n_idx
    while lo_i - 1 >= 0 and mean_hps[lo_i - 1] >= thr:
        lo_i -= 1
    hi_i = n_idx
    while hi_i + 1 < len(values) and mean_hps[hi_i + 1] >= thr:
        hi_i += 1
    best_i = int(np.argmax(mean_hps))
    return {
        "neutral_value": float(neutral_val),
        "baseline_hps": float(baseline),
        "threshold": float(thr),
        "suggested_lo": float(values[lo_i]),
        "suggested_hi": float(values[hi_i]),
        "argmax_value": float(values[best_i]),
        "argmax_hps": float(mean_hps[best_i]),
    }


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.prompts_file:
        lines = [l.strip() for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]
        prompt_items = [(f"p{i:02d}", t) for i, t in enumerate(lines)]
    else:
        keys = args.prompts or list(PROMPTS.keys())
        prompt_items = [(k, PROMPTS[k]) for k in keys]

    methods = args.methods or list(CALIB.keys())
    seeds = [42 + i for i in range(args.n_seeds)]

    print("=" * 70)
    print(f"HPSv3 boundary calibration on SD-1.5")
    print(f"  methods : {methods}")
    print(f"  prompts : {[k for k, _ in prompt_items]}")
    print(f"  seeds   : {seeds}   values/method: {args.n_values}")
    print("=" * 70)

    pipe = load_pipe()
    scorer = load_scorer(args.scorer, device="cuda")

    r = get_radial()
    low_idx = get_low_idx(r, K_AFFINE)
    z_bases = {s: get_base_latent(s) for s in seeds}

    suggestions = {}

    for method in methods:
        lo, hi, neutral = CALIB[method]
        values = np.linspace(lo, hi, args.n_values)
        mc = METHODS[method]
        print(f"\n[{method}]  sweep {lo}..{hi}  neutral={neutral}")

        rows = []          # (value, mean, std)
        per_value_scores = {}
        for val in values:
            theta = mc["base_theta"](float(val))
            scores = []
            for s in seeds:
                z_base = z_bases[s]
                if mc["needs_low_idx"]:
                    z_mod = mc["func"](z_base, theta, low_idx)
                else:
                    z_mod = mc["func"](z_base, theta, r)
                for _, ptext in prompt_items:
                    img = generate_from_latent(pipe, z_mod, ptext)
                    scores.append(score_one(scorer, img, ptext))
            m, sd = float(np.mean(scores)), float(np.std(scores))
            rows.append((float(val), m, sd))
            per_value_scores[f"{val:.4f}"] = scores
            print(f"    {val:+7.3f}  HPS={m:+.3f} ± {sd:.3f}")

        values_arr = np.array([x[0] for x in rows])
        mean_arr = np.array([x[1] for x in rows])
        std_arr = np.array([x[2] for x in rows])

        # CSV
        csv_path = save_dir / f"{method}.csv"
        with open(csv_path, "w") as f:
            f.write("value,mean_hps,std_hps\n")
            for v, m, sd in rows:
                f.write(f"{v:.6f},{m:.6f},{sd:.6f}\n")

        # suggestion
        sug = suggest_range(values_arr, mean_arr, neutral, args.margin)
        suggestions[method] = sug
        print(f"  -> suggested range [{sug['suggested_lo']:.3f}, {sug['suggested_hi']:.3f}]  "
              f"argmax={sug['argmax_value']:.3f} (HPS {sug['argmax_hps']:+.3f})")

        # plot
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(values_arr, mean_arr, yerr=std_arr, fmt="-o", capsize=3, lw=2)
        ax.axvline(sug["neutral_value"], color="gray", ls="--", label="neutral")
        ax.axhline(sug["baseline_hps"], color="gray", ls=":", alpha=0.7)
        ax.axhline(sug["threshold"], color="red", ls=":", alpha=0.7,
                   label=f"baseline-margin ({args.margin})")
        ax.axvspan(sug["suggested_lo"], sug["suggested_hi"], color="green",
                   alpha=0.12, label="suggested range")
        ax.axvline(sug["argmax_value"], color="green", lw=1.5, label="argmax")
        ax.set_title(f"{method}: HPSv3 vs {mc['param_name']}")
        ax.set_xlabel(mc["param_name"]); ax.set_ylabel("mean HPSv3")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"{method}.png", dpi=130, facecolor="white")
        plt.close(fig)

    # suggested test_values: 7 points spanning the suggested range, neutral included
    out = {}
    for method, sug in suggestions.items():
        lo, hi = sug["suggested_lo"], sug["suggested_hi"]
        pts = sorted(set(np.round(np.linspace(lo, hi, 7), 3)) | {round(sug["neutral_value"], 3)})
        out[method] = {
            "suggested_range": [lo, hi],
            "argmax_value": sug["argmax_value"],
            "test_values": [float(x) for x in pts],
            "baseline_hps": sug["baseline_hps"],
            "argmax_hps": sug["argmax_hps"],
        }
    with open(save_dir / "suggested_test_values.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nDone. Per-method CSV/PNG + suggested_test_values.json in {save_dir}/")
    print("Send suggested_test_values.json back and we'll update the collage scripts.")


if __name__ == "__main__":
    main()
