#!/usr/bin/env python3
"""
sd_fine_grids.py
----------------
Dense, fixed-STEP parameter grids for every DCT parametrization on SD-1.5,
saved into a SEPARATE folder (default results/sd_fine_grids), with the HPSv3
score printed under each image.

Same parametrizations / generation / collage as
sd_visualize_all_parametrizations_real.py — the only difference is that each
method is swept on a fine `np.arange(lo, hi, step)` grid instead of the
hand-picked test_values.

E.g. chebyshev_phase: c_0 in [-5, 15] step 1  -> 21 images per prompt.

Run:
    python sd_fine_grids.py --scorer hpsv3 --save_dir results/sd_fine_grids
    python sd_fine_grids.py --methods chebyshev_phase --prompts fantasy
    python sd_fine_grids.py --step 0.5 --methods dct_affine     # override step
"""

import argparse
from pathlib import Path

import numpy as np

from sd_common import (
    load_pipe, get_base_latent, generate_from_latent,
    get_radial, get_low_idx, SEED,
)
from scorers import load_scorer, score_one
from sd_visualize_all_parametrizations_real import (
    METHODS, K_AFFINE, PROMPTS, make_collage,
)


# Per-method fine grid: (lo, hi, step). Step chosen to suit each range.
FINE = {
    'dct_affine':        (0.0,  2.3,  0.1),
    'power_law':         (-1.0, 1.0,  0.1),
    'log_bands':         (0.0,  1.45, 0.1),
    'chebyshev':         (-1.2, 1.2,  0.1),
    'bspline':           (-1.2, 0.2,  0.1),
    'rbf':               (-0.4, 0.55, 0.05),
    'dct_affine_signed': (-2.0, 2.0,  0.2),
    'chebyshev_phase':   (-5.0, 15.0, 1.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default="results/sd_fine_grids")
    p.add_argument("--prompts",  nargs="+", default=None)
    p.add_argument("--methods",  nargs="+", default=None)
    p.add_argument("--scorer",   type=str, default="hpsv3",
                   help="none | hpsv3 | imagereward")
    p.add_argument("--step",     type=float, default=None,
                   help="override the per-method step for ALL selected methods")
    p.add_argument("--ncols",    type=int, default=None,
                   help="wrap the grid into rows of this many columns (default: one row)")
    return p.parse_args()


def fine_values(method, step_override):
    lo, hi, step = FINE[method]
    if step_override is not None:
        step = step_override
    vals = np.arange(lo, hi + step / 2.0, step)
    return [round(float(v), 4) for v in vals]


def chunk(items, n):
    if not n or n <= 0 or n >= len(items):
        return [items]
    return [items[i:i + n] for i in range(0, len(items), n)]


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    prompts_to_run = PROMPTS
    if args.prompts:
        prompts_to_run = [p for p in PROMPTS if p[0] in args.prompts]

    methods_to_run = list(FINE.keys())
    if args.methods:
        methods_to_run = [m for m in methods_to_run if m in args.methods]

    grids = {m: fine_values(m, args.step) for m in methods_to_run}
    total = len(prompts_to_run) * sum(len(v) for v in grids.values())
    print("=" * 70)
    print(f"SD-1.5 FINE grids -> {save_dir}/")
    for m in methods_to_run:
        v = grids[m]
        print(f"  {m:20s} {FINE[m][0]}..{FINE[m][1]} step {args.step or FINE[m][2]}"
              f"  -> {len(v)} values")
    print(f"  prompts: {[p[0] for p in prompts_to_run]}   total gens ~{total}")
    print("=" * 70)

    print("\nLoading SD-1.5 ...")
    pipe = load_pipe()
    scorer = load_scorer(args.scorer, device="cuda")

    r = get_radial()
    low_idx = get_low_idx(r, K_AFFINE)
    z_base = get_base_latent(SEED)
    print(f"  latent shape = {z_base.shape}\n")

    for p_label, p_text in prompts_to_run:
        print(f"\n{'='*70}\nPrompt: [{p_label}] {p_text[:60]}\n{'='*70}")

        baseline_img = generate_from_latent(pipe, z_base, p_text)
        baseline_img.save(save_dir / f"{p_label}_baseline.png")

        for method_name in methods_to_run:
            mc = METHODS[method_name]
            values = grids[method_name]
            print(f"[{p_label}] {method_name}  ({len(values)} values)")
            col_items = []
            for val in values:
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

            rows = [(f"{mc['param_name']} row{ri+1}", ch)
                    for ri, ch in enumerate(chunk(col_items, args.ncols))]
            make_collage(
                f"{mc['title']} [{p_label}] FINE step={args.step or FINE[method_name][2]}",
                p_text, rows,
                save_dir / f"{p_label}_{method_name}_fine.png",
                mc['subtitle'],
            )

        print(f"  [{p_label}] done.")

    print(f"\nAll fine grids saved to {save_dir}/")


if __name__ == "__main__":
    main()
