#!/usr/bin/env python3
"""
sd_common.py
------------
Shared SD-1.5 helpers for the noise-parametrization collage scripts.

SD-1.5 (runwayml/stable-diffusion-v1-5) latent space:
  512x512 image  ->  latent (4, 64, 64), VAE scaling_factor 0.18215.

Unlike the FLUX scripts there is no packing/unpacking: we modify the initial
N(0, I) latent in the DCT domain, renormalize per-channel norm, and hand it to
StableDiffusionPipeline via `latents=`. The default scheduler is deterministic
(DDIM), so the same modified latent always maps to the same image.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.fft import dctn, idctn
from diffusers import StableDiffusionPipeline, DDIMScheduler


# ---- SD-1.5 config -----------------------------------------------------------

MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE   = "cuda"
DTYPE    = torch.float16
H, W     = 512, 512          # image size
LH, LW   = H // 8, W // 8    # latent size = 64 x 64
C        = 4                 # latent channels
STEPS    = 50
CFG      = 7.5
SEED     = 42


def load_pipe(model_id=MODEL_ID, device=DEVICE):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=DTYPE,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_base_latent(seed=SEED):
    """Base N(0, I) latent (C, LH, LW) as float32 numpy."""
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(C, LH, LW, generator=g).numpy().astype(np.float32)


@torch.no_grad()
def generate_from_latent(pipe, z_np, prompt, steps=STEPS, cfg=CFG):
    """Run SD-1.5 from a (C, LH, LW) latent, return a PIL image."""
    lat = torch.tensor(z_np[None], device=pipe.device, dtype=DTYPE)
    out = pipe(prompt, latents=lat, num_inference_steps=steps,
               guidance_scale=cfg, height=H, width=W)
    return out.images[0]


# ---- DCT utilities (shared with the FLUX scripts) ----------------------------

def get_radial(h=LH, w=LW):
    fy = np.arange(h, dtype=np.float32)
    fx = np.arange(w, dtype=np.float32)
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).flatten()


def get_low_idx(r, k):
    return np.argsort(r)[:k]


def renormalize(z_new, z_orig):
    for ch in range(z_new.shape[0]):
        scale = np.linalg.norm(z_orig[ch]) / (np.linalg.norm(z_new[ch]) + 1e-9)
        z_new[ch] *= scale
    return z_new
