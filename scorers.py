#!/usr/bin/env python3
"""
scorers.py
----------
Unified scorer wrapper so the experiment scripts can switch between
ImageReward and HPSv3 with one consistent API:

    scorer = load_scorer("hpsv3", device="cuda")
    s = score_one(scorer, pil_image, prompt)     # -> float

Important: the two underlying libraries take their arguments in OPPOSITE
order (ImageReward.score(prompt, img) vs HPSv3.reward(images, prompts)).
This wrapper hides that so callers never get it silently wrong.

The HPSv3 path mirrors fabric/hpsv3_scorer.py from the model repo
(fabric_implementation) so test_method stays self-contained and uses the
SAME reward model / call convention as methods A, B, C.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from PIL import Image


# ---- HPSv3 (copied convention from fabric_implementation/fabric/hpsv3_scorer.py)

_HPSV3_INFERENCER: Any = None
_HPSV3_API: str | None = None


def _extract_first_scalar(obj: Any) -> float:
    import torch
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        flat = obj.reshape(-1)
        if flat.numel() == 0:
            raise ValueError("Empty tensor in HPSv3 result")
        return float(flat[0].item())
    if hasattr(obj, "ndim") and hasattr(obj, "flatten"):
        flat = obj.flatten()
        if len(flat) == 0:
            raise ValueError("Empty array in HPSv3 result")
        return float(flat[0])
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise ValueError("Empty sequence in HPSv3 result")
        return _extract_first_scalar(obj[0])
    if hasattr(obj, "item"):
        try:
            return float(obj.item())
        except (ValueError, RuntimeError):
            pass
    raise TypeError(f"Cannot extract scalar from HPSv3 result {type(obj)}: {obj!r}")


class HPSv3:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        global _HPSV3_INFERENCER
        if _HPSV3_INFERENCER is None:
            from hpsv3 import HPSv3RewardInferencer
            _HPSV3_INFERENCER = HPSv3RewardInferencer(device=device)
        self.model = _HPSV3_INFERENCER

    def _call_reward(self, image_path: str, prompt: str):
        global _HPSV3_API
        if _HPSV3_API in (None, "two_lists"):
            try:
                res = self.model.reward([image_path], [prompt])
                _HPSV3_API = "two_lists"
                return res
            except TypeError:
                if _HPSV3_API == "two_lists":
                    raise
        if _HPSV3_API in (None, "dict_list"):
            try:
                res = self.model.reward([{"image_path": [image_path], "prompt": prompt}])
                _HPSV3_API = "dict_list"
                return res
            except TypeError:
                if _HPSV3_API == "dict_list":
                    raise
        res = self.model.reward(image_paths=[image_path], prompts=[prompt])
        _HPSV3_API = "kwargs"
        return res

    def score(self, img: Image.Image, prompt: str) -> float:
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cand.png")
            img.convert("RGB").save(path)
            with torch.no_grad():
                res = self._call_reward(path, prompt)
        return _extract_first_scalar(res)


# ---- ImageReward -------------------------------------------------------------

class ImageRewardScorer:
    def __init__(self, device: str = "cuda") -> None:
        import ImageReward as RM
        self.model = RM.load("ImageReward-v1.0", device=device)

    def score(self, img: Image.Image, prompt: str) -> float:
        return float(self.model.score(prompt, img))   # note: (prompt, img)


# ---- factory -----------------------------------------------------------------

def load_scorer(name: str, device: str = "cuda"):
    name = (name or "none").lower()
    if name in ("none", "off", ""):
        return None
    if name in ("hpsv3", "hps", "hps3"):
        return HPSv3(device=device)
    if name in ("imagereward", "ir", "image_reward"):
        return ImageRewardScorer(device=device)
    raise ValueError(f"Unknown scorer {name!r} (use none|hpsv3|imagereward)")


def score_one(scorer, img: Image.Image, prompt: str) -> float:
    """Consistent (image, prompt) -> float, regardless of backend."""
    if scorer is None:
        return 0.0
    return float(scorer.score(img, prompt))
