"""Test-time augmentation: HFlip + 3-crop (center, top-left, bottom-right).

Spec: §5.5.

Inputs are pre-normalized 224×224 tensors (eval-time CenterCrop already applied).
Therefore "3-crop" is implemented by slicing 192×192 sub-regions and resizing
back to 224 — this is a pragmatic in-tensor TTA that doesn't require re-doing
the file IO. For exact paper-style 3-crop, evaluate on 256-resized + crop pre-
transform; we keep things simple and consistent with our pipeline.

6 forward passes total: 3 crops × 2 flip directions (original + horizontal flip).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _three_crops(x: torch.Tensor, crop: int = 192, target: int = 224) -> list[torch.Tensor]:
    """Returns [center, top-left, bottom-right] each bilinear-upsampled back to target×target."""
    H, W = x.shape[-2:]
    cy = (H - crop) // 2
    cx = (W - crop) // 2
    cuts = [
        x[..., cy:cy + crop, cx:cx + crop],          # center
        x[..., :crop, :crop],                          # top-left
        x[..., H - crop:, W - crop:],                 # bottom-right
    ]
    return [F.interpolate(c, size=target, mode="bilinear", align_corners=False) for c in cuts]


@torch.no_grad()
def tta_predict(
    model,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    *,
    use_depth: bool = True,
) -> Dict[str, torch.Tensor]:
    """Average model outputs over {center, TL, BR} × {original, hflip} = 6 forward passes.

    Returns the same dict shape as model.forward:
        {"scalar": (B,5), "ingr_logits": (B, n_ingr), "ingr_mass": (B, n_ingr)}
    """
    model.eval()

    crops_rgb = _three_crops(rgb)
    crops_d = _three_crops(depth)

    # Build the 6-view lists: original crop then horizontally flipped crop
    rgb_list: list[torch.Tensor] = []
    d_list: list[torch.Tensor] = []
    for r, d in zip(crops_rgb, crops_d):
        rgb_list.append(r)
        d_list.append(d)
        rgb_list.append(torch.flip(r, dims=[-1]))
        d_list.append(torch.flip(d, dims=[-1]))

    accum: Dict[str, float | torch.Tensor] = {
        "scalar": 0.0,
        "ingr_logits": 0.0,
        "ingr_mass": 0.0,
    }
    for r, d in zip(rgb_list, d_list):
        out = model(r, d, use_depth=use_depth)
        for k in accum:
            accum[k] = accum[k] + out[k]

    n = len(rgb_list)  # 6
    return {k: v / n for k, v in accum.items()}
