"""Tests for TTA inference (HFlip + 3-crop).

Spec: §5.5.
"""
from __future__ import annotations

import torch
from src.v2.model import NutritionRGBDModel
from src.v2.tta import tta_predict


def test_tta_returns_correct_shapes():
    m = NutritionRGBDModel(n_ingredients=555).eval()
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 2, 224, 224)
    out = tta_predict(m, rgb, depth, use_depth=True)
    assert out["scalar"].shape == (2, 5)
    assert out["ingr_logits"].shape == (2, 555)
    assert out["ingr_mass"].shape == (2, 555)


def test_tta_deterministic_in_eval_mode():
    m = NutritionRGBDModel(n_ingredients=555).eval()
    rgb = torch.randn(1, 3, 224, 224)
    depth = torch.randn(1, 2, 224, 224)
    a = tta_predict(m, rgb, depth)
    b = tta_predict(m, rgb, depth)
    for k in a:
        assert torch.allclose(a[k], b[k])
