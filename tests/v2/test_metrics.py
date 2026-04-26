from __future__ import annotations
import math
import numpy as np
from src.v2.metrics import (
    mae,
    pct_mae,
    multilabel_f1_micro_macro,
    top_k_set_iou,
    paired_bootstrap_mae,
)


def test_mae_zero_when_equal():
    a = np.array([1.0, 2.0, 3.0])
    assert mae(a, a) == 0.0


def test_mae_value():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 5.0])
    assert math.isclose(mae(a, b), 2.5)


def test_pct_mae_uses_target_mean():
    p = np.array([100.0, 200.0])
    t = np.array([110.0, 190.0])
    # MAE = 10; mean(t) = 150 -> 10/150 *100 = 6.6667
    assert math.isclose(pct_mae(p, t), 100 * 10 / 150)


def test_f1_perfect():
    pred = np.array([[1, 1, 0], [0, 1, 0]])
    targ = np.array([[1, 1, 0], [0, 1, 0]])
    micro, macro = multilabel_f1_micro_macro(pred, targ)
    assert micro == 1.0 and macro >= 1.0 - 1e-6


def test_top_k_iou():
    pred = np.array([[10.0, 9.0, 8.0, 0.0]])
    targ = np.array([[1, 0, 1, 0]])
    iou = top_k_set_iou(pred, targ, k=2)
    # Top-2 pred: {0, 1}; GT positives: {0, 2}. Intersection {0}, union {0,1,2}: 1/3
    assert math.isclose(iou.mean(), 1 / 3)


def test_paired_bootstrap_mae_shape():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    b = rng.normal(size=100)
    t = rng.normal(size=100)
    delta_mean, ci_low, ci_high = paired_bootstrap_mae(a, b, t, n=200, seed=0)
    assert ci_low <= delta_mean <= ci_high
