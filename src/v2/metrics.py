"""Evaluation metrics. All numpy, no torch.

Spec: §6.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def pct_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """100 × MAE / mean(target). Returns Inf if target is all zeros."""
    m = float(np.mean(target))
    if m == 0:
        return float("inf")
    return 100.0 * mae(pred, target) / m


def multilabel_f1_micro_macro(pred_binary: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Inputs are (N, V) 0/1 arrays.

    micro F1: aggregate TP/FP/FN over all (n,v); macro: per-class F1, mean over classes
    with ≥1 positive in target.
    """
    pred = pred_binary.astype(bool)
    targ = target.astype(bool)
    tp = (pred & targ).sum(axis=0)
    fp = (pred & ~targ).sum(axis=0)
    fn = (~pred & targ).sum(axis=0)
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_p = micro_tp / max(micro_tp + micro_fp, 1)
    micro_r = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12)

    per_p = tp / np.maximum(tp + fp, 1)
    per_r = tp / np.maximum(tp + fn, 1)
    per_f1 = np.where(
        (per_p + per_r) > 0,
        2 * per_p * per_r / np.maximum(per_p + per_r, 1e-12),
        0.0,
    )
    # Average over classes that have ≥1 positive in target
    valid = target.sum(axis=0) > 0
    macro_f1 = float(per_f1[valid].mean()) if valid.any() else 0.0
    return float(micro_f1), float(macro_f1)


def top_k_set_iou(scores: np.ndarray, target_binary: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-row IoU between top-k predicted indices and GT-positive indices.

    Returns (N,) array.
    """
    n, v = scores.shape
    if k > v:
        raise ValueError(f"k={k} cannot exceed vocab size v={v}")
    top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    iou = np.zeros(n, dtype=np.float64)
    for i in range(n):
        pred_set = set(top_idx[i].tolist())
        gt_set = set(np.where(target_binary[i] > 0)[0].tolist())
        if not pred_set and not gt_set:
            iou[i] = 1.0
            continue
        inter = pred_set & gt_set
        uni = pred_set | gt_set
        iou[i] = len(inter) / max(len(uni), 1)
    return iou


def paired_bootstrap_mae(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    n: int = 1000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Paired bootstrap on per-sample absolute errors.

    Returns (delta_mae_mean, low95, high95) where delta = MAE(b) - MAE(a).
    Negative delta means b is better.

    Paired: the same resample indices are used for both err_a and err_b.
    """
    err_a = np.abs(pred_a - target)
    err_b = np.abs(pred_b - target)
    rng = np.random.default_rng(seed)
    N = len(target)
    deltas = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, N, N)
        deltas[i] = err_b[idx].mean() - err_a[idx].mean()
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def bootstrap_ci_mae(
    pred: np.ndarray,
    target: np.ndarray,
    n: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Single-population bootstrap of MAE: returns (low95, high95)."""
    err = np.abs(pred - target)
    rng = np.random.default_rng(seed)
    N = len(target)
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, N, N)
        vals[i] = err[idx].mean()
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
