"""Sanity & qualitative visualizations.

Saves:
    G1 sanity: 5-dish RGB+Depth+labels panel
    G4 scatter: pred vs GT for 5 scalars
    Final qualitative: best/worst 5 per-dish panels

Spec: §10.3, G1, G4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image


def sanity_panel(
    samples: Sequence[dict],
    out_path: Path | str,
) -> None:
    """One row per sample: [RGB | Depth | label table]. samples = list of __getitem__ outputs."""
    n = len(samples)
    fig = plt.figure(figsize=(12, 3 * n))
    gs = GridSpec(n, 3, figure=fig, width_ratios=[1, 1, 1.4])
    for i, s in enumerate(samples):
        rgb = s["rgb"].permute(1, 2, 0).numpy()
        rgb = (rgb - rgb.min()) / max(rgb.ptp(), 1e-6)
        depth = s["depth"][0].numpy()
        ax_r = fig.add_subplot(gs[i, 0]); ax_r.imshow(rgb); ax_r.set_title(f"{s['dish_id']} rgb"); ax_r.axis("off")
        ax_d = fig.add_subplot(gs[i, 1]); ax_d.imshow(depth, cmap="viridis"); ax_d.set_title("depth (z)"); ax_d.axis("off")
        ax_l = fig.add_subplot(gs[i, 2]); ax_l.axis("off")
        kcal, mass, fat, carb, prot = s["y_scalar_raw"].tolist()
        n_ingr = int(s["y_ingr_mask"].sum())
        ax_l.text(0.0, 0.9, f"kcal={kcal:.0f}  mass={mass:.0f}g", fontsize=10, family="monospace")
        ax_l.text(0.0, 0.6, f"fat={fat:.1f}  carb={carb:.1f}  prot={prot:.1f}", fontsize=10, family="monospace")
        ax_l.text(0.0, 0.3, f"ingredients: {n_ingr}", fontsize=10, family="monospace")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def scatter_pred_vs_gt(
    preds: np.ndarray, targets: np.ndarray, names: Sequence[str], out_path: Path | str,
) -> None:
    """N-column scatter plots, one per metric name."""
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for i, name in enumerate(names):
        ax = axes[i]
        ax.scatter(targets[:, i], preds[:, i], s=6, alpha=0.4)
        m = max(targets[:, i].max(), preds[:, i].max())
        ax.plot([0, m], [0, m], "r--", lw=0.5)
        ax.set_xlabel("GT"); ax.set_ylabel("pred"); ax.set_title(name)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
