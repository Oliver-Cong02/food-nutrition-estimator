# src/v2/evaluate.py
"""Eval loop: produces predictions.csv, groundtruth.csv, eval_results.json.

Usage:
    python -m src.v2.evaluate \
        --checkpoint checkpoints/v2/<run_id>/ema.pt \
        --vocab      checkpoints/v2/<run_id>/vocab.json \
        --stats      checkpoints/v2/<run_id>/train_stats.json \
        --split-file data/raw/dish_ids/splits/rgb_test_ids.txt \
        --output-dir docs/runs/<run_id>/eval/

Spec: §6.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Nutrition5kRGBD, build_default_eval_transform
from .metrics import (
    mae, pct_mae, multilabel_f1_micro_macro, top_k_set_iou, bootstrap_ci_mae,
)
from .model import NutritionRGBDModel
from .stats import TrainStats
from .tta import tta_predict
from .vocab import Vocab


def _resolve_dish_ids(split_file: Path, available_ids: set[str]) -> list[str]:
    ids = [ln.strip() for ln in Path(split_file).read_text().splitlines() if ln.strip()]
    return [i for i in ids if i in available_ids]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(args.vocab)
    stats = TrainStats.load(args.stats)

    available_ids = set([ln.strip() for ln in Path(args.available_dish_ids).read_text().splitlines() if ln.strip()])
    test_ids = _resolve_dish_ids(Path(args.split_file), available_ids)
    print(f"Eval on {len(test_ids)} dishes")
    if len(test_ids) == 0:
        raise RuntimeError(
            "No test dishes found — check --split-file and --available-dish-ids paths."
        )

    ds = Nutrition5kRGBD(
        dish_ids=test_ids,
        metadata_csvs=[args.metadata_cafe1, args.metadata_cafe2],
        imagery_root=args.imagery_root,
        vocab=vocab,
        stats=stats,
        transform=build_default_eval_transform(),
        require_depth=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = NutritionRGBDModel(n_ingredients=vocab.size).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)

    densities = torch.tensor(vocab.idx_to_density, dtype=torch.float32, device=device)
    scalar_mean = torch.tensor(stats.scalar_mean, dtype=torch.float32, device=device)
    scalar_std  = torch.tensor(stats.scalar_std, dtype=torch.float32, device=device)

    preds_scalar_raw = []   # (N, 5) raw kcal/mass/fat/carb/protein
    preds_kcal_direct = []
    preds_kcal_derived = []
    preds_ingr_logits = []
    preds_ingr_mass_raw = []   # raw grams per ingredient
    targs_scalar_raw = []
    targs_ingr_binary = []
    targs_ingr_mass_raw = []
    targs_ingr_mask = []
    dish_ids = []

    use_depth = not args.no_depth
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            out = tta_predict(model, rgb, depth, use_depth=use_depth)
            scalar_z = out["scalar"]                              # (B, 5) z-scored
            ingr_mass_z = out["ingr_mass"]                        # (B, V) log1p+z
            scalar_raw = scalar_z * scalar_std + scalar_mean      # (B, 5) raw kcal/mass/fat/carb/protein
            kcal_direct = scalar_raw[:, 0]                        # (B,)
            ingr_mass_raw = torch.expm1(
                ingr_mass_z * stats.mass_log1p_std + stats.mass_log1p_mean
            ).clamp(min=0.0)
            kcal_derived = (ingr_mass_raw * densities[None, :]).sum(dim=1)
            preds_scalar_raw.append(scalar_raw.cpu().numpy())
            preds_kcal_direct.append(kcal_direct.cpu().numpy())
            preds_kcal_derived.append(kcal_derived.cpu().numpy())
            preds_ingr_logits.append(out["ingr_logits"].cpu().numpy())
            preds_ingr_mass_raw.append(ingr_mass_raw.cpu().numpy())
            targs_scalar_raw.append(batch["y_scalar_raw"].numpy())
            targs_ingr_binary.append(batch["y_ingr_binary"].numpy())
            # Targets for ingredient mass need inverse-transform too
            ymass_z = batch["y_ingr_mass"].numpy()
            ymass_mask = batch["y_ingr_mask"].numpy()
            ymass_raw = np.where(
                ymass_mask > 0,
                np.expm1(ymass_z * stats.mass_log1p_std + stats.mass_log1p_mean),
                0.0,
            )
            targs_ingr_mass_raw.append(ymass_raw)
            targs_ingr_mask.append(ymass_mask)
            dish_ids.extend(batch["dish_id"])

    preds_scalar_raw = np.concatenate(preds_scalar_raw, axis=0)
    preds_kcal_direct = np.concatenate(preds_kcal_direct, axis=0)
    preds_kcal_derived = np.concatenate(preds_kcal_derived, axis=0)
    preds_ingr_logits = np.concatenate(preds_ingr_logits, axis=0)
    preds_ingr_mass_raw = np.concatenate(preds_ingr_mass_raw, axis=0)
    targs_scalar_raw = np.concatenate(targs_scalar_raw, axis=0)
    targs_ingr_binary = np.concatenate(targs_ingr_binary, axis=0)
    targs_ingr_mass_raw = np.concatenate(targs_ingr_mass_raw, axis=0)
    targs_ingr_mask = np.concatenate(targs_ingr_mask, axis=0)

    # Headline kcal: 50/50 average
    kcal_avg = 0.5 * preds_kcal_direct + 0.5 * preds_kcal_derived

    # Replace direct-only kcal in scalar_raw[:,0] with the averaged version
    preds_scalar_for_report = preds_scalar_raw.copy()
    preds_scalar_for_report[:, 0] = kcal_avg

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predictions CSV
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dish_id", "kcal", "mass", "fat", "carb", "protein", "kcal_direct", "kcal_derived"])
        for i, did in enumerate(dish_ids):
            w.writerow([did, *preds_scalar_for_report[i].tolist(),
                        float(preds_kcal_direct[i]), float(preds_kcal_derived[i])])

    with open(out_dir / "groundtruth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dish_id", "kcal", "mass", "fat", "carb", "protein"])
        for i, did in enumerate(dish_ids):
            w.writerow([did, *targs_scalar_raw[i].tolist()])

    # Metrics
    metric_names = ["kcal", "mass", "fat", "carb", "protein"]
    results = {"n": len(dish_ids)}
    for j, name in enumerate(metric_names):
        p = preds_scalar_for_report[:, j]
        t = targs_scalar_raw[:, j]
        m = mae(p, t)
        pct = pct_mae(p, t)
        lo, hi = bootstrap_ci_mae(p, t)
        results[f"{name}_mae"] = m
        results[f"{name}_pct_mae"] = pct
        results[f"{name}_mae_ci95"] = [lo, hi]

    # Ingredient F1
    with np.errstate(over="ignore"):
        ingr_pred_bin = (1.0 / (1.0 + np.exp(-preds_ingr_logits)) > 0.5).astype(np.int32)
    micro, macro = multilabel_f1_micro_macro(ingr_pred_bin, targs_ingr_binary.astype(np.int32))
    results["ingr_f1_micro"] = micro
    results["ingr_f1_macro"] = macro
    rng_f1 = np.random.default_rng(0)
    boot_micro = np.empty(1000, dtype=np.float64)
    boot_macro = np.empty(1000, dtype=np.float64)
    for i in range(1000):
        idx = rng_f1.integers(0, len(dish_ids), len(dish_ids))
        m_i, M_i = multilabel_f1_micro_macro(ingr_pred_bin[idx], targs_ingr_binary.astype(np.int32)[idx])
        boot_micro[i] = m_i
        boot_macro[i] = M_i
    results["ingr_f1_micro_ci95"] = [float(np.percentile(boot_micro, 2.5)), float(np.percentile(boot_micro, 97.5))]
    results["ingr_f1_macro_ci95"] = [float(np.percentile(boot_macro, 2.5)), float(np.percentile(boot_macro, 97.5))]

    # Top-5 IoU
    iou = top_k_set_iou(preds_ingr_logits, targs_ingr_binary, k=5)
    results["top5_ingr_iou"] = float(iou.mean())
    # Bootstrap CI (per-dish IoU values)
    rng_iou = np.random.default_rng(0)
    boot_iou = np.empty(1000, dtype=np.float64)
    for i in range(1000):
        idx = rng_iou.integers(0, len(iou), len(iou))
        boot_iou[i] = iou[idx].mean()
    results["top5_ingr_iou_ci95"] = [float(np.percentile(boot_iou, 2.5)), float(np.percentile(boot_iou, 97.5))]

    # Per-ingredient mass MAE at GT-positive positions
    if targs_ingr_mask.sum() > 0:
        # per-dish masked mass MAE: sum(|err|*mask) / sum(mask) per dish (NaN if mask sum 0)
        diff = np.abs(preds_ingr_mass_raw - targs_ingr_mass_raw) * targs_ingr_mask
        mask_sums = targs_ingr_mask.sum(axis=1)
        per_dish = np.where(mask_sums > 0, diff.sum(axis=1) / np.maximum(mask_sums, 1), np.nan)
        valid = ~np.isnan(per_dish)
        results["per_ingredient_mass_mae"] = float(per_dish[valid].mean())
        rng_pm = np.random.default_rng(0)
        v = per_dish[valid]
        boot_pm = np.empty(1000, dtype=np.float64)
        for i in range(1000):
            idx = rng_pm.integers(0, len(v), len(v))
            boot_pm[i] = v[idx].mean()
        results["per_ingredient_mass_mae_ci95"] = [float(np.percentile(boot_pm, 2.5)), float(np.percentile(boot_pm, 97.5))]
    else:
        results["per_ingredient_mass_mae"] = float("nan")
        results["per_ingredient_mass_mae_ci95"] = [float("nan"), float("nan")]

    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--vocab", type=str, required=True)
    p.add_argument("--stats", type=str, required=True)
    p.add_argument("--split-file", type=str, required=True)
    p.add_argument("--available-dish-ids", type=str, default="data/sample/available_dish_ids.txt")
    p.add_argument("--imagery-root", type=str, default="data/sample/imagery")
    p.add_argument("--metadata-cafe1", type=str, default="data/raw/metadata/dish_metadata_cafe1.csv")
    p.add_argument("--metadata-cafe2", type=str, default="data/raw/metadata/dish_metadata_cafe2.csv")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-depth", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
