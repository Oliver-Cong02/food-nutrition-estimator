# tests/v2/test_dataset.py
from __future__ import annotations
import math
import numpy as np
import pytest
import torch
from src.v2.vocab import Vocab
from src.v2.stats import TrainStats
from src.v2.dataset import (
    parse_dish_metadata_row,
    DishLabels,
    Nutrition5kRGBD,
    build_default_train_transform,
    build_default_eval_transform,
)


def _dummy_stats():
    return TrainStats(
        scalar_mean=np.array([250., 200., 12., 25., 15.], dtype=np.float32),
        scalar_std=np.array([180., 130., 8., 18., 12.], dtype=np.float32),
        depth_mean=450.0, depth_std=80.0,
        mass_log1p_mean=2.0, mass_log1p_std=1.5,
    )


def test_parse_metadata_row(ingredients_csv):
    v = Vocab.from_csv(ingredients_csv)
    line = "dish_X,300.0,193.0,12.4,28.2,18.6,ingr_0000000508,soy sauce,3.4,1.8,0.02,0.17,0.28"
    row = parse_dish_metadata_row(line, v)
    assert row.dish_id == "dish_X"
    assert math.isclose(row.kcal, 300.0)
    assert math.isclose(row.mass, 193.0)
    assert len(row.ingr_ids) == 1
    assert row.ingr_ids[0] == "ingr_0000000508"
    assert math.isclose(row.ingr_grams[0], 3.4)


def test_parse_metadata_handles_multiple_ingredients(ingredients_csv):
    v = Vocab.from_csv(ingredients_csv)
    line = "dish_Y,100,50,1,2,3,ingr_0000000026,white rice,30,33,0.1,7,0.6,ingr_0000000508,soy sauce,5,2.6,0.03,0.25,0.4"
    row = parse_dish_metadata_row(line, v)
    assert len(row.ingr_ids) == 2
    assert row.ingr_grams == [30.0, 5.0]


def test_dataset_returns_correct_shapes(repo_root, ingredients_csv, dish_csv_cafe1):
    # We test against real metadata but only need a few dishes that have RGB locally.
    v = Vocab.from_csv(ingredients_csv)
    s = _dummy_stats()
    img_root = repo_root / "data" / "sample" / "imagery"
    if not img_root.is_dir():
        pytest.skip("no imagery available")

    # Pick first 2 dish IDs that have rgb.png
    dish_ids = [d.name for d in img_root.iterdir() if (d / "rgb.png").is_file()][:2]
    if len(dish_ids) < 2:
        pytest.skip("need at least 2 dishes with rgb.png")

    ds = Nutrition5kRGBD(
        dish_ids=dish_ids,
        metadata_csvs=[dish_csv_cafe1],
        imagery_root=img_root,
        vocab=v,
        stats=s,
        transform=build_default_eval_transform(),
        require_depth=False,    # sample test allows no depth
    )
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["rgb"].shape == (3, 224, 224)
    assert sample["depth"].shape == (2, 224, 224)
    assert sample["y_scalar"].shape == (5,)
    assert sample["y_ingr_binary"].shape == (v.size,)
    assert sample["y_ingr_mass"].shape == (v.size,)
    assert sample["y_ingr_mask"].shape == (v.size,)
    # Mass mask consistency
    assert int(sample["y_ingr_mask"].sum()) >= 1


def test_train_transform_produces_aligned_rgb_depth(repo_root, ingredients_csv, dish_csv_cafe1):
    """Train-mode transform must produce RGB and depth at the SAME 224x224 shape."""
    v = Vocab.from_csv(ingredients_csv)
    s = _dummy_stats()
    img_root = repo_root / "data" / "sample" / "imagery"
    if not img_root.is_dir():
        pytest.skip("no imagery available")
    dish_ids = [d.name for d in img_root.iterdir() if (d / "rgb.png").is_file()][:2]
    if len(dish_ids) < 1:
        pytest.skip("no dishes")
    ds = Nutrition5kRGBD(
        dish_ids=dish_ids, metadata_csvs=[dish_csv_cafe1], imagery_root=img_root,
        vocab=v, stats=s, transform=build_default_train_transform(),
        require_depth=False,
    )
    if len(ds) == 0:
        pytest.skip("no usable dishes after metadata filter")
    # Run __getitem__ a few times to exercise random crop + flip
    for _ in range(5):
        sample = ds[0]
        assert sample["rgb"].shape == (3, 224, 224), f"rgb {sample['rgb'].shape}"
        assert sample["depth"].shape == (2, 224, 224), f"depth {sample['depth'].shape}"


def test_label_construction_sums_to_total_mass(ingredients_csv, dish_csv_cafe1):
    """Per-ingredient grams should sum (within rounding) to total dish mass for at
    least 90% of rows. Some dishes have small upstream rounding mismatches in
    Nutrition5k metadata; allow a 10% violation rate."""
    v = Vocab.from_csv(ingredients_csv)
    rows = []
    with open(dish_csv_cafe1) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(parse_dish_metadata_row(ln, v))
            except Exception:
                pass
    n_checked, n_pass = 0, 0
    failures = []
    for r in rows[:50]:
        if not r.ingr_grams:
            continue
        s = sum(r.ingr_grams)
        ok = abs(s - r.mass) <= max(1.0, 0.02 * r.mass)
        if not ok:
            failures.append((r.dish_id, s, r.mass))
        n_pass += int(ok)
        n_checked += 1
    assert n_checked >= 10, "too few rows checked"
    pass_frac = n_pass / n_checked
    # 90% threshold chosen because dish_1550876012 (and a small handful of others)
    # have a known ~3-5% rounding mismatch in cafe1 metadata between Σ ingredient
    # grams and the recorded total dish mass. This is a Nutrition5k upstream issue,
    # not a parsing bug. If this threshold is ever breached, run the same check on
    # cafe2 and a fresh metadata CSV to determine if upstream changed.
    assert pass_frac >= 0.90, (
        f"only {n_pass}/{n_checked} dishes pass per-ingredient sum check "
        f"(< 90%); failures: {failures[:5]}"
    )


# ---------------------------------------------------------------------------
# Synthetic transform tests (no disk dependency)
# ---------------------------------------------------------------------------

import numpy as np
from PIL import Image
import torch
from src.v2.dataset import Nutrition5kTransform


def _synth_inputs(seed: int = 0):
    """Return (rgb_pil 480x640, depth_arr uint16 480x640, valid_mask 480x640)."""
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = rng.integers(200, 800, (480, 640), dtype=np.uint16)
    mask = (depth > 0).astype(np.bool_)
    return Image.fromarray(rgb), depth, mask


def test_transform_eval_produces_correct_shapes_synthetic():
    """Eval-mode (center-crop) yields aligned (3,224,224) and (2,224,224) on synthetic input."""
    rgb_pil, depth, mask = _synth_inputs()
    tf = Nutrition5kTransform(train=False)
    rgb_t, depth_t = tf(rgb_pil, depth, mask, depth_mean=400.0, depth_std=80.0)
    assert rgb_t.shape == (3, 224, 224)
    assert depth_t.shape == (2, 224, 224)


def test_transform_handles_zero_depth_std_without_nan():
    """Depth z-score must not produce NaN/Inf when depth_std=0 (degenerate stats)."""
    rgb_pil, depth, mask = _synth_inputs()
    tf = Nutrition5kTransform(train=False)
    rgb_t, depth_t = tf(rgb_pil, depth, mask, depth_mean=400.0, depth_std=0.0)
    assert torch.isfinite(rgb_t).all()
    assert torch.isfinite(depth_t).all()


def test_transform_train_hflip_sync():
    """When train HFlip fires, BOTH rgb and depth get flipped (single decision)."""
    # Force the flip path by seeding python's random module.
    import random
    rgb_pil, depth, mask = _synth_inputs()
    # Make depth distinguishable left↔right so flip is detectable
    depth = depth.copy()
    depth[:, :320] = 200       # left half
    depth[:, 320:] = 700       # right half
    valid = (depth > 0).astype(np.bool_)
    tf = Nutrition5kTransform(train=True, size=224, resize=256)

    flips_seen = []
    for seed in range(10):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        rgb_t, depth_t = tf(Image.fromarray(np.array(rgb_pil)), depth, valid,
                            depth_mean=400.0, depth_std=80.0)
        # Center column of depth tells us which half was on the left after possible flip
        # depth_t[0] is the z-scored depth; left half should be smaller (200<700) before flip
        left_mean = depth_t[0, :, :112].mean().item()
        right_mean = depth_t[0, :, 112:].mean().item()
        flips_seen.append(left_mean < right_mean)
    # In 10 seeds, we should see both flipped and non-flipped outcomes
    # (so this isn't trivially always-true / always-false)
    assert any(flips_seen) and not all(flips_seen), (
        f"flip behavior not random across seeds: {flips_seen}")


def test_transform_train_resize_alignment_synthetic():
    """Train-mode random crop applied to RGB and depth uses the SAME post-resize coordinate system."""
    rgb_pil, depth, mask = _synth_inputs()
    tf = Nutrition5kTransform(train=True)
    for _ in range(5):
        rgb_t, depth_t = tf(rgb_pil, depth, mask, depth_mean=400.0, depth_std=80.0)
        # If the resize-sync bug returned, depth would either silently truncate
        # or produce a shape !=224. Both rgb and depth must match.
        assert rgb_t.shape == (3, 224, 224)
        assert depth_t.shape == (2, 224, 224)
        # And the depth tensor values must be finite
        assert torch.isfinite(depth_t).all()
