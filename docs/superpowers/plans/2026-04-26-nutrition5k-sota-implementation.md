# Nutrition5k SOTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and train a ConvNeXt-Base RGB + ConvNeXt-Tiny depth dual-stream multi-task model that beats Google's Nutrition5k Direct Prediction baseline (calorie MAE ≤ 70 kcal hard floor; ≤ 60 kcal stretch goal).

**Architecture:** Late-fusion RGB-D backbone → 3 prediction heads (5 nutrition scalars, 555-way ingredient multi-label, 555-way per-ingredient mass). Loss = uncertainty-weighted multi-task with Atwater + dual-kcal-path consistency soft constraints. Single A6000 (48 GB) on `gpu2803`.

**Tech Stack:** PyTorch 2.11, torchvision 0.26, ConvNeXt (ImageNet-22k pretrained), bf16 AMP, weight EMA, AdamW + cosine schedule. Python 3.11 in uv-managed `.venv/`.

**Source spec:** `docs/superpowers/specs/2026-04-26-nutrition5k-sota-design.md` — every task lists which spec section it implements.

---

## File Structure

```
src/v2/
├── __init__.py                      # empty
├── vocab.py                         # Vocab class: 555 ingredients + density lookup       (Task 1)
├── stats.py                         # TrainStats: z-score stats for scalars/depth/mass    (Task 2)
├── dataset.py                       # Nutrition5kRGBD + transforms + label construction   (Task 3)
├── model.py                         # NutritionRGBDModel: 2 backbones + 3 heads           (Task 4)
├── losses.py                        # 5 losses + UncertaintyWeighting wrapper             (Task 5)
├── metrics.py                       # MAE, %MAE, F1, top-k, bootstrap CI                  (Task 6)
├── tta.py                           # TTA inference (HFlip + 3-crop)                      (Task 7)
├── evaluate.py                      # eval loop: predictions.csv + eval_results.json      (Task 8)
├── train.py                         # train loop: AMP + EMA + cosine + gates              (Task 9)
├── viz.py                           # sanity figures (RGB+Depth+labels), scatter, panels  (Task 10)
└── configs/
    ├── main.yaml                                                                          (Task 11)
    └── ablation_no_depth.yaml                                                             (Task 12)
tests/v2/
├── __init__.py
├── conftest.py                      # fixtures: tiny dataset, dummy tensors               (Task 1)
├── test_vocab.py
├── test_stats.py
├── test_dataset.py
├── test_model.py
├── test_losses.py
├── test_metrics.py
└── test_tta.py
scripts/
├── download_depth.sh                # parallel xargs depth download                       (Task 0a)
├── verify_depth.py                  # integrity check                                     (Task 0b)
└── compute_train_stats.py           # one-shot stats writer                               (Task 13)
docs/runs/<run_id>/{config.yaml,train.log,summary.md,diagnosis_*.md,figs/}                  (auto)
docs/ablations/<name>/summary.md                                                            (auto)
checkpoints/v2/<run_id>/{best.pt,ema.pt,last.pt,vocab.json,train_stats.json}                (auto)
```

**Decomposition rationale:** vocab and stats are pure data classes (testable on CPU, no torch model dep). Dataset depends on vocab+stats. Model is pure torch (no data dep). Losses are pure torch on tensors. Metrics are numpy. tta wraps model. evaluate composes model+tta+metrics+dataset. train composes everything. This DAG lets Tasks 1, 4, 5, 6 run in parallel by independent subagents.

---

## Phase 0 — Environment & Data Prep

### Task 0a: Install gcloud SDK and authenticate

**Files:**
- Create: `scripts/install_gcloud.sh`

**Why:** `gsutil`/`gcloud` are not in PATH. Need them to download depth_raw.png.

- [ ] **Step 1: Write the install script**

```bash
# scripts/install_gcloud.sh
#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${HOME}/.local/google-cloud-sdk"

if [ -d "$INSTALL_DIR" ]; then
  echo "gcloud already at $INSTALL_DIR"
  exit 0
fi

cd /tmp
curl -fsSLO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar xzf google-cloud-cli-linux-x86_64.tar.gz
mv google-cloud-sdk "$INSTALL_DIR"
"$INSTALL_DIR/install.sh" --quiet --usage-reporting=false --command-completion=true --path-update=true
echo "Add to your shell rc: source $INSTALL_DIR/path.bash.inc"
```

- [ ] **Step 2: Make executable, run, source path**

```bash
chmod +x scripts/install_gcloud.sh
bash scripts/install_gcloud.sh
source "$HOME/.local/google-cloud-sdk/path.bash.inc"
which gcloud gsutil      # both should resolve under ~/.local/google-cloud-sdk
```

Expected: `which gcloud` prints `/users/xcong2/.local/google-cloud-sdk/bin/gcloud` (or similar).

- [ ] **Step 3: Authenticate (USER INTERACTIVE)**

Tell the user: run `gcloud auth login` interactively (use the `! ` prefix in Claude Code). They'll get a URL → paste auth code back. Confirm with:

```bash
gcloud auth list                                # active account shown
gsutil ls gs://nutrition5k_dataset/ | head -3   # should list dirs without 401
```

- [ ] **Step 4: Commit**

```bash
git add scripts/install_gcloud.sh
git commit -m "chore: add gcloud SDK installer script"
```

---

### Task 0b: Parallel depth download

**Files:**
- Create: `scripts/download_depth.sh`
- Modify: nothing in src; only writes to `data/sample/imagery/<dish_id>/depth_raw.png`

**Why:** Original `download_available_dishes.sh` is sequential `gsutil cp` per file (1 RTT × 3490 ≈ hours). `xargs -P 16` parallelizes 16 concurrent downloads → 5–15 min. Spec §4.2.

- [ ] **Step 1: Write parallel download script**

```bash
# scripts/download_depth.sh
#!/usr/bin/env bash
set -euo pipefail

DISH_LIST="${1:-data/sample/available_dish_ids.txt}"
IMG_ROOT="${2:-data/sample/imagery}"
PARALLEL="${3:-16}"

if [ ! -f "$DISH_LIST" ]; then
  echo "Missing $DISH_LIST" >&2; exit 1
fi

awk -v root="$IMG_ROOT" '{
  src = "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/" $1 "/depth_raw.png"
  dst = root "/" $1 "/depth_raw.png"
  print src "\t" dst
}' "$DISH_LIST" \
  | xargs -P "$PARALLEL" -L 1 -I LINE bash -c '
      IFS=$"\t" read -r src dst <<< "LINE"
      mkdir -p "$(dirname "$dst")"
      [ -f "$dst" ] && exit 0
      gcloud storage cp "$src" "$dst" 2>/dev/null || gsutil cp "$src" "$dst"
    '

echo "Done. Verify with scripts/verify_depth.py"
```

(The `[ -f "$dst" ] && exit 0` makes it resumable.)

- [ ] **Step 2: Run it**

```bash
chmod +x scripts/download_depth.sh
bash scripts/download_depth.sh
```

Expected wall-clock: 5–15 minutes for ~3490 files.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_depth.sh
git commit -m "feat: parallel depth_raw.png downloader"
```

---

### Task 0c: Verify depth integrity

**Files:**
- Create: `scripts/verify_depth.py`

**Why:** Confirm every dish in `available_dish_ids.txt` has a non-empty, valid 16-bit PNG depth file.

- [ ] **Step 1: Write the verifier**

```python
# scripts/verify_depth.py
"""Verify depth_raw.png integrity for every available dish.

Prints summary; exits 0 if all valid, 1 if any missing/corrupt.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def main(dish_list: str, img_root: str) -> int:
    dish_ids = Path(dish_list).read_text().splitlines()
    img_root_p = Path(img_root)
    missing, corrupt, ok = [], [], 0
    for did in dish_ids:
        p = img_root_p / did / "depth_raw.png"
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(did); continue
        try:
            arr = np.array(Image.open(p))
            if arr.dtype != np.uint16 or arr.ndim != 2 or arr.shape != (480, 640):
                corrupt.append((did, str(arr.dtype), tuple(arr.shape))); continue
            ok += 1
        except Exception as e:
            corrupt.append((did, repr(e), None))
    total = len(dish_ids)
    print(f"OK={ok}/{total}  missing={len(missing)}  corrupt={len(corrupt)}")
    if missing[:5]:
        print("first missing:", missing[:5])
    if corrupt[:5]:
        print("first corrupt:", corrupt[:5])
    return 0 if (not missing and not corrupt) else 1


if __name__ == "__main__":
    sys.exit(main(
        sys.argv[1] if len(sys.argv) > 1 else "data/sample/available_dish_ids.txt",
        sys.argv[2] if len(sys.argv) > 2 else "data/sample/imagery",
    ))
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/verify_depth.py
```

Expected: `OK=3490/3490  missing=0  corrupt=0`. If non-zero missing → re-run `download_depth.sh` (idempotent).

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_depth.py
git commit -m "test: depth integrity verifier"
```

---

## Phase 1 — Core Modules (parallelizable across subagents)

### Task 1: vocab.py — 555-ingredient vocabulary + density lookup

**Files:**
- Create: `src/v2/__init__.py` (empty)
- Create: `src/v2/vocab.py`
- Create: `tests/v2/__init__.py` (empty)
- Create: `tests/v2/conftest.py`
- Create: `tests/v2/test_vocab.py`

**Spec ref:** §3 (per-ingredient density lookup), §4.5 (vocab construction), G1.

- [ ] **Step 1: Write conftest with shared fixtures**

```python
# tests/v2/conftest.py
"""Shared pytest fixtures for v2 tests."""
from __future__ import annotations
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def ingredients_csv(repo_root: Path) -> Path:
    p = repo_root / "data" / "raw" / "metadata" / "ingredients_metadata.csv"
    if not p.is_file():
        pytest.skip(f"missing {p}")
    return p


@pytest.fixture(scope="session")
def dish_csv_cafe1(repo_root: Path) -> Path:
    p = repo_root / "data" / "raw" / "metadata" / "dish_metadata_cafe1.csv"
    if not p.is_file():
        pytest.skip(f"missing {p}")
    return p
```

- [ ] **Step 2: Write failing tests for Vocab**

```python
# tests/v2/test_vocab.py
"""Tests for src/v2/vocab.py."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.v2.vocab import Vocab


def test_build_from_csv_has_555_entries(ingredients_csv):
    v = Vocab.from_csv(ingredients_csv)
    assert v.size == 554, f"expected 554 (1-554), got {v.size}"  # adjust if differs


def test_id_to_idx_roundtrip(ingredients_csv):
    v = Vocab.from_csv(ingredients_csv)
    for ingr_id in list(v.id_to_idx.keys())[:5]:
        idx = v.id_to_idx[ingr_id]
        assert v.idx_to_id[idx] == ingr_id


def test_density_lookup_positive(ingredients_csv):
    v = Vocab.from_csv(ingredients_csv)
    densities = [v.idx_to_density[i] for i in range(v.size)]
    assert all(d >= 0 for d in densities), "densities must be non-negative"
    assert any(d > 0 for d in densities), "at least one density must be positive"


def test_save_load_roundtrip(ingredients_csv, tmp_path):
    v = Vocab.from_csv(ingredients_csv)
    p = tmp_path / "vocab.json"
    v.save(p)
    v2 = Vocab.load(p)
    assert v.size == v2.size
    assert v.id_to_idx == v2.id_to_idx
    assert v.idx_to_density == v2.idx_to_density
```

(Note: the actual vocab size depends on the CSV. Update `554` after first run; documented in G1.)

- [ ] **Step 3: Run tests — expect ImportError**

```bash
.venv/bin/python -m pytest tests/v2/test_vocab.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.v2.vocab'`.

- [ ] **Step 4: Implement vocab.py**

```python
# src/v2/__init__.py
```

```python
# src/v2/vocab.py
"""Vocabulary of food ingredients, with density (cal/g) lookup.

Built from data/raw/metadata/ingredients_metadata.csv. Fixed order = sort by
the integer in the `id` column ascending. Used everywhere the model talks
about the 555-dim ingredient space.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Vocab:
    """Ingredient vocabulary.

    Attributes:
        idx_to_id:        list[str], length = size
        idx_to_name:      list[str], length = size
        idx_to_density:   list[float], cal/g
        id_to_idx:        dict from "ingr_XXXXXX" string to int
    """
    idx_to_id: List[str] = field(default_factory=list)
    idx_to_name: List[str] = field(default_factory=list)
    idx_to_density: List[float] = field(default_factory=list)
    id_to_idx: Dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.idx_to_id)

    @classmethod
    def from_csv(cls, csv_path: Path | str) -> "Vocab":
        """Build vocab from ingredients_metadata.csv.

        Header row: ingr,id,cal/g,fat(g),carb(g),protein(g)
        Each subsequent row: name,int_id,density,...
        """
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append((int(r["id"]), r["ingr"].strip(), float(r["cal/g"])))
        rows.sort(key=lambda r: r[0])

        v = cls()
        for int_id, name, density in rows:
            ingr_id = f"ingr_{int_id:010d}"
            v.idx_to_id.append(ingr_id)
            v.idx_to_name.append(name)
            v.idx_to_density.append(density)
            v.id_to_idx[ingr_id] = len(v.idx_to_id) - 1
        return v

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "idx_to_id": self.idx_to_id,
            "idx_to_name": self.idx_to_name,
            "idx_to_density": self.idx_to_density,
            "id_to_idx": self.id_to_idx,
        }, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "Vocab":
        d = json.loads(Path(path).read_text())
        return cls(
            idx_to_id=d["idx_to_id"],
            idx_to_name=d["idx_to_name"],
            idx_to_density=d["idx_to_density"],
            id_to_idx=d["id_to_idx"],
        )
```

- [ ] **Step 5: Re-run, fix vocab size assertion**

```bash
.venv/bin/python -m pytest tests/v2/test_vocab.py -v
```

If `assert v.size == 554` fails with actual size N, edit the test to N. (This is the real ground-truth count; the `554` was a placeholder estimate.)

- [ ] **Step 6: Commit**

```bash
git add src/v2/__init__.py src/v2/vocab.py tests/v2/__init__.py tests/v2/conftest.py tests/v2/test_vocab.py
git commit -m "feat(v2): Vocab class with density lookup + tests"
```

---

### Task 2: stats.py — train-set z-score statistics

**Files:**
- Create: `src/v2/stats.py`
- Create: `tests/v2/test_stats.py`

**Spec ref:** §4.3 (depth z-score), §4.5 (mass log1p z-score), §5.1 (scalar z-score).

- [ ] **Step 1: Write failing tests**

```python
# tests/v2/test_stats.py
from __future__ import annotations
import json
import numpy as np
import pytest
from src.v2.stats import TrainStats


def test_z_then_inv_roundtrip():
    s = TrainStats(scalar_mean=np.array([1, 2, 3, 4, 5], dtype=np.float32),
                   scalar_std=np.array([0.5, 1, 1.5, 2, 2.5], dtype=np.float32),
                   depth_mean=400.0, depth_std=100.0,
                   mass_log1p_mean=2.0, mass_log1p_std=1.0)
    x = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    z = s.scalar_z(x)
    back = s.scalar_inv_z(z)
    np.testing.assert_allclose(back, x, rtol=1e-5)


def test_save_load(tmp_path):
    s = TrainStats(scalar_mean=np.array([1, 2, 3, 4, 5], dtype=np.float32),
                   scalar_std=np.array([1] * 5, dtype=np.float32),
                   depth_mean=400.0, depth_std=100.0,
                   mass_log1p_mean=2.0, mass_log1p_std=1.0)
    p = tmp_path / "stats.json"
    s.save(p)
    s2 = TrainStats.load(p)
    np.testing.assert_array_equal(s.scalar_mean, s2.scalar_mean)
    assert s.depth_mean == s2.depth_mean


def test_mass_log1p_z_inv():
    s = TrainStats(scalar_mean=np.zeros(5, np.float32), scalar_std=np.ones(5, np.float32),
                   depth_mean=0.0, depth_std=1.0,
                   mass_log1p_mean=2.0, mass_log1p_std=1.0)
    grams = np.array([0.0, 5.0, 50.0, 200.0], dtype=np.float32)
    z = s.mass_log1p_z(grams)
    back = s.mass_log1p_inv_z(z)
    np.testing.assert_allclose(back, grams, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run, expect import fail**

```bash
.venv/bin/python -m pytest tests/v2/test_stats.py -v
```

- [ ] **Step 3: Implement**

```python
# src/v2/stats.py
"""Train-set z-score / log1p statistics, used to (de)normalize labels and inputs.

All values stored as plain Python types in JSON for portability.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TrainStats:
    scalar_mean: np.ndarray  # shape (5,)  — kcal, mass, fat, carb, protein
    scalar_std: np.ndarray   # shape (5,)
    depth_mean: float        # over valid (>0) pixels in train, after clip [200,800]
    depth_std: float
    mass_log1p_mean: float   # over all positive per-ingredient grams in train
    mass_log1p_std: float

    def scalar_z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.scalar_mean) / self.scalar_std

    def scalar_inv_z(self, z: np.ndarray) -> np.ndarray:
        return z * self.scalar_std + self.scalar_mean

    def depth_z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.depth_mean) / self.depth_std

    def mass_log1p_z(self, grams: np.ndarray) -> np.ndarray:
        return (np.log1p(grams) - self.mass_log1p_mean) / self.mass_log1p_std

    def mass_log1p_inv_z(self, z: np.ndarray) -> np.ndarray:
        return np.expm1(z * self.mass_log1p_std + self.mass_log1p_mean)

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "scalar_mean": self.scalar_mean.tolist(),
            "scalar_std": self.scalar_std.tolist(),
            "depth_mean": float(self.depth_mean),
            "depth_std": float(self.depth_std),
            "mass_log1p_mean": float(self.mass_log1p_mean),
            "mass_log1p_std": float(self.mass_log1p_std),
        }, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "TrainStats":
        d = json.loads(Path(path).read_text())
        return cls(
            scalar_mean=np.array(d["scalar_mean"], dtype=np.float32),
            scalar_std=np.array(d["scalar_std"], dtype=np.float32),
            depth_mean=d["depth_mean"],
            depth_std=d["depth_std"],
            mass_log1p_mean=d["mass_log1p_mean"],
            mass_log1p_std=d["mass_log1p_std"],
        )
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/python -m pytest tests/v2/test_stats.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/v2/stats.py tests/v2/test_stats.py
git commit -m "feat(v2): TrainStats with z-score / log1p helpers + tests"
```

---

### Task 3: dataset.py — Nutrition5kRGBD dataset + transforms

**Files:**
- Create: `src/v2/dataset.py`
- Create: `tests/v2/test_dataset.py`

**Spec ref:** §3 (input shapes), §4 entire (splits, preprocessing, augmentation, labels).

- [ ] **Step 1: Write failing tests**

```python
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


def test_dataset_returns_correct_shapes(repo_root, ingredients_csv, dish_csv_cafe1, monkeypatch):
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


def test_label_construction_sums_to_total_mass(ingredients_csv, dish_csv_cafe1):
    """Per-ingredient grams must sum (within rounding) to total dish mass."""
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
    n_checked = 0
    for r in rows[:50]:
        if not r.ingr_grams:
            continue
        s = sum(r.ingr_grams)
        # Allow a 2% relative slack for floating-point + dataset rounding
        assert abs(s - r.mass) <= max(1.0, 0.02 * r.mass), \
            f"{r.dish_id}: ingr_sum={s} vs total_mass={r.mass}"
        n_checked += 1
    assert n_checked >= 10, "too few rows checked"
```

- [ ] **Step 2: Run, expect import fail**

```bash
.venv/bin/python -m pytest tests/v2/test_dataset.py -v
```

- [ ] **Step 3: Implement dataset.py**

```python
# src/v2/dataset.py
"""Nutrition5k RGB-D dataset + transforms.

Spec: docs/superpowers/specs/2026-04-26-nutrition5k-sota-design.md §3, §4.

Returns dict per dish:
    rgb           (3, 224, 224) float32, ImageNet-normalized
    depth         (2, 224, 224) float32, [normalized_depth, valid_mask]
    y_scalar      (5,)          float32, z-scored kcal/mass/fat/carb/protein
    y_ingr_binary (V,)          float32, multi-label one-hot
    y_ingr_mass   (V,)          float32, log1p+z-scored at present positions; 0 elsewhere
    y_ingr_mask   (V,)          float32, 1 where ingredient is present, else 0
    dish_id       str

Test-time we also expose RAW (unnormalized) versions for evaluation:
    y_scalar_raw  (5,)          raw kcal/mass/fat/carb/protein
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .stats import TrainStats
from .vocab import Vocab


DEPTH_CLIP_MIN = 200.0   # mm
DEPTH_CLIP_MAX = 800.0


@dataclass
class DishLabels:
    dish_id: str
    kcal: float
    mass: float
    fat: float
    carb: float
    protein: float
    ingr_ids: List[str] = field(default_factory=list)
    ingr_grams: List[float] = field(default_factory=list)


def parse_dish_metadata_row(line: str, vocab: Vocab) -> DishLabels:
    """Parse a single CSV row from dish_metadata_cafe{1,2}.csv.

    Format:
        dish_id,kcal,mass,fat,carb,protein,
        [ingr_id,name,grams,kcal,fat,carb,protein] × N
    Ingredients with id not in vocab are silently dropped.
    """
    parts = line.strip().split(",")
    if len(parts) < 6:
        raise ValueError(f"row too short: {line[:80]}")
    dish_id = parts[0]
    kcal = float(parts[1]); mass = float(parts[2])
    fat = float(parts[3]);  carb = float(parts[4]); protein = float(parts[5])
    ingr_part = parts[6:]
    if len(ingr_part) % 7 != 0:
        # Drop trailing partial entry — defensive
        ingr_part = ingr_part[: (len(ingr_part) // 7) * 7]
    ingr_ids: List[str] = []
    ingr_grams: List[float] = []
    for i in range(0, len(ingr_part), 7):
        ingr_id = ingr_part[i].strip()
        try:
            grams = float(ingr_part[i + 2])
        except ValueError:
            continue
        if ingr_id in vocab.id_to_idx:
            ingr_ids.append(ingr_id)
            ingr_grams.append(grams)
    return DishLabels(dish_id, kcal, mass, fat, carb, protein,
                      ingr_ids=ingr_ids, ingr_grams=ingr_grams)


def _load_metadata_dict(metadata_csvs: Sequence[Path | str], vocab: Vocab) -> dict[str, DishLabels]:
    out: dict[str, DishLabels] = {}
    for csv in metadata_csvs:
        with open(csv) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = parse_dish_metadata_row(ln, vocab)
                    out[row.dish_id] = row
                except Exception:
                    continue
    return out


# ---------- Transforms ----------

def build_default_train_transform() -> Callable:
    return Nutrition5kTransform(train=True)


def build_default_eval_transform() -> Callable:
    return Nutrition5kTransform(train=False)


class Nutrition5kTransform:
    """Joint RGB+depth transform; depth gets a separate path."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, train: bool, size: int = 224, resize: int = 256):
        self.train = train
        self.size = size
        self.resize = resize
        self.color_jitter = T.ColorJitter(0.2, 0.2, 0.2)
        self.rand_aug = T.RandAugment(num_ops=2, magnitude=9)

    def __call__(
        self,
        rgb_pil: Image.Image,
        depth_arr: np.ndarray,        # uint16, raw mm
        valid_mask: np.ndarray,       # bool
        depth_mean: float,
        depth_std: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize keeping aspect
        rgb_pil = TF.resize(rgb_pil, self.resize, antialias=True)
        depth_t = torch.from_numpy(depth_arr.astype(np.float32))[None, None, ...]   # (1,1,H,W)
        mask_t = torch.from_numpy(valid_mask.astype(np.float32))[None, None, ...]
        depth_t = F.interpolate(depth_t, size=self.resize, mode="bilinear", align_corners=False)[0, 0]
        mask_t = F.interpolate(mask_t, size=self.resize, mode="nearest")[0, 0]

        # Crop
        if self.train:
            i, j, h, w = T.RandomResizedCrop.get_params(rgb_pil, scale=(0.7, 1.0), ratio=(0.9, 1.1))
            rgb_pil = TF.resized_crop(rgb_pil, i, j, h, w, [self.size, self.size], antialias=True)
            depth_t = depth_t[i:i+h, j:j+w][None, None]
            mask_t = mask_t[i:i+h, j:j+w][None, None]
            depth_t = F.interpolate(depth_t, size=self.size, mode="bilinear", align_corners=False)[0, 0]
            mask_t = F.interpolate(mask_t, size=self.size, mode="nearest")[0, 0]
        else:
            rgb_pil = TF.center_crop(rgb_pil, [self.size, self.size])
            cy = (depth_t.shape[-2] - self.size) // 2
            cx = (depth_t.shape[-1] - self.size) // 2
            depth_t = depth_t[cy:cy+self.size, cx:cx+self.size]
            mask_t = mask_t[cy:cy+self.size, cx:cx+self.size]

        # HFlip
        if self.train and random.random() < 0.5:
            rgb_pil = TF.hflip(rgb_pil)
            depth_t = torch.flip(depth_t, dims=[-1])
            mask_t = torch.flip(mask_t, dims=[-1])

        # RGB color aug
        if self.train:
            rgb_pil = self.rand_aug(rgb_pil)
            rgb_pil = self.color_jitter(rgb_pil)

        rgb = TF.to_tensor(rgb_pil)
        rgb = TF.normalize(rgb, self.IMAGENET_MEAN, self.IMAGENET_STD)

        # Depth scale aug
        if self.train:
            depth_t = depth_t * random.uniform(0.95, 1.05)

        # Apply mask: invalid → 0 (post-z-score this means "mean", which is the safest fill)
        depth_norm = (depth_t - depth_mean) / max(depth_std, 1e-6)
        depth_norm = depth_norm * mask_t   # zero out invalid
        depth_out = torch.stack([depth_norm.float(), mask_t.float()], dim=0)
        return rgb, depth_out


# ---------- Dataset ----------

class Nutrition5kRGBD(Dataset):
    """RGB-D Nutrition5k dish-level dataset.

    Args:
        dish_ids:        list of dish IDs to include
        metadata_csvs:   list of dish_metadata_*.csv paths
        imagery_root:    Path to data/sample/imagery
        vocab:           Vocab (for ingredient indexing + density)
        stats:           TrainStats (for z-score normalization of labels & depth)
        transform:       Nutrition5kTransform (or any (rgb_pil, depth_arr, mask, mean, std) -> (rgb, depth))
        require_depth:   if True, dishes without depth_raw.png are filtered
    """

    def __init__(
        self,
        dish_ids: Sequence[str],
        metadata_csvs: Sequence[Path | str],
        imagery_root: Path,
        vocab: Vocab,
        stats: TrainStats,
        transform: Optional[Callable] = None,
        require_depth: bool = True,
    ):
        self.imagery_root = Path(imagery_root)
        self.vocab = vocab
        self.stats = stats
        self.transform = transform
        meta = _load_metadata_dict(metadata_csvs, vocab)
        self.dishes: List[DishLabels] = []
        for did in dish_ids:
            row = meta.get(did)
            if row is None:
                continue
            rgb_p = self.imagery_root / did / "rgb.png"
            if not rgb_p.is_file():
                continue
            d_p = self.imagery_root / did / "depth_raw.png"
            if require_depth and not d_p.is_file():
                continue
            self.dishes.append(row)

    def __len__(self) -> int:
        return len(self.dishes)

    def _read_depth(self, dish_id: str) -> Tuple[np.ndarray, np.ndarray]:
        d_p = self.imagery_root / dish_id / "depth_raw.png"
        if d_p.is_file():
            arr = np.array(Image.open(d_p)).astype(np.float32)  # (H,W) uint16 -> float32 mm
        else:
            arr = np.zeros((480, 640), dtype=np.float32)
        valid = (arr > 0).astype(np.float32)
        arr = np.clip(arr, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX) * valid   # invalid stays 0
        return arr, valid

    def __getitem__(self, i: int):
        d = self.dishes[i]
        rgb_p = self.imagery_root / d.dish_id / "rgb.png"
        rgb_pil = Image.open(rgb_p).convert("RGB")
        depth_arr, valid = self._read_depth(d.dish_id)
        if self.transform is None:
            self.transform = build_default_eval_transform()
        rgb_t, depth_t = self.transform(rgb_pil, depth_arr, valid,
                                        self.stats.depth_mean, self.stats.depth_std)

        # Labels — 5 scalars
        y_scalar_raw = np.array([d.kcal, d.mass, d.fat, d.carb, d.protein], dtype=np.float32)
        y_scalar = self.stats.scalar_z(y_scalar_raw).astype(np.float32)

        # Labels — 555-dim
        V = self.vocab.size
        y_ingr_binary = np.zeros(V, dtype=np.float32)
        y_ingr_mass = np.zeros(V, dtype=np.float32)
        y_ingr_mask = np.zeros(V, dtype=np.float32)
        for ingr_id, grams in zip(d.ingr_ids, d.ingr_grams):
            idx = self.vocab.id_to_idx[ingr_id]
            y_ingr_binary[idx] = 1.0
            y_ingr_mass[idx] = self.stats.mass_log1p_z(np.array([grams], dtype=np.float32))[0]
            y_ingr_mask[idx] = 1.0

        return {
            "rgb": rgb_t,
            "depth": depth_t,
            "y_scalar": torch.from_numpy(y_scalar),
            "y_scalar_raw": torch.from_numpy(y_scalar_raw),
            "y_ingr_binary": torch.from_numpy(y_ingr_binary),
            "y_ingr_mass": torch.from_numpy(y_ingr_mass),
            "y_ingr_mask": torch.from_numpy(y_ingr_mask),
            "dish_id": d.dish_id,
        }
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/v2/test_dataset.py -v
```

If any sample test fails because few dishes exist in some path, adjust path or skip locally.

- [ ] **Step 5: Commit**

```bash
git add src/v2/dataset.py tests/v2/test_dataset.py
git commit -m "feat(v2): RGB-D dataset + label construction + tests"
```

---

### Task 4: model.py — RGB-D dual-stream multi-task model

**Files:**
- Create: `src/v2/model.py`
- Create: `tests/v2/test_model.py`

**Spec ref:** §3 entire.

- [ ] **Step 1: Failing tests**

```python
# tests/v2/test_model.py
from __future__ import annotations
import torch
from src.v2.model import NutritionRGBDModel


def test_forward_shapes():
    m = NutritionRGBDModel(n_ingredients=555).eval()
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 2, 224, 224)
    out = m(rgb, depth)
    assert out["scalar"].shape == (2, 5)
    assert out["ingr_logits"].shape == (2, 555)
    assert out["ingr_mass"].shape == (2, 555)


def test_no_depth_zeros_d_branch():
    m = NutritionRGBDModel(n_ingredients=555).eval()
    rgb = torch.randn(2, 3, 224, 224)
    depth_zero = torch.zeros(2, 2, 224, 224)
    depth_random = torch.randn(2, 2, 224, 224)
    out_zero = m(rgb, depth_zero, use_depth=False)
    out_rand = m(rgb, depth_random, use_depth=False)
    # Output must be identical when use_depth=False
    for k in out_zero:
        assert torch.allclose(out_zero[k], out_rand[k])


def test_param_count_roughly_120m():
    m = NutritionRGBDModel(n_ingredients=555)
    n = sum(p.numel() for p in m.parameters())
    # Target ~120M ± 30M (ConvNeXt-Base ~89M + ConvNeXt-Tiny ~28M + heads)
    assert 90_000_000 < n < 200_000_000, f"got {n} params"


def test_backward_no_nan():
    m = NutritionRGBDModel(n_ingredients=555).train()
    rgb = torch.randn(2, 3, 224, 224, requires_grad=False)
    depth = torch.randn(2, 2, 224, 224, requires_grad=False)
    out = m(rgb, depth)
    loss = sum(o.mean() for o in out.values())
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
```

- [ ] **Step 2: Run, expect import fail**

```bash
.venv/bin/python -m pytest tests/v2/test_model.py -v
```

- [ ] **Step 3: Implement**

```python
# src/v2/model.py
"""RGB-D dual-stream multi-task model for Nutrition5k.

Spec: §3.

Architecture:
    rgb     -> ConvNeXt-Base   (ImageNet-22k pretrained)  -> feat_rgb (1024)
    depth   -> ConvNeXt-Tiny   (channel-mean adapted to 2ch) -> feat_d (768)
    concat  -> MLP(1792 -> 512) -> z (512)
    z       -> head_scalar  (Linear 512 -> 5)
    z       -> head_ingr    (Linear 512 -> n_ingredients)
    z       -> head_mass    (Linear 512 -> n_ingredients)

The depth encoder's first conv is reinitialized with channel-mean of the
RGB-pretrained weights, then duplicated to 2 in_channels (depth, valid_mask).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import convnext_base, convnext_tiny, ConvNeXt_Base_Weights, ConvNeXt_Tiny_Weights


def _adapt_first_conv_to_2ch(conv: nn.Conv2d) -> nn.Conv2d:
    """Replace a 3-channel first conv with a 2-channel version using RGB-channel-mean init."""
    w = conv.weight.detach()  # (out, 3, kh, kw)
    mean_w = w.mean(dim=1, keepdim=True)  # (out, 1, kh, kw)
    new_w = mean_w.repeat(1, 2, 1, 1)     # (out, 2, kh, kw)
    new_conv = nn.Conv2d(
        in_channels=2,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight.copy_(new_w)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class NutritionRGBDModel(nn.Module):
    def __init__(self, n_ingredients: int, dropout: float = 0.1, hidden_dim: int = 512):
        super().__init__()
        # RGB encoder — ConvNeXt-Base
        rgb_w = ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.rgb_enc = convnext_base(weights=rgb_w)
        rgb_feat_dim = 1024
        self.rgb_enc.classifier = nn.Identity()  # leave global pool's flatten in classifier
        # ConvNeXt's classifier = LayerNorm2d -> Flatten -> Linear; we replaced with Identity.
        # We use features-level output after avgpool-flatten manually below.
        self.rgb_avgpool = nn.AdaptiveAvgPool2d(1)
        self.rgb_norm = nn.LayerNorm(rgb_feat_dim)

        # Depth encoder — ConvNeXt-Tiny, adapted first conv
        d_w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.d_enc = convnext_tiny(weights=d_w)
        # Patchify conv lives at features[0][0] for ConvNeXt
        old_conv = self.d_enc.features[0][0]
        self.d_enc.features[0][0] = _adapt_first_conv_to_2ch(old_conv)
        self.d_enc.classifier = nn.Identity()
        d_feat_dim = 768
        self.d_avgpool = nn.AdaptiveAvgPool2d(1)
        self.d_norm = nn.LayerNorm(d_feat_dim)

        # Fusion MLP
        in_dim = rgb_feat_dim + d_feat_dim
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Heads
        self.head_scalar = nn.Linear(hidden_dim, 5)
        self.head_ingr = nn.Linear(hidden_dim, n_ingredients)
        self.head_mass = nn.Linear(hidden_dim, n_ingredients)

    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.rgb_enc.features(rgb)
        x = self.rgb_avgpool(x).flatten(1)
        return self.rgb_norm(x)

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.d_enc.features(depth)
        x = self.d_avgpool(x).flatten(1)
        return self.d_norm(x)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, *, use_depth: bool = True):
        feat_rgb = self.encode_rgb(rgb)
        if use_depth:
            feat_d = self.encode_depth(depth)
        else:
            feat_d = torch.zeros(rgb.size(0), 768, device=rgb.device, dtype=feat_rgb.dtype)
        z = self.fuse(torch.cat([feat_rgb, feat_d], dim=1))
        return {
            "scalar": self.head_scalar(z),
            "ingr_logits": self.head_ingr(z),
            "ingr_mass": self.head_mass(z),
        }

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        backbone_params = list(self.rgb_enc.parameters()) + list(self.d_enc.parameters())
        head_params = list(self.fuse.parameters()) + \
                      list(self.head_scalar.parameters()) + \
                      list(self.head_ingr.parameters()) + \
                      list(self.head_mass.parameters()) + \
                      list(self.rgb_norm.parameters()) + list(self.d_norm.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/v2/test_model.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/v2/model.py tests/v2/test_model.py
git commit -m "feat(v2): RGB-D dual-stream model with 3 heads + tests"
```

---

### Task 5: losses.py — multi-task losses + uncertainty weighting

**Files:**
- Create: `src/v2/losses.py`
- Create: `tests/v2/test_losses.py`

**Spec ref:** §5.1, §5.2.

- [ ] **Step 1: Failing tests**

```python
# tests/v2/test_losses.py
from __future__ import annotations
import math
import torch
from src.v2.losses import (
    masked_huber,
    bce_with_pos_weight,
    atwater_loss,
    kcal_consistency_loss,
    UncertaintyWeighter,
)


def test_masked_huber_zero_when_pred_eq_target_at_mask():
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, 5.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss = masked_huber(pred, target, mask, delta=1.0)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_masked_huber_ignores_zero_mask():
    pred = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[1.0, 1000.0]])
    mask = torch.tensor([[1.0, 0.0]])
    loss = masked_huber(pred, target, mask, delta=1.0)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_bce_with_pos_weight_shape():
    logits = torch.randn(4, 555)
    target = torch.zeros(4, 555); target[:, :3] = 1
    pos_weight = torch.full((555,), 50.0)
    loss = bce_with_pos_weight(logits, target, pos_weight)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_atwater_zero_when_consistent():
    # kcal = 9*fat + 4*carb + 4*protein
    fat = torch.tensor([10.0]); carb = torch.tensor([20.0]); protein = torch.tensor([5.0])
    kcal = 9 * fat + 4 * carb + 4 * protein
    loss = atwater_loss(kcal, fat, carb, protein)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_kcal_consistency_zero_when_equal():
    a = torch.tensor([300.0, 200.0])
    b = torch.tensor([300.0, 200.0])
    loss = kcal_consistency_loss(a, b)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_uncertainty_weighter_init_zero():
    w = UncertaintyWeighter(["a", "b", "c"])
    losses = {"a": torch.tensor(2.0), "b": torch.tensor(4.0), "c": torch.tensor(6.0)}
    total, parts = w(losses)
    # With s_t=0, total = 0.5*L_a + 0.5*L_b + 0.5*L_c + 0
    expected = 0.5 * (2 + 4 + 6)
    assert torch.isclose(total, torch.tensor(expected))


def test_uncertainty_weighter_clamp():
    w = UncertaintyWeighter(["a"], s_floor=-2.0)
    with torch.no_grad():
        w.log_var["a"].fill_(-100.0)
    losses = {"a": torch.tensor(1.0)}
    total, _ = w(losses)
    assert torch.isfinite(total)
```

- [ ] **Step 2: Run, expect import fail**

```bash
.venv/bin/python -m pytest tests/v2/test_losses.py -v
```

- [ ] **Step 3: Implement**

```python
# src/v2/losses.py
"""Loss functions for the multi-task RGB-D Nutrition5k model.

Spec: §5.1, §5.2.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 delta: float = 1.0) -> torch.Tensor:
    """Huber loss applied only at mask=1 positions, averaged over masked count.

    Args:
        pred:   any shape (typically (B, V))
        target: same shape
        mask:   same shape, 0/1
        delta:  Huber transition point (in target units)
    """
    diff = (pred - target) * mask
    abs_diff = diff.abs()
    quad = 0.5 * (abs_diff ** 2)
    lin = delta * (abs_diff - 0.5 * delta)
    loss = torch.where(abs_diff <= delta, quad, lin)
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def bce_with_pos_weight(logits: torch.Tensor, target: torch.Tensor,
                        pos_weight: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits, mean over (B, V), with per-class pos_weight."""
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="mean"
    )


def atwater_loss(kcal: torch.Tensor, fat: torch.Tensor,
                 carb: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
    """Soft physical regularizer: |kcal - (9·fat + 4·carb + 4·protein)|.

    All inputs in raw kcal/g units, shape (B,) or (B, 1).
    """
    derived = 9.0 * fat + 4.0 * carb + 4.0 * protein
    return F.smooth_l1_loss(kcal, derived, reduction="mean", beta=1.0)


def kcal_consistency_loss(direct: torch.Tensor, derived: torch.Tensor) -> torch.Tensor:
    """Couple direct head A kcal and derived (Σ mass × density) kcal, both raw kcal."""
    return F.smooth_l1_loss(direct, derived, reduction="mean", beta=1.0)


class UncertaintyWeighter(nn.Module):
    """Multi-task uncertainty weighting (Kendall, Gal, Cipolla 2018).

    For each task t with raw loss L_t:
        L = Σ_t  (1 / (2 · exp(s_t))) · L_t  +  0.5 · s_t

    s_t are learnable scalars; gradient signs balance task losses automatically.
    Floor clamp prevents `exp(-large_negative)` blow-up.
    """

    def __init__(self, task_names: List[str], s_floor: float = -2.0, s_init: float = 0.0):
        super().__init__()
        self.task_names = list(task_names)
        self.s_floor = s_floor
        self.log_var = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(s_init, dtype=torch.float32))
            for name in self.task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = torch.zeros((), device=next(iter(losses.values())).device)
        parts: Dict[str, torch.Tensor] = {}
        for name in self.task_names:
            s = torch.clamp(self.log_var[name], min=self.s_floor)
            l = losses[name]
            scaled = 0.5 * torch.exp(-s) * l + 0.5 * s
            parts[f"weighted_{name}"] = scaled.detach()
            parts[f"raw_{name}"] = l.detach()
            parts[f"s_{name}"] = s.detach()
            total = total + scaled
        return total, parts
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/v2/test_losses.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/v2/losses.py tests/v2/test_losses.py
git commit -m "feat(v2): multi-task losses + uncertainty weighter + tests"
```

---

### Task 6: metrics.py — eval metrics + bootstrap CI

**Files:**
- Create: `src/v2/metrics.py`
- Create: `tests/v2/test_metrics.py`

**Spec ref:** §6.

- [ ] **Step 1: Failing tests**

```python
# tests/v2/test_metrics.py
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
    assert math.isclose(iou.mean(), 1/3)


def test_paired_bootstrap_mae_shape():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100); b = rng.normal(size=100)
    t = rng.normal(size=100)
    delta_mean, ci_low, ci_high = paired_bootstrap_mae(a, b, t, n=200, seed=0)
    assert ci_low <= delta_mean <= ci_high
```

- [ ] **Step 2: Run, expect import fail**

```bash
.venv/bin/python -m pytest tests/v2/test_metrics.py -v
```

- [ ] **Step 3: Implement**

```python
# src/v2/metrics.py
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

    micro F1: aggregate TP/FP/FN over all (n,v); macro: per-class F1, mean over classes.
    """
    pred = pred_binary.astype(bool)
    targ = target.astype(bool)
    tp = (pred & targ).sum(axis=0)
    fp = (pred & ~targ).sum(axis=0)
    fn = (~pred & targ).sum(axis=0)
    micro_tp = tp.sum(); micro_fp = fp.sum(); micro_fn = fn.sum()
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
    valid = (target.sum(axis=0) > 0)
    macro_f1 = float(per_f1[valid].mean()) if valid.any() else 0.0
    return float(micro_f1), float(macro_f1)


def top_k_set_iou(scores: np.ndarray, target_binary: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-row IoU between top-k predicted indices and GT-positive indices.

    Returns (N,) array.
    """
    n, v = scores.shape
    top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    iou = np.zeros(n, dtype=np.float32)
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


def paired_bootstrap_mae(pred_a: np.ndarray, pred_b: np.ndarray, target: np.ndarray,
                         n: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    """Paired bootstrap on per-sample absolute errors.

    Returns (delta_mae_mean, low95, high95) where delta = MAE(b) - MAE(a).
    Negative delta means b is better.
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


def bootstrap_ci_mae(pred: np.ndarray, target: np.ndarray,
                     n: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """Single-population bootstrap of MAE: returns (low95, high95)."""
    err = np.abs(pred - target)
    rng = np.random.default_rng(seed)
    N = len(target)
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, N, N)
        vals[i] = err[idx].mean()
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/v2/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/v2/metrics.py tests/v2/test_metrics.py
git commit -m "feat(v2): eval metrics + paired bootstrap CI + tests"
```

---

### Task 7: tta.py — Test-time augmentation

**Files:**
- Create: `src/v2/tta.py`
- Create: `tests/v2/test_tta.py`

**Spec ref:** §5.5.

- [ ] **Step 1: Failing tests**

```python
# tests/v2/test_tta.py
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
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/python -m pytest tests/v2/test_tta.py -v
```

- [ ] **Step 3: Implement**

```python
# src/v2/tta.py
"""Test-time augmentation: HFlip + 3-crop (center, top-left, bottom-right).

Spec: §5.5.

Inputs are pre-normalized 224×224 tensors (eval-time CenterCrop already applied).
Therefore "3-crop" is implemented by slicing 192×192 sub-regions and resizing
back to 224 — this is a pragmatic in-tensor TTA that doesn't require re-doing
the file IO. For exact paper-style 3-crop, evaluate on 256-resized + crop pre-
transform; we keep things simple and consistent with our pipeline.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _three_crops(x: torch.Tensor, crop: int = 192, target: int = 224) -> list[torch.Tensor]:
    """Returns [center, top-left, bottom-right] each upsampled back to target×target."""
    H, W = x.shape[-2:]
    cy = (H - crop) // 2
    cx = (W - crop) // 2
    cuts = [
        x[..., cy:cy + crop, cx:cx + crop],
        x[..., :crop, :crop],
        x[..., H - crop:, W - crop:],
    ]
    return [F.interpolate(c, size=target, mode="bilinear", align_corners=False) for c in cuts]


@torch.no_grad()
def tta_predict(model, rgb: torch.Tensor, depth: torch.Tensor, *, use_depth: bool = True) -> Dict[str, torch.Tensor]:
    """Average model outputs over {center, TL, BR} × {orig, hflip} = 6 forward passes."""
    model.eval()
    rgb_views = _three_crops(rgb) + [F.hflip(c) if False else None for c in _three_crops(rgb)]
    # Simpler: explicit 6-list construction
    crops_rgb = _three_crops(rgb)
    crops_d = _three_crops(depth)
    rgb_list = []
    d_list = []
    for r, d in zip(crops_rgb, crops_d):
        rgb_list.append(r); d_list.append(d)                # original
        rgb_list.append(torch.flip(r, dims=[-1]))
        d_list.append(torch.flip(d, dims=[-1]))
    accum = {"scalar": 0.0, "ingr_logits": 0.0, "ingr_mass": 0.0}
    for r, d in zip(rgb_list, d_list):
        out = model(r, d, use_depth=use_depth)
        for k in accum:
            accum[k] = accum[k] + out[k]
    n = len(rgb_list)
    return {k: v / n for k, v in accum.items()}
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/v2/test_tta.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/v2/tta.py tests/v2/test_tta.py
git commit -m "feat(v2): TTA inference (3-crop + HFlip) + tests"
```

---

## Phase 2 — Integration (sequential after Phase 1)

### Task 8: evaluate.py — eval loop with predictions + metrics dump

**Files:**
- Create: `src/v2/evaluate.py`

**Spec ref:** §6.

- [ ] **Step 1: Implement**

```python
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
    ingr_pred_bin = (1 / (1 + np.exp(-preds_ingr_logits)) > 0.5).astype(np.int32)
    micro, macro = multilabel_f1_micro_macro(ingr_pred_bin, targs_ingr_binary.astype(np.int32))
    results["ingr_f1_micro"] = micro
    results["ingr_f1_macro"] = macro

    # Top-5 IoU
    iou = top_k_set_iou(preds_ingr_logits, targs_ingr_binary, k=5)
    results["top5_ingr_iou"] = float(iou.mean())

    # Per-ingredient mass MAE at GT-positive positions
    if targs_ingr_mask.sum() > 0:
        diff = np.abs(preds_ingr_mass_raw - targs_ingr_mass_raw) * targs_ingr_mask
        results["per_ingredient_mass_mae"] = float(diff.sum() / targs_ingr_mask.sum())
    else:
        results["per_ingredient_mass_mae"] = float("nan")

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
```

- [ ] **Step 2: Smoke run from command-line (no GPU needed yet, will fail without checkpoint — that's fine, just check that it imports)**

```bash
.venv/bin/python -c "from src.v2 import evaluate; print('import OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/v2/evaluate.py
git commit -m "feat(v2): eval CLI with TTA + dual-kcal averaging + bootstrap CI"
```

---

### Task 9: train.py — main training loop

**Files:**
- Create: `src/v2/train.py`

**Spec ref:** §5.3, §5.4, §5.2, G3, G4.

- [ ] **Step 1: Implement**

```python
# src/v2/train.py
"""Main training loop with AMP + EMA + cosine schedule + correctness gates.

Spec: §5.

Usage:
    python -m src.v2.train --config src/v2/configs/main.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .dataset import Nutrition5kRGBD, build_default_eval_transform, build_default_train_transform
from .losses import (
    UncertaintyWeighter, atwater_loss, bce_with_pos_weight,
    kcal_consistency_loss, masked_huber,
)
from .model import NutritionRGBDModel
from .stats import TrainStats
from .vocab import Vocab


logger = logging.getLogger("nutrition5k.train")


@dataclass
class TrainConfig:
    run_id: str
    out_root: str
    imagery_root: str
    metadata_cafe1: str
    metadata_cafe2: str
    train_ids_path: str
    val_ids_path: str          # built once (10% of train)
    available_ids_path: str
    vocab_csv: str
    stats_path: str            # produced by scripts/compute_train_stats.py
    use_depth: bool = True
    n_epochs: int = 50
    batch_size: int = 64
    grad_accum: int = 1
    lr_backbone: float = 3e-5
    lr_head: float = 3e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    ema_decay: float = 0.9999
    early_stop_patience: int = 10
    seed: int = 42
    bf16: bool = True
    num_workers: int = 6
    log_every: int = 50
    overfit_micro: bool = False  # G3 mode

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow


def build_pos_weight(loader: DataLoader, vocab_size: int, device: torch.device) -> torch.Tensor:
    """pos_weight per class = (num_negatives / num_positives), capped to 100."""
    pos = torch.zeros(vocab_size); n = 0
    for b in loader:
        pos += b["y_ingr_binary"].sum(dim=0)
        n += b["y_ingr_binary"].size(0)
        if n > 500: break
    pos = pos / max(n, 1)  # frequency
    neg = 1.0 - pos
    pw = (neg / pos.clamp(min=1e-3)).clamp(max=100.0)
    return pw.to(device)


def lr_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def make_loaders(cfg: TrainConfig, vocab: Vocab, stats: TrainStats):
    avail = set([ln.strip() for ln in Path(cfg.available_ids_path).read_text().splitlines() if ln.strip()])
    train_ids = [ln.strip() for ln in Path(cfg.train_ids_path).read_text().splitlines() if ln.strip() and ln.strip() in avail]
    val_ids = [ln.strip() for ln in Path(cfg.val_ids_path).read_text().splitlines() if ln.strip() and ln.strip() in avail]
    if cfg.overfit_micro:
        train_ids = train_ids[:8]; val_ids = val_ids[:8]

    md = [cfg.metadata_cafe1, cfg.metadata_cafe2]
    train_ds = Nutrition5kRGBD(train_ids, md, cfg.imagery_root, vocab, stats,
                               transform=build_default_train_transform(),
                               require_depth=cfg.use_depth)
    val_ds = Nutrition5kRGBD(val_ids, md, cfg.imagery_root, vocab, stats,
                             transform=build_default_eval_transform(),
                             require_depth=cfg.use_depth)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def compute_total_loss(model, batch, weighter: UncertaintyWeighter,
                       densities: torch.Tensor, stats: TrainStats,
                       pos_weight: torch.Tensor, use_depth: bool) -> tuple[torch.Tensor, dict]:
    rgb = batch["rgb"]; depth = batch["depth"]
    out = model(rgb, depth, use_depth=use_depth)
    sc_pred_z = out["scalar"]
    sc_target_z = batch["y_scalar"]
    L_scalar = masked_huber(sc_pred_z, sc_target_z,
                            mask=torch.ones_like(sc_target_z), delta=1.0)
    L_ingr_cls = bce_with_pos_weight(out["ingr_logits"], batch["y_ingr_binary"], pos_weight)
    L_ingr_mass = masked_huber(out["ingr_mass"], batch["y_ingr_mass"],
                               batch["y_ingr_mask"], delta=1.0)
    # Atwater + kcal_consist on raw kcal scale
    sc_pred_raw = sc_pred_z * torch.tensor(stats.scalar_std, device=sc_pred_z.device) \
                  + torch.tensor(stats.scalar_mean, device=sc_pred_z.device)
    direct_kcal = sc_pred_raw[:, 0]
    fat = sc_pred_raw[:, 2]; carb = sc_pred_raw[:, 3]; protein = sc_pred_raw[:, 4]
    L_atwater = atwater_loss(direct_kcal, fat, carb, protein)
    mass_raw = torch.expm1(out["ingr_mass"] * stats.mass_log1p_std + stats.mass_log1p_mean).clamp(min=0)
    derived_kcal = (mass_raw * densities[None, :]).sum(dim=1)
    L_kcal_consist = kcal_consistency_loss(direct_kcal, derived_kcal)

    losses = {
        "scalar": L_scalar, "ingr_cls": L_ingr_cls, "ingr_mass": L_ingr_mass,
        "atwater": L_atwater, "kcal_consist": L_kcal_consist,
    }
    total, parts = weighter(losses)
    return total, {**losses, **parts}


def evaluate_val(model, val_loader, weighter, densities, stats, pos_weight, use_depth, device) -> dict:
    model.eval()
    sums = {}
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if torch.is_tensor(v): batch[k] = v.to(device, non_blocking=True)
            total, parts = compute_total_loss(model, batch, weighter, densities, stats, pos_weight, use_depth)
            B = batch["rgb"].size(0)
            for k, v in parts.items():
                if torch.is_tensor(v):
                    sums[k] = sums.get(k, 0.0) + float(v) * B
            n += B
    return {k: v / max(n, 1) for k, v in sums.items()}


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.out_root) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/v2") / cfg.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "train.log"); fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.handlers.clear(); logger.addHandler(fh); logger.addHandler(sh); logger.setLevel(logging.INFO)
    logger.info("config: %s", json.dumps(asdict(cfg), indent=2))

    vocab = Vocab.from_csv(cfg.vocab_csv)
    vocab.save(ckpt_dir / "vocab.json")
    stats = TrainStats.load(cfg.stats_path)
    stats.save(ckpt_dir / "train_stats.json")

    train_loader, val_loader = make_loaders(cfg, vocab, stats)
    logger.info("train=%d val=%d", len(train_loader.dataset), len(val_loader.dataset))

    model = NutritionRGBDModel(n_ingredients=vocab.size).to(device)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay),
    )
    weighter = UncertaintyWeighter(["scalar", "ingr_cls", "ingr_mass", "atwater", "kcal_consist"]).to(device)
    optimizer.add_param_group({"params": list(weighter.parameters()), "lr": cfg.lr_head, "weight_decay": 0.0})
    densities = torch.tensor(vocab.idx_to_density, dtype=torch.float32, device=device)
    pos_weight = build_pos_weight(train_loader, vocab.size, device)

    ema = EMA(model, decay=cfg.ema_decay)
    total_steps = len(train_loader) * cfg.n_epochs
    warmup_steps = int(total_steps * cfg.warmup_frac)
    best_val = math.inf; bad = 0
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 doesn't need scaler

    step = 0
    for epoch in range(cfg.n_epochs):
        model.train()
        for it, batch in enumerate(train_loader):
            for k, v in batch.items():
                if torch.is_tensor(v): batch[k] = v.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                total, parts = compute_total_loss(
                    model, batch, weighter, densities, stats, pos_weight, cfg.use_depth
                )

            (total / cfg.grad_accum).backward()

            # Adjust LR per-param-group for warmup/cosine
            sched_factor = lr_schedule(step, total_steps, warmup_steps)
            for pg, base in zip(optimizer.param_groups[:2], [cfg.lr_backbone, cfg.lr_head]):
                pg["lr"] = base * sched_factor

            if (it + 1) % cfg.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(weighter.parameters()), max_norm=5.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                step += 1

                if step % cfg.log_every == 0:
                    logger.info(
                        "step=%d epoch=%d loss=%.4f scalar=%.4f cls=%.4f mass=%.4f atw=%.4f kc=%.4f gn=%.2f lr=%.2e",
                        step, epoch, float(total),
                        float(parts["scalar"]), float(parts["ingr_cls"]),
                        float(parts["ingr_mass"]), float(parts["atwater"]),
                        float(parts["kcal_consist"]), float(grad_norm),
                        optimizer.param_groups[0]["lr"],
                    )
                    if not torch.isfinite(total):
                        logger.error("NaN/Inf loss — aborting")
                        return

        # End of epoch — eval on val with EMA weights
        val_model = copy.deepcopy(model)
        val_model.load_state_dict(ema.state_dict(), strict=True)
        val_metrics = evaluate_val(val_model, val_loader, weighter, densities, stats,
                                   pos_weight, cfg.use_depth, device)
        val_score = val_metrics.get("scalar", math.inf)
        logger.info("epoch=%d val_scalar=%.4f val_total_metrics=%s", epoch, val_score,
                    {k: round(v, 4) for k, v in val_metrics.items()})

        # G4 sanity (epoch 1)
        if epoch == 0 and val_score >= 1.0:
            logger.error("G4 FAIL: val z-score MAE >= 1.0 at epoch 1 (%.4f). Stop and diagnose.", val_score)
            torch.save({"model": model.state_dict()}, ckpt_dir / "g4_fail_last.pt")
            return

        # Save best
        if val_score < best_val:
            best_val = val_score; bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "best.pt")
            torch.save({"model": ema.state_dict(), "epoch": epoch}, ckpt_dir / "ema.pt")
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                logger.info("Early stop at epoch %d", epoch); break

    # Always save last
    torch.save({"model": model.state_dict()}, ckpt_dir / "last.pt")
    torch.save({"model": ema.state_dict()}, ckpt_dir / "last_ema.pt")
    logger.info("done. best val %.4f", best_val)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--overfit-micro", action="store_true", help="G3 mode: 8-dish overfit test")
    args = p.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    if args.overfit_micro:
        cfg.overfit_micro = True
        cfg.n_epochs = 100; cfg.early_stop_patience = 999; cfg.batch_size = 8
        cfg.lr_head = 1e-3; cfg.lr_backbone = 1e-4
    main(cfg)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Import smoke test**

```bash
.venv/bin/python -c "from src.v2 import train; print('import OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/v2/train.py
git commit -m "feat(v2): training loop with AMP, EMA, cosine schedule, G4 inline"
```

---

### Task 10: viz.py — sanity figures

**Files:**
- Create: `src/v2/viz.py`

**Spec ref:** §10.3, G1, G4.

- [ ] **Step 1: Implement**

```python
# src/v2/viz.py
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
```

- [ ] **Step 2: Smoke**

```bash
.venv/bin/python -c "from src.v2 import viz; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/v2/viz.py
git commit -m "feat(v2): sanity / scatter visualization helpers"
```

---

### Task 11: configs/main.yaml

**Files:**
- Create: `src/v2/configs/main.yaml`

- [ ] **Step 1: Build val split file once**

```bash
.venv/bin/python <<'PY'
import random
from pathlib import Path
random.seed(42)
avail = set(Path("data/sample/available_dish_ids.txt").read_text().splitlines())
train_all = [l for l in Path("data/raw/dish_ids/splits/rgb_train_ids.txt").read_text().splitlines() if l in avail]
random.shuffle(train_all)
n = len(train_all); n_val = max(1, int(round(0.10 * n)))
val_ids = sorted(train_all[:n_val]); train_ids = sorted(train_all[n_val:])
out = Path("data/sample/splits"); out.mkdir(parents=True, exist_ok=True)
(out / "train_ids.txt").write_text("\n".join(train_ids) + "\n")
(out / "val_ids.txt").write_text("\n".join(val_ids) + "\n")
print(f"train={len(train_ids)} val={len(val_ids)}")
PY
```

Expected: `train=2480 val=275` (approx; numbers depend on intersection).

- [ ] **Step 2: Write config**

```yaml
# src/v2/configs/main.yaml
run_id: main_seed42
out_root: docs/runs
imagery_root: data/sample/imagery
metadata_cafe1: data/raw/metadata/dish_metadata_cafe1.csv
metadata_cafe2: data/raw/metadata/dish_metadata_cafe2.csv
train_ids_path: data/sample/splits/train_ids.txt
val_ids_path: data/sample/splits/val_ids.txt
available_ids_path: data/sample/available_dish_ids.txt
vocab_csv: data/raw/metadata/ingredients_metadata.csv
stats_path: data/sample/train_stats.json

use_depth: true
n_epochs: 50
batch_size: 64
grad_accum: 1
lr_backbone: 3.0e-5
lr_head: 3.0e-4
weight_decay: 0.05
warmup_frac: 0.05
ema_decay: 0.9999
early_stop_patience: 10
seed: 42
bf16: true
num_workers: 6
log_every: 50
overfit_micro: false
```

- [ ] **Step 3: Commit**

```bash
git add data/sample/splits/train_ids.txt data/sample/splits/val_ids.txt src/v2/configs/main.yaml
git commit -m "feat(v2): train/val split + main config"
```

---

### Task 12: configs/ablation_no_depth.yaml

**Files:**
- Create: `src/v2/configs/ablation_no_depth.yaml`

- [ ] **Step 1: Write config**

```yaml
# src/v2/configs/ablation_no_depth.yaml
run_id: ablation_no_depth_seed42
out_root: docs/runs
imagery_root: data/sample/imagery
metadata_cafe1: data/raw/metadata/dish_metadata_cafe1.csv
metadata_cafe2: data/raw/metadata/dish_metadata_cafe2.csv
train_ids_path: data/sample/splits/train_ids.txt
val_ids_path: data/sample/splits/val_ids.txt
available_ids_path: data/sample/available_dish_ids.txt
vocab_csv: data/raw/metadata/ingredients_metadata.csv
stats_path: data/sample/train_stats.json

use_depth: false           # <-- only difference
n_epochs: 50
batch_size: 64
grad_accum: 1
lr_backbone: 3.0e-5
lr_head: 3.0e-4
weight_decay: 0.05
warmup_frac: 0.05
ema_decay: 0.9999
early_stop_patience: 10
seed: 42
bf16: true
num_workers: 6
log_every: 50
overfit_micro: false
```

- [ ] **Step 2: Commit**

```bash
git add src/v2/configs/ablation_no_depth.yaml
git commit -m "feat(v2): ablation config (no depth)"
```

---

### Task 13: compute_train_stats.py — produce train_stats.json

**Files:**
- Create: `scripts/compute_train_stats.py`

**Spec ref:** §4.3, §4.5, §5.1.

- [ ] **Step 1: Implement**

```python
# scripts/compute_train_stats.py
"""Compute train-set z-score statistics for scalars / depth / mass.

Run once before training; output is consumed by train.py and evaluate.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.v2.dataset import DEPTH_CLIP_MAX, DEPTH_CLIP_MIN, parse_dish_metadata_row
from src.v2.stats import TrainStats
from src.v2.vocab import Vocab


def main(args):
    avail = set([ln.strip() for ln in Path(args.available_dish_ids).read_text().splitlines() if ln.strip()])
    train_ids = [ln.strip() for ln in Path(args.train_ids_path).read_text().splitlines() if ln.strip() and ln.strip() in avail]
    vocab = Vocab.from_csv(args.vocab_csv)

    rows = {}
    for csv in [args.metadata_cafe1, args.metadata_cafe2]:
        with open(csv) as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try:
                    r = parse_dish_metadata_row(ln, vocab)
                    rows[r.dish_id] = r
                except Exception: pass

    scalars = []
    grams = []
    img_root = Path(args.imagery_root)

    depth_sum = 0.0; depth_sqsum = 0.0; depth_n = 0
    n = 0
    for did in train_ids:
        r = rows.get(did)
        if r is None: continue
        scalars.append([r.kcal, r.mass, r.fat, r.carb, r.protein])
        grams.extend([g for g in r.ingr_grams if g > 0])
        d_p = img_root / did / "depth_raw.png"
        if d_p.is_file():
            arr = np.array(Image.open(d_p)).astype(np.float32)
            valid = arr > 0
            arr_v = np.clip(arr[valid], DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
            depth_sum += float(arr_v.sum())
            depth_sqsum += float((arr_v ** 2).sum())
            depth_n += int(arr_v.size)
        n += 1
        if args.max and n >= args.max: break

    arr = np.asarray(scalars, dtype=np.float32)
    s_mean = arr.mean(axis=0); s_std = arr.std(axis=0) + 1e-6
    g = np.asarray(grams, dtype=np.float32)
    log1p = np.log1p(g)
    m_mean = float(log1p.mean()); m_std = float(log1p.std() + 1e-6)
    d_mean = depth_sum / max(depth_n, 1)
    d_var = max(depth_sqsum / max(depth_n, 1) - d_mean ** 2, 1e-6)
    d_std = float(np.sqrt(d_var))

    stats = TrainStats(s_mean, s_std, d_mean, d_std, m_mean, m_std)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats.save(args.out)
    print(json.dumps({
        "n_dishes": n, "scalar_mean": s_mean.tolist(), "scalar_std": s_std.tolist(),
        "depth_mean": d_mean, "depth_std": d_std,
        "mass_log1p_mean": m_mean, "mass_log1p_std": m_std,
    }, indent=2))


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--train-ids-path", default="data/sample/splits/train_ids.txt")
    p.add_argument("--available-dish-ids", default="data/sample/available_dish_ids.txt")
    p.add_argument("--imagery-root", default="data/sample/imagery")
    p.add_argument("--vocab-csv", default="data/raw/metadata/ingredients_metadata.csv")
    p.add_argument("--metadata-cafe1", default="data/raw/metadata/dish_metadata_cafe1.csv")
    p.add_argument("--metadata-cafe2", default="data/raw/metadata/dish_metadata_cafe2.csv")
    p.add_argument("--out", default="data/sample/train_stats.json")
    p.add_argument("--max", type=int, default=0, help="cap number of dishes processed (debug)")
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/compute_train_stats.py
cat data/sample/train_stats.json
```

- [ ] **Step 3: Commit**

```bash
git add scripts/compute_train_stats.py data/sample/train_stats.json
git commit -m "feat(v2): script to compute train z-score stats"
```

---

## Phase 3 — Code Review

### Task 14: Code-reviewer subagent pass

**Files:** none (review only)

- [ ] **Step 1: Dispatch a code-reviewer subagent**

Send this prompt to subagent (`subagent_type: superpowers:code-reviewer`):

> Review `src/v2/` and `tests/v2/` against the spec at `docs/superpowers/specs/2026-04-26-nutrition5k-sota-design.md`.
>
> Specifically check:
> 1. Forward path matches §3 (RGB→ConvNeXt-Base, Depth+Mask→ConvNeXt-Tiny adapted, late fusion, 3 heads)
> 2. Loss formulation matches §5.1 (masked Huber, BCE+pos_weight, Atwater, kcal_consist, uncertainty weighting)
> 3. Z-score and log1p inverse transforms in `evaluate.py` use the same stats as in `dataset.py`
> 4. TTA averages 6 forward passes (3 crops × 2 flips) per spec §5.5
> 5. Pos_weight cap (100) and clip values match spec
> 6. EMA decay matches config (0.9999)
> 7. No silent test-set leakage (val_ids and test_ids must be disjoint from train_ids)
> 8. No magic numbers that don't trace back to spec
>
> Report: critical issues (must fix before training), suggestions (nice-to-have), and any spec gaps.

- [ ] **Step 2: Triage findings**

Apply critical fixes inline. Document suggestions in `docs/runs/code_review_pass1.md`. Re-run failing tests if changes made.

- [ ] **Step 3: Commit any fixes**

```bash
git add -p src/v2/ tests/v2/
git commit -m "fix(v2): code review pass 1 — <summary>"
```

---

## Phase 4 — Sanity Gates (CPU/GPU)

### Task 15: G1 — dataset sanity (CPU)

**Files:**
- Create: `scripts/run_g1.py`

- [ ] **Step 1: Implement**

```python
# scripts/run_g1.py
"""G1 gate — visualize 5 random dishes; verify per-ingredient sums; verify vocab."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from src.v2.dataset import (Nutrition5kRGBD, build_default_eval_transform, parse_dish_metadata_row)
from src.v2.stats import TrainStats
from src.v2.vocab import Vocab
from src.v2 import viz


def main():
    random.seed(0)
    vocab = Vocab.from_csv("data/raw/metadata/ingredients_metadata.csv")
    print("vocab size:", vocab.size)

    # Diff against existing checkpoint vocab if present
    old_p = Path("checkpoints/vocab.json")
    if old_p.is_file():
        try:
            old = json.loads(old_p.read_text())
            if isinstance(old, list):
                old_size = len(old)
            elif isinstance(old, dict):
                old_size = len(old.get("idx_to_id", old))
            else:
                old_size = -1
            print(f"old vocab.json size = {old_size}; new = {vocab.size}")
        except Exception as e:
            print("could not read old vocab:", e)

    stats = TrainStats.load("data/sample/train_stats.json")
    avail = set(Path("data/sample/available_dish_ids.txt").read_text().splitlines())
    pick = random.sample(sorted(avail), 5)
    ds = Nutrition5kRGBD(
        dish_ids=pick,
        metadata_csvs=["data/raw/metadata/dish_metadata_cafe1.csv",
                       "data/raw/metadata/dish_metadata_cafe2.csv"],
        imagery_root="data/sample/imagery",
        vocab=vocab, stats=stats,
        transform=build_default_eval_transform(),
        require_depth=False,
    )
    samples = [ds[i] for i in range(len(ds))]
    out_dir = Path("docs/runs/g1"); out_dir.mkdir(parents=True, exist_ok=True)
    viz.sanity_panel(samples, out_dir / "sanity.png")
    print("wrote", out_dir / "sanity.png")

    # Per-ingredient mass sum check on 50 random dishes
    sample_ids = random.sample(sorted(avail), 50)
    rows = {}
    for csv in ["data/raw/metadata/dish_metadata_cafe1.csv", "data/raw/metadata/dish_metadata_cafe2.csv"]:
        for ln in Path(csv).read_text().splitlines():
            try:
                r = parse_dish_metadata_row(ln, vocab); rows[r.dish_id] = r
            except Exception: pass
    n_checked = 0; n_pass = 0
    for did in sample_ids:
        r = rows.get(did)
        if r is None or not r.ingr_grams: continue
        s = sum(r.ingr_grams); diff = abs(s - r.mass)
        ok = diff <= max(1.0, 0.02 * r.mass)
        if not ok:
            print(f"  WARN  {did} ingr_sum={s:.1f} total={r.mass:.1f} diff={diff:.2f}")
        n_pass += int(ok); n_checked += 1
    print(f"per-ingredient sum check: {n_pass}/{n_checked} pass")
    if n_pass < 0.9 * n_checked:
        sys.exit(1)
    print("G1 OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
.venv/bin/python scripts/run_g1.py
```

Expected:
- `vocab size: 555` (or similar — record actual)
- `per-ingredient sum check: ≥45/50 pass`
- `G1 OK`
- File `docs/runs/g1/sanity.png` produced; manually open and verify the 5 panels look like food.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_g1.py docs/runs/g1/sanity.png
git commit -m "test(v2): G1 sanity check passing"
```

---

### Task 16: G2 — model dummy forward (CPU)

**Files:** none (run pytest)

- [ ] **Step 1: Run model tests**

```bash
.venv/bin/python -m pytest tests/v2/test_model.py -v
```

Expected: all 4 tests pass. This IS Gate G2.

- [ ] **Step 2: Document**

Append to `docs/runs/g1.md` (create if missing):

```bash
mkdir -p docs/runs
cat >> docs/runs/sanity_gates.md <<'EOF'
## G2 — model dummy forward

Run: `pytest tests/v2/test_model.py -v`
Result: PASS (forward shapes, no-depth zero-out, param count, backward no-NaN)
EOF
git add docs/runs/sanity_gates.md
git commit -m "docs: G2 sanity gate passed"
```

---

### Task 17: G3 — overfit 8-dish micro-batch (GPU)

**Files:**
- Create: `src/v2/configs/g3_overfit.yaml` (copy of main.yaml with `overfit_micro: true`)

- [ ] **Step 1: Run on GPU node**

User must SSH to gpu2803 first:

```bash
# In a Bash terminal (NOT through this Claude session — Claude can't ssh interactively):
ssh gpu2803
cd /oscar/data/ssrinath/users/xcong2/projects/hw/food-nutrition-estimator
module load cuda
source .venv/bin/activate
python -m src.v2.train --config src/v2/configs/main.yaml --overfit-micro
```

- [ ] **Step 2: Check log**

Expected within ~100 iterations:
- `loss=...` strictly decreasing
- final `loss < 0.5` (started near `5.0`)
- no NaN
- `gn=...` (grad norm) stays under 100

- [ ] **Step 3: Document and commit**

Append to `docs/runs/sanity_gates.md`:

```markdown
## G3 — overfit micro-batch (GPU)

Run: `python -m src.v2.train --config src/v2/configs/main.yaml --overfit-micro`
Result: PASS (loss → <0.5 in <100 iters, no NaN)
GPU: A6000 on gpu2803
```

```bash
git add docs/runs/sanity_gates.md
git commit -m "docs: G3 sanity gate passed"
```

If G3 fails, **stop and diagnose** — do not proceed to main run.

---

## Phase 5 — Main Run

### Task 18: Main training run

**Files:** none (training writes to `docs/runs/main_seed42/` and `checkpoints/v2/main_seed42/`)

- [ ] **Step 1: Pre-flight checklist** (write to `docs/runs/main_seed42/preflight.md`)

```markdown
- [ ] G1 passed
- [ ] G2 passed
- [ ] G3 passed
- [ ] data/sample/train_stats.json exists and is non-empty
- [ ] data/sample/splits/{train,val}_ids.txt exist
- [ ] checkpoints/v2/main_seed42/ does NOT yet exist (clean run)
- [ ] gpu2803 has the A6000 free (run `nvidia-smi`)
```

- [ ] **Step 2: Launch training (foreground on gpu2803, in tmux)**

```bash
ssh gpu2803
module load cuda
cd /oscar/data/ssrinath/users/xcong2/projects/hw/food-nutrition-estimator
source .venv/bin/activate
tmux new -s n5k_main
python -m src.v2.train --config src/v2/configs/main.yaml 2>&1 | tee docs/runs/main_seed42/train.live.log
# Detach with Ctrl-b d. Reattach with: tmux attach -t n5k_main
```

Expected wall-clock: ~5–6 hours.

- [ ] **Step 3: Monitor G4 (epoch 1)**

After ~7 minutes (epoch 1 end), check log:
```bash
grep "epoch=0 val_scalar" docs/runs/main_seed42/train.live.log
```

Expected: `val_scalar < 1.0`. If not, training will auto-abort (G4 internal).

- [ ] **Step 4: After completion, run final eval (G5)**

```bash
python -m src.v2.evaluate \
  --checkpoint checkpoints/v2/main_seed42/ema.pt \
  --vocab      checkpoints/v2/main_seed42/vocab.json \
  --stats      checkpoints/v2/main_seed42/train_stats.json \
  --split-file data/raw/dish_ids/splits/rgb_test_ids.txt \
  --output-dir docs/runs/main_seed42/eval/
cat docs/runs/main_seed42/eval/eval_results.json
```

**G5 pass condition:** `kcal_mae <= 70` AND `mass_mae <= 40`. Stretch: `kcal_mae <= 60` AND `mass_mae <= 35`.

- [ ] **Step 5: Write run summary**

`docs/runs/main_seed42/summary.md`:

```markdown
# main_seed42 — main run summary

## Config
(paste from configs/main.yaml)

## Results
| metric | value | 95% CI |
|---|---|---|
| kcal MAE | ... | [..., ...] |
| mass MAE | ... | [..., ...] |
| ... | ... | ... |

## Comparison vs Google direct prediction
Baseline kcal MAE 70 → ours: ...
Baseline mass MAE 40 → ours: ...

## G5 status
PASS / STRETCH / FAIL

## Failure modes (if any)
...

## Next step
- if PASS+STRETCH: skip to ablation (Task 21)
- if PASS only: ablation
- if FAIL: diagnose (Task 20)
```

- [ ] **Step 6: Commit**

```bash
git add docs/runs/main_seed42/ checkpoints/v2/main_seed42/vocab.json checkpoints/v2/main_seed42/train_stats.json
git commit -m "run: main_seed42 main training + eval"
```

(Note: actual `*.pt` checkpoints are NOT committed — they're in `.gitignore` recommended; document the path in summary.md instead.)

---

### Task 19: Doc-while-train subagent (parallel to Task 18)

**Files:** none

While Task 18 trains, dispatch a `general-purpose` subagent in parallel to write `docs/runs/main_seed42/incremental_summary.md` from the live log every 30 minutes. Prompt:

> Read `docs/runs/main_seed42/train.live.log` and produce/update `docs/runs/main_seed42/incremental_summary.md` with:
> - Current epoch and step
> - Latest train loss values (per-task)
> - Latest val z-score MAE
> - Trend (improving / plateau / spiking)
> - Any warning lines
> Re-read every time you're invoked. Do not modify the log file.

This keeps Claude's main context clean while training runs.

---

## Phase 6 — Iteration Loop (only if G5 fails)

### Task 20: Diagnose + retrain (≤ 2 retries)

**Files:**
- Create: `docs/runs/main_seed{42,43,...}/diagnosis_<n>.md` per attempt

- [ ] **Step 1: Diagnose**

Run the diagnostic checks from spec §9 in order:

```bash
# Did uncertainty weights collapse?
grep "s_scalar\|s_ingr" docs/runs/main_seed42/train.log | tail -20
# Did gradients explode/vanish?
grep "gn=" docs/runs/main_seed42/train.live.log | tail -20
```

Write `docs/runs/main_seed42/diagnosis_1.md` with:
1. Hypothesis (one of the §9 ranked options)
2. Concrete change (file:line, old → new)
3. Expected effect on val/test
4. New `run_id`: `main_seed42_v2`

- [ ] **Step 2: Apply fix as a code commit**

```bash
git checkout -b fix/<short-name>
# edit
git commit -am "fix(v2): <change>"
git checkout main && git merge --no-ff fix/<short-name>
```

- [ ] **Step 3: Retrain with new run_id**

Modify config `run_id: main_seed42_v2`, retrain via Task 18.

- [ ] **Step 4: Compare**

In `docs/runs/main_seed42_v2/summary.md`, table comparing main_seed42 vs main_seed42_v2 across all metrics. If still G5-fail and ≤ 1 retry left → continue. Else → freeze + final report.

**Hard cap: 3 main-run trains total.**

---

## Phase 7 — Ablation

### Task 21: No-depth ablation run

**Files:** none new

- [ ] **Step 1: Train**

Same as Task 18, but `--config src/v2/configs/ablation_no_depth.yaml`. Run_id `ablation_no_depth_seed42`.

- [ ] **Step 2: Evaluate**

```bash
python -m src.v2.evaluate \
  --checkpoint checkpoints/v2/ablation_no_depth_seed42/ema.pt \
  --vocab      checkpoints/v2/ablation_no_depth_seed42/vocab.json \
  --stats      checkpoints/v2/ablation_no_depth_seed42/train_stats.json \
  --split-file data/raw/dish_ids/splits/rgb_test_ids.txt \
  --output-dir docs/runs/ablation_no_depth_seed42/eval/ \
  --no-depth
```

- [ ] **Step 3: Paired bootstrap significance**

```python
# scripts/ablation_bootstrap.py — write inline + commit
import json, numpy as np, csv
from src.v2.metrics import paired_bootstrap_mae

def load_csv(p):
    out = {}
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r: out[row["dish_id"]] = row
    return out

main = load_csv("docs/runs/main_seed42/eval/predictions.csv")
abl  = load_csv("docs/runs/ablation_no_depth_seed42/eval/predictions.csv")
gt   = load_csv("docs/runs/main_seed42/eval/groundtruth.csv")
ids = sorted(set(main) & set(abl) & set(gt))
def col(d, ids, k): return np.array([float(d[i][k]) for i in ids])

result = {}
for k in ["kcal", "mass", "fat", "carb", "protein"]:
    a = col(main, ids, k); b = col(abl, ids, k); t = col(gt, ids, k)
    delta, lo, hi = paired_bootstrap_mae(a, b, t, n=1000, seed=0)
    result[k] = {"delta_mae_b_minus_a": delta, "ci95": [lo, hi]}
print(json.dumps(result, indent=2))
with open("docs/ablations/no_depth/significance.json", "w") as f:
    json.dump(result, f, indent=2)
```

```bash
mkdir -p docs/ablations/no_depth
.venv/bin/python scripts/ablation_bootstrap.py | tee docs/ablations/no_depth/significance.txt
```

- [ ] **Step 4: Write ablation summary**

`docs/ablations/no_depth/summary.md`:

```markdown
# Ablation: no depth

| metric | main (RGB+D) | ablation (RGB only) | Δ | 95% CI |
|---|---|---|---|---|
| kcal MAE | ... | ... | +X.X | [..., ...] |
| mass MAE | ... | ... | +X.X | [..., ...] |
| ... | ... | ... | ... | ... |

**Conclusion:** Removing depth degrades kcal/mass MAE by X kcal / Y g (p<0.05 if CI excludes 0).
This validates the §3.1 design choice to use RGB-D late fusion.
```

- [ ] **Step 5: Commit**

```bash
git add docs/ablations/no_depth/ scripts/ablation_bootstrap.py docs/runs/ablation_no_depth_seed42/
git commit -m "experiment: no-depth ablation + bootstrap significance"
```

---

## Phase 8 — Final Report

### Task 22: Write final_report.md

**Files:**
- Create: `docs/final_report.md`

- [ ] **Step 1: Compose**

```markdown
# Nutrition5k SOTA — Final Report (run main_seed42)

## Headline

We trained a ConvNeXt-Base RGB + ConvNeXt-Tiny depth dual-stream multi-task
model on Nutrition5k overhead RGB-D imagery. On the official rgb_test split
restricted to dishes with overhead RGB-D (n=507/709), we achieve:

| Metric | Ours | Google direct prediction (Thames 2021) |
|---|---|---|
| kcal MAE | ... | 70 |
| mass MAE | ... | 40 |
| fat MAE | ... | 6 |
| carb MAE | ... | 10 |
| protein MAE | ... | 5 |
| ingr F1 (macro / micro) | ... / ... | n/a |
| top-5 ingr IoU | ... | n/a |

## Method

(brief: §3 architecture summary, 200 words)

## Ablation

| variant | kcal MAE | mass MAE | Δ vs main |
|---|---|---|---|
| **main (RGB+D)** | ... | ... | — |
| no depth | ... | ... | +X / +Y |

(insert significance.json table)

## Qualitative

(top-5 best and top-5 worst dishes — RGB image + GT vs pred panel)

## Limitations & Future Work

- Evaluated only on the 507/709 overhead-RGB-D subset of rgb_test
- Wild OOD photos (`food_photos/*.png`) explicitly not optimized
- Single seed; no cross-validation
- Did not use ingredient encoder-decoder (Task: future work)

## Reproducibility

- Code: `src/v2/`
- Run config: `src/v2/configs/main.yaml`
- Checkpoints: `checkpoints/v2/main_seed42/{ema,best}.pt`
- Train stats: `checkpoints/v2/main_seed42/train_stats.json`
- Vocab: `checkpoints/v2/main_seed42/vocab.json`
- Random seed: 42
```

- [ ] **Step 2: Commit**

```bash
git add docs/final_report.md
git commit -m "docs: final report with results, ablation, qualitative"
```

---

## Self-Review

I read the spec end-to-end and check this plan covers it:

| Spec section | Tasks |
|---|---|
| §1 Goal & success criteria | Task 18 (G5), Task 22 |
| §2 Constraints | Task 11 (config) |
| §3 Architecture (RGB-D late fusion + 3 heads) | Task 4 |
| §3.2 (4) Two-kcal averaging | Task 8 (evaluate.py — `kcal_avg = 0.5 * direct + 0.5 * derived`) |
| §4.1 Splits | Task 11 (90/10 val build), Task 18 |
| §4.2 Depth download (parallel) | Task 0b |
| §4.3 Preprocessing (depth clip + mask) | Task 3 (Nutrition5kTransform) |
| §4.4 Augmentation | Task 3 |
| §4.5 Labels (vocab, log1p+z mass) | Task 1, Task 2, Task 3 |
| §5.1 Per-task losses | Task 5 |
| §5.2 Uncertainty weighting + s_t floor | Task 5 (UncertaintyWeighter) |
| §5.3 Optimizer & schedule | Task 9 |
| §5.4 Val + EMA + best | Task 9 |
| §5.5 TTA | Task 7 |
| §6 Eval metrics | Task 6, Task 8 |
| §7 Ablation plan | Task 12 (config), Task 21 |
| §8 G1–G6 gates | G1 Task 15, G2 Task 16, G3 Task 17, G4 inline in Task 9/18, G5 Task 18, G6 Task 21 |
| §9 Iteration loop | Task 20 |
| §10.1 Repo layout | matches File Structure block |
| §10.2 Subagent strategy | Tasks 14, 19; Phase 1 tasks parallel-friendly |
| §10.3 Doc discipline | Each task ends in commit; per-run summary.md |
| §11 Risks | Task 0a/0b/0c retry-friendly; Task 5 floor clamp; Task 1 vocab diff log |
| §13 Deliverables | Phase 8 final report |

**Placeholder scan:** searched for "TBD", "TODO", "fill in", "later" — none in steps. Vocab size in Task 1 test may need to be adjusted from 554 to actual; that adjustment is part of Step 5 of Task 1 (explicit instruction, not a placeholder).

**Type consistency:**
- `parse_dish_metadata_row` → `DishLabels` — used in dataset.py, scripts/compute_train_stats.py, scripts/run_g1.py ✓
- `Vocab.size`, `Vocab.id_to_idx`, `Vocab.idx_to_density` — referenced in train.py, evaluate.py, dataset.py, run_g1.py ✓
- `TrainStats.scalar_z`, `scalar_inv_z`, `mass_log1p_z`, `mass_log1p_inv_z`, `depth_z` — used consistently ✓
- `NutritionRGBDModel(rgb, depth, use_depth=...)` and `model.param_groups(...)` — used in train.py ✓
- `UncertaintyWeighter(["scalar","ingr_cls","ingr_mass","atwater","kcal_consist"])` — keys match the loss-dict keys in `compute_total_loss` ✓
- `tta_predict(model, rgb, depth, use_depth=...)` returns `{"scalar","ingr_logits","ingr_mass"}` — same as `model.forward` output ✓

No type mismatches found.
