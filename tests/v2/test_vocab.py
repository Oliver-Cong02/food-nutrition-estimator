"""Tests for src/v2/vocab.py."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.v2.vocab import Vocab


def test_build_from_csv_has_555_entries(ingredients_csv):
    v = Vocab.from_csv(ingredients_csv)
    assert v.size == 555, f"expected 555 (1-555), got {v.size}"


def test_id_to_idx_roundtrip(ingredients_csv):
    v = Vocab.from_csv(ingredients_csv)
    for ingr_id, idx in v.id_to_idx.items():
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
    assert v.idx_to_id == v2.idx_to_id
    assert v.idx_to_name == v2.idx_to_name
