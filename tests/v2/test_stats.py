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
    np.testing.assert_array_equal(s.scalar_std, s2.scalar_std)
    assert s.depth_std == s2.depth_std
    assert s.mass_log1p_mean == s2.mass_log1p_mean
    assert s.mass_log1p_std == s2.mass_log1p_std


def test_mass_log1p_z_inv():
    s = TrainStats(scalar_mean=np.zeros(5, np.float32), scalar_std=np.ones(5, np.float32),
                   depth_mean=0.0, depth_std=1.0,
                   mass_log1p_mean=2.0, mass_log1p_std=1.0)
    grams = np.array([0.0, 5.0, 50.0, 200.0], dtype=np.float32)
    z = s.mass_log1p_z(grams)
    back = s.mass_log1p_inv_z(z)
    np.testing.assert_allclose(back, grams, rtol=1e-5, atol=1e-5)
