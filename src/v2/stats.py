# src/v2/stats.py
"""Train-set z-score / log1p statistics, used to (de)normalize labels and inputs.

All values stored as plain Python types in JSON for portability.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(eq=False)
class TrainStats:
    scalar_mean: np.ndarray  # shape (5,)  — kcal, mass, fat, carb, protein
    scalar_std: np.ndarray   # shape (5,)
    depth_mean: float        # over valid (>0) pixels in train, after clip [200,800]
    depth_std: float
    mass_log1p_mean: float   # over all positive per-ingredient grams in train
    mass_log1p_std: float

    def scalar_z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.scalar_mean) / (self.scalar_std + 1e-6)

    def scalar_inv_z(self, z: np.ndarray) -> np.ndarray:
        return z * self.scalar_std + self.scalar_mean

    def depth_z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.depth_mean) / (self.depth_std + 1e-6)

    def mass_log1p_z(self, grams: np.ndarray) -> np.ndarray:
        return (np.log1p(grams) - self.mass_log1p_mean) / (self.mass_log1p_std + 1e-6)

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
    def load(cls, path: Path | str) -> TrainStats:
        d = json.loads(Path(path).read_text())
        return cls(
            scalar_mean=np.array(d["scalar_mean"], dtype=np.float32),
            scalar_std=np.array(d["scalar_std"], dtype=np.float32),
            depth_mean=d["depth_mean"],
            depth_std=d["depth_std"],
            mass_log1p_mean=d["mass_log1p_mean"],
            mass_log1p_std=d["mass_log1p_std"],
        )
