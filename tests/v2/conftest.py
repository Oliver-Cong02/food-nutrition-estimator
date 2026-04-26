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
