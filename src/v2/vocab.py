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
        id_to_idx:        dict from "ingr_XXXXXXXXXX" string to int
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
