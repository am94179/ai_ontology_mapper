"""Local CSV loading with intentionally minimal dependencies."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import DatasetRef


def load_csv(dataset: DatasetRef) -> tuple[list[str], list[dict[str, str]]]:
    """Load a CSV into header names and rows while preserving values as strings."""
    path = Path(dataset.path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV must contain a header row: {path}")
        return list(reader.fieldnames), list(reader)

