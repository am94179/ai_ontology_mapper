"""Small evaluation helpers for candidate generation."""

from __future__ import annotations

import json
from pathlib import Path

from .models import CandidateMapping, GroundTruthMapping, Relationship


def load_ground_truth(path: str | Path) -> list[GroundTruthMapping]:
    """Load the project's small, hand-authored mapping fixture."""
    raw_mappings = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GroundTruthMapping(
            source_dataset=item["source_dataset"],
            source_column=item["source_column"],
            target_dataset=item["target_dataset"],
            target_column=item["target_column"],
            expected_relationship=Relationship(item["expected_relationship"]),
            notes=item.get("notes"),
        )
        for item in raw_mappings
    ]


def candidate_recall_at_k(
    candidates: list[CandidateMapping], ground_truth: list[GroundTruthMapping]
) -> float:
    """Measure whether expected MATCH pairs appear in the generated candidates."""
    expected = {
        (item.source_dataset, item.source_column, item.target_dataset, item.target_column)
        for item in ground_truth
        if item.expected_relationship == Relationship.MATCH
    }
    found = {
        (item.source_dataset, item.source_column, item.target_dataset, item.target_column)
        for item in candidates
    }
    return len(expected & found) / len(expected) if expected else 0.0
