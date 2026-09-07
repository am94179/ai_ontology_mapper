"""Deterministic lexical candidate generation for profiled datasets."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .models import CandidateMapping, DatasetProfile, MatchMethod


def normalize_column_name(name: str) -> str:
    """Return a compact, case-insensitive form suitable for lexical comparison."""
    split_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return "".join(re.findall(r"[a-z0-9]+", split_case.lower()))


def generate_candidates(
    source: DatasetProfile,
    target: DatasetProfile,
    *,
    top_k: int = 3,
    minimum_score: float = 0.6,
) -> list[CandidateMapping]:
    """Rank lexical column-name candidates for every source column."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1")

    candidates: list[CandidateMapping] = []
    for source_column in source.columns:
        source_normalized = normalize_column_name(source_column.name)
        column_candidates = [
            _candidate_for_pair(
                source.dataset.identifier,
                source_column.name,
                source_normalized,
                target.dataset.identifier,
                target_column.name,
            )
            for target_column in target.columns
        ]
        candidates.extend(
            candidate
            for candidate in sorted(
                column_candidates, key=lambda item: (-item.score, item.target_column)
            )[:top_k]
            if candidate.score >= minimum_score
        )
    return candidates


def _candidate_for_pair(
    source_dataset: str,
    source_column: str,
    source_normalized: str,
    target_dataset: str,
    target_column: str,
) -> CandidateMapping:
    target_normalized = normalize_column_name(target_column)
    if source_normalized and source_normalized == target_normalized:
        score, method = 1.0, MatchMethod.EXACT_NORMALIZED
    else:
        score = SequenceMatcher(None, source_normalized, target_normalized).ratio()
        method = MatchMethod.FUZZY_NAME
    return CandidateMapping(
        source_dataset=source_dataset,
        source_column=source_column,
        target_dataset=target_dataset,
        target_column=target_column,
        score=score,
        method=method,
        source_normalized=source_normalized,
        target_normalized=target_normalized,
    )
