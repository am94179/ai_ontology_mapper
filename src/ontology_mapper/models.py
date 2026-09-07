"""Stable, JSON-serializable models for observed dataset metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Relationship(StrEnum):
    """Relationships that later matching phases may assign to a column pair."""

    MATCH = "MATCH"
    RELATED = "RELATED"
    CONFLICT = "CONFLICT"
    NO_MATCH = "NO_MATCH"


class DecisionStatus(StrEnum):
    """Operational disposition for a finalized mapping result."""

    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class ValidationSeverity(StrEnum):
    """Severity assigned to deterministic validation evidence."""

    INFO = "INFO"
    WARNING = "WARNING"
    CONFLICT = "CONFLICT"


class MatchMethod(StrEnum):
    """Deterministic methods that can propose a candidate mapping."""

    EXACT_NORMALIZED = "EXACT_NORMALIZED"
    FUZZY_NAME = "FUZZY_NAME"
    EMBEDDING = "EMBEDDING"


@dataclass(frozen=True)
class DatasetRef:
    """A local tabular input known to the application."""

    identifier: str
    path: str
    format: str = "csv"
    display_name: str | None = None

    @classmethod
    def from_path(cls, path: str | Path, identifier: str | None = None) -> "DatasetRef":
        source_path = Path(path)
        if source_path.suffix.lower() != ".csv":
            raise ValueError(f"Only CSV input is supported in Phase 1: {source_path}")
        return cls(
            identifier=identifier or source_path.stem,
            path=str(source_path),
            format="csv",
            display_name=source_path.stem,
        )


@dataclass(frozen=True)
class ColumnProfile:
    """Facts observed from a single CSV column; no semantic inference is stored here."""

    name: str
    observed_dtype: str
    row_count: int
    null_count: int
    null_rate: float
    distinct_count: int
    unique_rate: float
    sample_values: list[str]
    minimum: float | int | None = None
    maximum: float | int | None = None
    mean: float | None = None
    median: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    mean_length: float | None = None


@dataclass(frozen=True)
class DatasetProfile:
    """Observed metadata for an entire input dataset."""

    dataset: DatasetRef
    row_count: int
    columns: list[ColumnProfile]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready data without relying on a custom JSON encoder."""
        return asdict(self)


@dataclass(frozen=True)
class GroundTruthMapping:
    """A known relationship used only to evaluate later matching phases."""

    source_dataset: str
    source_column: str
    target_dataset: str
    target_column: str
    expected_relationship: Relationship
    notes: str | None = None


@dataclass(frozen=True)
class CandidateMapping:
    """A ranked lexical candidate, not a semantic mapping decision."""

    source_dataset: str
    source_column: str
    target_dataset: str
    target_column: str
    score: float
    method: MatchMethod
    source_normalized: str
    target_normalized: str

@dataclass(frozen=True)
class ReasoningResult:
    """A provisional structured LLM assessment of one retrieved candidate."""

    source_dataset: str
    source_column: str
    target_dataset: str
    target_column: str
    candidate_score: float
    candidate_method: MatchMethod
    relationship: Relationship
    confidence: float
    explanation: str
    conflicts: list[str]


@dataclass(frozen=True)
class ValidationFinding:
    """One explainable result from deterministic mapping validation."""

    code: str
    severity: ValidationSeverity
    message: str


@dataclass(frozen=True)
class MappingResult:
    """A finalized mapping assessment combining LLM and deterministic evidence."""

    source_dataset: str
    source_column: str
    target_dataset: str
    target_column: str
    candidate_score: float
    candidate_method: MatchMethod
    provisional_relationship: Relationship
    llm_confidence: float
    explanation: str
    llm_conflicts: list[str]
    validation_findings: list[ValidationFinding]
    relationship: Relationship
    score: float
    status: DecisionStatus
    reason: str

@dataclass(frozen=True)
class CanonicalField:
    """One dataset column represented in a canonical concept."""

    dataset: str
    column: str


@dataclass(frozen=True)
class CanonicalConcept:
    """A flat canonical concept supported by accepted mapping evidence."""

    identifier: str
    canonical_name: str
    fields: list[CanonicalField]
    mapping_results: list[MappingResult]


@dataclass(frozen=True)
class CanonicalSchema:
    """Canonical concepts together with mappings not asserted as membership."""

    concepts: list[CanonicalConcept]
    review_mappings: list[MappingResult]
    rejected_mappings: list[MappingResult]
    ambiguous_mappings: list[MappingResult]

@dataclass(frozen=True)
class CandidateEvaluation:
    """Retrieval coverage for expected MATCH pairs under one strategy."""

    strategy: str
    top_k: int
    expected_matches: int
    retrieved_matches: int
    recall_at_k: float


@dataclass(frozen=True)
class RelationshipEvaluation:
    """Fixture-limited metrics for final relationship predictions."""

    evaluated_pairs: int
    correct_relationships: int
    relationship_accuracy: float
    match_true_positives: int
    match_false_positives: int
    match_false_negatives: int
    match_precision: float
    match_recall: float
    match_f1: float
    false_positive_rate: float
    missing_predictions: int
    unlabeled_predictions: int

@dataclass(frozen=True)
class FunnelEvaluation:
    """Compact wide-schema funnel coverage, reduction, and reuse metrics."""

    total_possible_pairs: int
    pairs_retained: int
    filter_reduction_rate: float
    shortlisted_candidates: int
    shortlist_reduction_rate: float
    routed_candidates: int
    routing_reduction_rate: float
    planned_batches: int
    planned_provider_requests: int
    planned_request_reduction_rate: float
    routed_match_recall: float
    cache_hits: dict[str, int] | None
    cache_misses: dict[str, int] | None
    routed_missing_expected_matches: list[GroundTruthMapping]
