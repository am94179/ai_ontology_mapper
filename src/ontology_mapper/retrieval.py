"""Conservative embedding retrieval diagnostics for schema-mapping candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .embeddings import TextEncoder, column_representation, cosine_similarity
from .matching import normalize_column_name
from .models import CandidateMapping, ColumnProfile, DatasetProfile, MatchMethod


class ObservedTypeCompatibility(StrEnum):
    """A deterministic compatibility signal, not a semantic relationship."""

    COMPATIBLE = "COMPATIBLE"
    IDENTIFIER_COMPATIBLE = "IDENTIFIER_COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PairEvidence:
    """Deterministic evidence recorded for one possible source-target pair."""

    source_column: str
    target_column: str
    source_observed_dtype: str
    target_observed_dtype: str
    exact_normalized_name: bool
    type_compatibility: ObservedTypeCompatibility
    retained: bool


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Counts and pair-level evidence for one embedding retrieval run."""

    total_possible_pairs: int
    pairs_retained: int
    embedding_ranked_pairs: int
    shortlisted_candidates: int
    pair_evidence: list[PairEvidence]


@dataclass(frozen=True)
class EmbeddingRetrieval:
    """Embedding candidates together with their deterministic funnel evidence."""

    candidates: list[CandidateMapping]
    diagnostics: RetrievalDiagnostics


def generate_embedding_retrieval(
    source: DatasetProfile,
    target: DatasetProfile,
    encoder: TextEncoder,
    *,
    top_k: int = 3,
    minimum_score: float = 0.35,
) -> EmbeddingRetrieval:
    """Generate embedding candidates after conservative observed-type narrowing.

    The encoder is invoked exactly once per dataset. Every possible pair is
    represented in diagnostics; only pairs classified as INCOMPATIBLE are kept
    out of embedding ranking.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not -1.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between -1 and 1")

    source_columns = source.columns
    target_columns = target.columns
    source_vectors = encoder.encode([column_representation(column) for column in source_columns])
    target_vectors = encoder.encode([column_representation(column) for column in target_columns])
    if len(source_vectors) != len(source_columns) or len(target_vectors) != len(target_columns):
        raise ValueError("encoder must return one embedding per input text")

    pair_evidence: list[PairEvidence] = []
    candidates: list[CandidateMapping] = []
    retained_pair_count = 0
    for source_column, source_vector in zip(source_columns, source_vectors, strict=True):
        source_normalized = normalize_column_name(source_column.name)
        ranked: list[CandidateMapping] = []
        for target_column, target_vector in zip(target_columns, target_vectors, strict=True):
            target_normalized = normalize_column_name(target_column.name)
            compatibility = observed_type_compatibility(source_column, target_column)
            retained = compatibility != ObservedTypeCompatibility.INCOMPATIBLE
            pair_evidence.append(
                PairEvidence(
                    source_column=source_column.name,
                    target_column=target_column.name,
                    source_observed_dtype=source_column.observed_dtype,
                    target_observed_dtype=target_column.observed_dtype,
                    exact_normalized_name=bool(source_normalized and source_normalized == target_normalized),
                    type_compatibility=compatibility,
                    retained=retained,
                )
            )
            if not retained:
                continue
            retained_pair_count += 1
            ranked.append(
                CandidateMapping(
                    source_dataset=source.dataset.identifier,
                    source_column=source_column.name,
                    target_dataset=target.dataset.identifier,
                    target_column=target_column.name,
                    score=cosine_similarity(source_vector, target_vector),
                    method=MatchMethod.EMBEDDING,
                    source_normalized=source_normalized,
                    target_normalized=target_normalized,
                )
            )
        candidates.extend(
            candidate
            for candidate in sorted(ranked, key=lambda item: (-item.score, item.target_column))[:top_k]
            if candidate.score >= minimum_score
        )

    return EmbeddingRetrieval(
        candidates=candidates,
        diagnostics=RetrievalDiagnostics(
            total_possible_pairs=len(source_columns) * len(target_columns),
            pairs_retained=retained_pair_count,
            embedding_ranked_pairs=retained_pair_count,
            shortlisted_candidates=len(candidates),
            pair_evidence=pair_evidence,
        ),
    )


def observed_type_compatibility(
    source: ColumnProfile, target: ColumnProfile
) -> ObservedTypeCompatibility:
    """Classify only clear physical-type incompatibilities conservatively."""
    source_type = source.observed_dtype
    target_type = target.observed_dtype
    if source_type == "empty" or target_type == "empty":
        return ObservedTypeCompatibility.UNKNOWN
    if source_type == target_type or {source_type, target_type} <= {"integer", "float"}:
        return ObservedTypeCompatibility.COMPATIBLE
    if {source_type, target_type} <= {"string", "integer", "float"}:
        if _is_identifier(source.name) and _is_identifier(target.name):
            return ObservedTypeCompatibility.IDENTIFIER_COMPATIBLE
        return ObservedTypeCompatibility.INCOMPATIBLE
    if "boolean" in {source_type, target_type} and (
        "integer" in {source_type, target_type}
        or "float" in {source_type, target_type}
        or "date" in {source_type, target_type}
    ):
        return ObservedTypeCompatibility.INCOMPATIBLE
    return ObservedTypeCompatibility.UNKNOWN


def _is_identifier(name: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name.lower()))
    return bool({"id", "identifier", "key", "uuid"} & set(tokens))
