"""Typed cache integration for deterministic schema-mapping pipeline stages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable, TypeVar

from .cache import FileCache, file_fingerprint, fingerprint
from .embeddings import OnnxEmbeddingEncoder
from .models import CandidateMapping, ColumnProfile, DatasetProfile, DatasetRef, MatchMethod
from .profiling import profile_dataset
from .retrieval import (
    EmbeddingRetrieval,
    ObservedTypeCompatibility,
    PairEvidence,
    RetrievalDiagnostics,
    generate_embedding_retrieval,
)
from .routing import (
    CandidateRouting,
    RoutingAction,
    RoutingDecision,
    RoutingDiagnostics,
    route_embedding_candidates,
)

PIPELINE_CACHE_VERSION = 1
Value = TypeVar("Value")


class PipelineCache:
    """Cache profiles, retrieval, and routing without changing their typed API."""

    def __init__(self, cache: FileCache | None) -> None:
        self._cache = cache
        self._hits: dict[str, int] = defaultdict(int)
        self._misses: dict[str, int] = defaultdict(int)

    @property
    def file_cache(self) -> FileCache | None:
        return self._cache

    @property
    def enabled(self) -> bool:
        return self._cache is not None

    def diagnostics(self) -> dict[str, dict[str, int]]:
        return {
            "hits": dict(sorted(self._hits.items())),
            "misses": dict(sorted(self._misses.items())),
        }

    def record_batch_reasoning(self, hits: int, misses: int) -> None:
        if self.enabled:
            self._hits["batch_reasoning"] += hits
            self._misses["batch_reasoning"] += misses

    def profile(self, reference: DatasetRef, sample_size: int) -> DatasetProfile:
        return self._get_or_create(
            "profiles",
            {
                "input_content": file_fingerprint(reference.path),
                "dataset": asdict(reference),
                "sample_size": sample_size,
            },
            lambda: profile_dataset(reference, sample_size),
            _dataset_profile_from_dict,
        )

    def embedding_retrieval(
        self,
        source: DatasetProfile,
        target: DatasetProfile,
        *,
        model_id: str,
        top_k: int,
        minimum_score: float,
    ) -> EmbeddingRetrieval:
        return self._get_or_create(
            "embedding_retrieval",
            {
                "source_profile": asdict(source),
                "target_profile": asdict(target),
                "model_id": model_id,
                "top_k": top_k,
                "minimum_score": minimum_score,
            },
            lambda: generate_embedding_retrieval(
                source, target, OnnxEmbeddingEncoder(model_id), top_k=top_k, minimum_score=minimum_score
            ),
            _embedding_retrieval_from_dict,
        )

    def routing(
        self,
        candidates: list[CandidateMapping],
        diagnostics: RetrievalDiagnostics,
        *,
        minimum_score: float,
        ambiguity_margin: float,
    ) -> CandidateRouting:
        return self._get_or_create(
            "routing",
            {
                "candidates": [asdict(candidate) for candidate in candidates],
                "retrieval_diagnostics": asdict(diagnostics),
                "minimum_score": minimum_score,
                "ambiguity_margin": ambiguity_margin,
            },
            lambda: route_embedding_candidates(
                candidates, diagnostics, minimum_score=minimum_score, ambiguity_margin=ambiguity_margin
            ),
            _candidate_routing_from_dict,
        )

    def _get_or_create(
        self,
        namespace: str,
        identity: dict[str, Any],
        create: Callable[[], Value],
        decode: Callable[[dict[str, Any]], Value],
    ) -> Value:
        if not self._cache:
            return create()
        key = fingerprint({"artifact": namespace, "version": PIPELINE_CACHE_VERSION, "identity": identity})
        cached = self._cache.get(namespace, key)
        if isinstance(cached, dict):
            try:
                value = decode(cached)
            except (KeyError, TypeError, ValueError):
                value = None
            if value is not None:
                self._hits[namespace] += 1
                return value
        self._misses[namespace] += 1
        value = create()
        self._cache.set(namespace, key, asdict(value))
        return value


def _dataset_profile_from_dict(value: dict[str, Any]) -> DatasetProfile:
    return DatasetProfile(
        dataset=DatasetRef(**value["dataset"]),
        row_count=value["row_count"],
        columns=[ColumnProfile(**column) for column in value["columns"]],
    )


def _candidate_from_dict(value: dict[str, Any]) -> CandidateMapping:
    return CandidateMapping(
        source_dataset=value["source_dataset"],
        source_column=value["source_column"],
        target_dataset=value["target_dataset"],
        target_column=value["target_column"],
        score=value["score"],
        method=MatchMethod(value["method"]),
        source_normalized=value["source_normalized"],
        target_normalized=value["target_normalized"],
    )


def _retrieval_diagnostics_from_dict(value: dict[str, Any]) -> RetrievalDiagnostics:
    return RetrievalDiagnostics(
        total_possible_pairs=value["total_possible_pairs"],
        pairs_retained=value["pairs_retained"],
        embedding_ranked_pairs=value["embedding_ranked_pairs"],
        shortlisted_candidates=value["shortlisted_candidates"],
        pair_evidence=[
            PairEvidence(
                source_column=item["source_column"],
                target_column=item["target_column"],
                source_observed_dtype=item["source_observed_dtype"],
                target_observed_dtype=item["target_observed_dtype"],
                exact_normalized_name=item["exact_normalized_name"],
                type_compatibility=ObservedTypeCompatibility(item["type_compatibility"]),
                retained=item["retained"],
            )
            for item in value["pair_evidence"]
        ],
    )


def _embedding_retrieval_from_dict(value: dict[str, Any]) -> EmbeddingRetrieval:
    return EmbeddingRetrieval(
        candidates=[_candidate_from_dict(candidate) for candidate in value["candidates"]],
        diagnostics=_retrieval_diagnostics_from_dict(value["diagnostics"]),
    )


def _candidate_routing_from_dict(value: dict[str, Any]) -> CandidateRouting:
    decisions = [
        RoutingDecision(
            candidate=_candidate_from_dict(item["candidate"]),
            rank=item["rank"],
            top_score_margin=item["top_score_margin"],
            exact_normalized_name=item["exact_normalized_name"],
            type_compatibility=ObservedTypeCompatibility(item["type_compatibility"]),
            action=RoutingAction(item["action"]),
            reason=item["reason"],
        )
        for item in value["decisions"]
    ]
    diagnostics = value["diagnostics"]
    return CandidateRouting(
        decisions=decisions,
        reasoning_candidates=[_candidate_from_dict(item) for item in value["reasoning_candidates"]],
        diagnostics=RoutingDiagnostics(**diagnostics),
    )
