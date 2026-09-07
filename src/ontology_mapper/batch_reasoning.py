"""Strict batched reasoning over routed candidates for one source column."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .models import CandidateMapping, ColumnProfile, ReasoningResult
from .reasoning import ReasoningError, validate_reasoning_payload
from .cache import FileCache, fingerprint

DEFAULT_REASONING_BATCH_SIZE = 2
BATCH_REASONING_CACHE_VERSION = 1


class BatchReasoningClient(Protocol):
    """Provider boundary for one strict structured batch request."""

    def reason_batch(self, prompt: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class BatchReasoningDiagnostics:
    """Counts for a completed sequence of sequential batch requests."""

    input_candidates: int
    batches: int
    provider_requests: int
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass(frozen=True)
class BatchReasoningRun:
    """Individual results recovered from validated provider batch responses."""

    results: list[ReasoningResult]
    diagnostics: BatchReasoningDiagnostics


def group_candidate_batches(
    candidates: Sequence[CandidateMapping], maximum_batch_size: int = DEFAULT_REASONING_BATCH_SIZE
) -> list[list[CandidateMapping]]:
    """Group candidates by source column and chunk them deterministically."""
    if maximum_batch_size < 1:
        raise ValueError("maximum_batch_size must be at least 1")
    grouped: dict[tuple[str, str], list[CandidateMapping]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.source_dataset, candidate.source_column)].append(candidate)

    batches: list[list[CandidateMapping]] = []
    for key in sorted(grouped):
        ranked = sorted(grouped[key], key=lambda item: (-item.score, item.target_column))
        for start in range(0, len(ranked), maximum_batch_size):
            batches.append(ranked[start : start + maximum_batch_size])
    return batches


def batch_reasoning_plan(
    candidates: Sequence[CandidateMapping], maximum_batch_size: int = DEFAULT_REASONING_BATCH_SIZE
) -> BatchReasoningDiagnostics:
    """Return the sequential provider-request plan without calling a provider."""
    batches = group_candidate_batches(candidates, maximum_batch_size)
    return BatchReasoningDiagnostics(len(candidates), len(batches), len(batches))


def build_batch_reasoning_prompt(
    source: ColumnProfile,
    targets: Mapping[str, ColumnProfile],
    candidates: Sequence[CandidateMapping],
) -> str:
    """Build compact evidence for candidates sharing exactly one source column."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    if any(candidate.source_column != source.name for candidate in candidates):
        raise ValueError("all batch candidates must share the source column")
    assessments = []
    for candidate in candidates:
        try:
            target = targets[candidate.target_column]
        except KeyError as error:
            raise ReasoningError(f"Missing profile for candidate column: {error.args[0]}") from error
        assessments.append({"candidate": asdict(candidate), "target_column_profile": asdict(target)})
    evidence = {"source_column_profile": asdict(source), "candidates": assessments}
    return (
        "Classify the semantic relationship for every requested candidate. "
        "MATCH means the fields represent the same concept. RELATED means they are connected but "
        "not interchangeable. CONFLICT means similar concepts have an incompatible unit, currency, "
        "time meaning, or granularity. NO_MATCH means they are unrelated. Use only supplied evidence. "
        "Return exactly one assessment for each requested target_column.\n\n"
        f"Evidence:\n{json.dumps(evidence, indent=2, default=str)}"
    )


def reason_candidate_batches(
    source_columns: Mapping[str, ColumnProfile],
    target_columns: Mapping[str, ColumnProfile],
    candidates: Sequence[CandidateMapping],
    client: BatchReasoningClient,
    *,
    maximum_batch_size: int = DEFAULT_REASONING_BATCH_SIZE,
    cache: FileCache | None = None,
    cache_context: Mapping[str, Any] | None = None,
) -> BatchReasoningRun:
    """Request and validate sequential source-column batch assessments."""
    batches = group_candidate_batches(candidates, maximum_batch_size)
    results: list[ReasoningResult] = []
    cache_hits = 0
    cache_misses = 0
    provider_requests = 0
    for batch in batches:
        source_name = batch[0].source_column
        try:
            source = source_columns[source_name]
        except KeyError as error:
            raise ReasoningError(f"Missing profile for candidate column: {error.args[0]}") from error
        prompt = build_batch_reasoning_prompt(source, target_columns, batch)
        cache_key = fingerprint({
            "artifact": "batch_reasoning",
            "version": BATCH_REASONING_CACHE_VERSION,
            "context": cache_context or {},
            "prompt": prompt,
        })
        payload = cache.get("batch_reasoning", cache_key) if cache else None
        from_cache = payload is not None
        if from_cache:
            cache_hits += 1
        else:
            if cache:
                cache_misses += 1
            payload = client.reason_batch(prompt)
            provider_requests += 1
        batch_results = _results_from_batch_payload(batch, payload)
        if cache and not from_cache:
            cache.set("batch_reasoning", cache_key, payload)
        results.extend(batch_results)
    return BatchReasoningRun(
        results=results,
        diagnostics=BatchReasoningDiagnostics(
            len(candidates), len(batches), provider_requests, cache_hits, cache_misses
        ),
    )


def _results_from_batch_payload(
    candidates: Sequence[CandidateMapping], payload: Mapping[str, Any]
) -> list[ReasoningResult]:
    if set(payload) != {"assessments"} or not isinstance(payload["assessments"], list):
        raise ReasoningError("Batch reasoning response must contain exactly an assessments list")
    expected = {candidate.target_column: candidate for candidate in candidates}
    received: dict[str, Mapping[str, Any]] = {}
    for assessment in payload["assessments"]:
        if not isinstance(assessment, Mapping) or set(assessment) != {
            "target_column", "relationship", "confidence", "explanation", "conflicts"
        }:
            raise ReasoningError("Each batch assessment must contain exactly the required fields")
        target_column = assessment["target_column"]
        if not isinstance(target_column, str) or target_column not in expected:
            raise ReasoningError("Batch response contains an unexpected target_column")
        if target_column in received:
            raise ReasoningError("Batch response contains a duplicate target_column")
        received[target_column] = assessment
    if set(received) != set(expected):
        raise ReasoningError("Batch response must include exactly one assessment for every requested candidate")

    results: list[ReasoningResult] = []
    for candidate in candidates:
        assessment = received[candidate.target_column]
        relationship, confidence, explanation, conflicts = validate_reasoning_payload(
            {key: assessment[key] for key in ("relationship", "confidence", "explanation", "conflicts")}
        )
        results.append(
            ReasoningResult(
                source_dataset=candidate.source_dataset,
                source_column=candidate.source_column,
                target_dataset=candidate.target_dataset,
                target_column=candidate.target_column,
                candidate_score=candidate.score,
                candidate_method=candidate.method,
                relationship=relationship,
                confidence=confidence,
                explanation=explanation,
                conflicts=conflicts,
            )
        )
    return results
