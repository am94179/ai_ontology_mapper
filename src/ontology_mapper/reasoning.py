"""Provider-independent LLM reasoning over retrieved mapping candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from .models import CandidateMapping, ColumnProfile, ReasoningResult, Relationship


class ReasoningError(ValueError):
    """Raised when a provider response cannot become a valid reasoning result."""


class ReasoningClient(Protocol):
    """The small boundary required by the reasoning orchestration."""

    def reason(self, prompt: str) -> Mapping[str, Any]: ...


def build_reasoning_prompt(
    source: ColumnProfile, target: ColumnProfile, candidate: CandidateMapping
) -> str:
    """Build a self-contained prompt for exactly one retrieved candidate pair."""
    evidence = {
        "candidate": asdict(candidate),
        "source_column_profile": asdict(source),
        "target_column_profile": asdict(target),
    }
    return (
        "Classify the semantic relationship between the source and target columns. "
        "MATCH means they represent the same concept. RELATED means they are meaningfully "
        "connected but not interchangeable. CONFLICT means they appear to represent a similar "
        "concept but have an incompatible unit, currency, time meaning, or granularity. "
        "NO_MATCH means they are unrelated. Use only the supplied evidence and do not treat the "
        "retrieval score as a final decision.\n\n"
        f"Evidence:\n{json.dumps(evidence, indent=2, default=str)}"
    )


def reason_candidate(
    source: ColumnProfile,
    target: ColumnProfile,
    candidate: CandidateMapping,
    client: ReasoningClient,
) -> ReasoningResult:
    """Reason over one candidate and locally validate the structured response."""
    payload = client.reason(build_reasoning_prompt(source, target, candidate))
    relationship, confidence, explanation, conflicts = validate_reasoning_payload(payload)
    return ReasoningResult(
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


def reason_candidates(
    source_columns: Mapping[str, ColumnProfile],
    target_columns: Mapping[str, ColumnProfile],
    candidates: Sequence[CandidateMapping],
    client: ReasoningClient,
) -> list[ReasoningResult]:
    """Reason over candidates whose corresponding profiles are available."""
    results: list[ReasoningResult] = []
    for candidate in candidates:
        try:
            source = source_columns[candidate.source_column]
            target = target_columns[candidate.target_column]
        except KeyError as error:
            raise ReasoningError(f"Missing profile for candidate column: {error.args[0]}") from error
        results.append(reason_candidate(source, target, candidate, client))
    return results


def validate_reasoning_payload(
    payload: Mapping[str, Any],
) -> tuple[Relationship, float, str, list[str]]:
    """Validate the provider contract without trusting provider output blindly."""
    expected = {"relationship", "confidence", "explanation", "conflicts"}
    if set(payload) != expected:
        raise ReasoningError(f"Reasoning response must contain exactly: {sorted(expected)}")
    try:
        relationship = Relationship(payload["relationship"])
    except (TypeError, ValueError) as error:
        raise ReasoningError("Reasoning response contains an unknown relationship") from error
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ReasoningError("Reasoning confidence must be a number")
    if not 0.0 <= confidence <= 1.0:
        raise ReasoningError("Reasoning confidence must be between 0 and 1")
    explanation = payload["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ReasoningError("Reasoning explanation must be a non-empty string")
    conflicts = payload["conflicts"]
    if not isinstance(conflicts, list) or not all(isinstance(item, str) for item in conflicts):
        raise ReasoningError("Reasoning conflicts must be a list of strings")
    return relationship, float(confidence), explanation, conflicts
