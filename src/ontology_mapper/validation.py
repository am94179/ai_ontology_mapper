"""Deterministic validation and finalization of provisional mapping reasoning."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .models import (
    ColumnProfile,
    DecisionStatus,
    MappingResult,
    ReasoningResult,
    Relationship,
    ValidationFinding,
    ValidationSeverity,
)

ACCEPT_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.70
LLM_CONFIDENCE_WEIGHT = 0.70
CANDIDATE_SCORE_WEIGHT = 0.30
TYPE_DIFFERENCE_PENALTY = 0.05
CURRENCY_CONFLICT_PENALTY = 0.25
KNOWN_CURRENCIES = frozenset({"usd", "eur", "gbp", "jpy"})
NUMERIC_TYPES = frozenset({"integer", "float"})


def validate_profiles(source: ColumnProfile, target: ColumnProfile) -> list[ValidationFinding]:
    """Return narrow, explainable evidence from two observed column profiles."""
    findings: list[ValidationFinding] = []
    source_currencies = _currency_tokens(source.name)
    target_currencies = _currency_tokens(target.name)
    if source_currencies and target_currencies:
        if source_currencies != target_currencies:
            findings.append(ValidationFinding(
                "EXPLICIT_CURRENCY_CONFLICT", ValidationSeverity.CONFLICT,
                "Column names contain different explicit currency tokens: "
                f"{', '.join(sorted(source_currencies)).upper()} versus "
                f"{', '.join(sorted(target_currencies)).upper()}.",
            ))
        else:
            findings.append(ValidationFinding(
                "EXPLICIT_CURRENCY_MATCH", ValidationSeverity.INFO,
                "Column names contain the same explicit currency token: "
                f"{', '.join(sorted(source_currencies)).upper()}.",
            ))
    if source.observed_dtype == target.observed_dtype:
        findings.append(ValidationFinding(
            "OBSERVED_TYPE_MATCH", ValidationSeverity.INFO,
            f"Observed data types match: {source.observed_dtype}.",
        ))
    elif {source.observed_dtype, target.observed_dtype} <= NUMERIC_TYPES:
        findings.append(ValidationFinding(
            "NUMERIC_TYPES_COMPATIBLE", ValidationSeverity.INFO,
            "Observed numeric types are compatible: "
            f"{source.observed_dtype} and {target.observed_dtype}.",
        ))
    else:
        findings.append(ValidationFinding(
            "OBSERVED_TYPE_DIFFER", ValidationSeverity.WARNING,
            "Observed data types differ: "
            f"{source.observed_dtype} and {target.observed_dtype}. "
            "This is not by itself a semantic conflict.",
        ))
    return findings


def finalize_mapping(source: ColumnProfile, target: ColumnProfile, reasoning: ReasoningResult) -> MappingResult:
    """Combine provisional reasoning with deterministic validation evidence."""
    findings = validate_profiles(source, target)
    currency_conflict = any(item.code == "EXPLICIT_CURRENCY_CONFLICT" for item in findings)
    relationship = Relationship.CONFLICT if currency_conflict else reasoning.relationship
    score = _final_score(reasoning, findings)
    return MappingResult(
        source_dataset=reasoning.source_dataset,
        source_column=reasoning.source_column,
        target_dataset=reasoning.target_dataset,
        target_column=reasoning.target_column,
        candidate_score=reasoning.candidate_score,
        candidate_method=reasoning.candidate_method,
        provisional_relationship=reasoning.relationship,
        llm_confidence=reasoning.confidence,
        explanation=reasoning.explanation,
        llm_conflicts=reasoning.conflicts,
        validation_findings=findings,
        relationship=relationship,
        score=score,
        status=_decision_status(relationship, score),
        reason=_final_reason(reasoning.relationship, relationship, findings),
    )


def finalize_mappings(
    source_columns: Mapping[str, ColumnProfile],
    target_columns: Mapping[str, ColumnProfile],
    reasoning_results: Sequence[ReasoningResult],
) -> list[MappingResult]:
    """Finalize all reasoning results using their corresponding column profiles."""
    results: list[MappingResult] = []
    for reasoning in reasoning_results:
        try:
            source = source_columns[reasoning.source_column]
            target = target_columns[reasoning.target_column]
        except KeyError as error:
            raise ValueError(f"Missing profile for reasoning column: {error.args[0]}") from error
        results.append(finalize_mapping(source, target, reasoning))
    return results


def _currency_tokens(name: str) -> frozenset[str]:
    split_case = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    tokens = {token.lower() for token in re.findall(r"[A-Za-z]+", split_case)}
    return frozenset(tokens & KNOWN_CURRENCIES)


def _final_score(reasoning: ReasoningResult, findings: Sequence[ValidationFinding]) -> float:
    candidate_score = min(max(reasoning.candidate_score, 0.0), 1.0)
    score = LLM_CONFIDENCE_WEIGHT * reasoning.confidence + CANDIDATE_SCORE_WEIGHT * candidate_score
    for finding in findings:
        if finding.code == "OBSERVED_TYPE_DIFFER":
            score -= TYPE_DIFFERENCE_PENALTY
        elif finding.code == "EXPLICIT_CURRENCY_CONFLICT":
            score -= CURRENCY_CONFLICT_PENALTY
    return min(max(score, 0.0), 1.0)


def _decision_status(relationship: Relationship, score: float) -> DecisionStatus:
    if relationship == Relationship.MATCH:
        if score >= ACCEPT_THRESHOLD:
            return DecisionStatus.ACCEPT
        if score >= REVIEW_THRESHOLD:
            return DecisionStatus.REVIEW
    if relationship == Relationship.RELATED:
        return DecisionStatus.REVIEW
    return DecisionStatus.REJECT


def _final_reason(provisional: Relationship, final: Relationship, findings: Sequence[ValidationFinding]) -> str:
    if final != provisional:
        conflict = next(item for item in findings if item.severity == ValidationSeverity.CONFLICT)
        return f"Deterministic validation overrides provisional {provisional}: {conflict.message}"
    if any(item.severity == ValidationSeverity.WARNING for item in findings):
        return "No deterministic conflict override; provisional relationship retained with validation warning."
    return "No deterministic conflict override; provisional relationship retained."
