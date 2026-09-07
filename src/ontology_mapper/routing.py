"""Confidence-based routing from embedding candidates to LLM reasoning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from .models import CandidateMapping
from .retrieval import ObservedTypeCompatibility, PairEvidence, RetrievalDiagnostics

DEFAULT_ROUTING_MINIMUM_SCORE = 0.80
DEFAULT_AMBIGUITY_MARGIN = 0.03


class RoutingAction(StrEnum):
    """Disposition of a retrieved candidate before semantic reasoning."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    REASON = "REASON"
    NO_DECISION = "NO_DECISION"


@dataclass(frozen=True)
class RoutingDecision:
    """Explainable decision about whether one candidate needs LLM reasoning."""

    candidate: CandidateMapping
    rank: int
    top_score_margin: float | None
    exact_normalized_name: bool
    type_compatibility: ObservedTypeCompatibility
    action: RoutingAction
    reason: str


@dataclass(frozen=True)
class RoutingDiagnostics:
    """Summary of LLM routing after embedding candidate retrieval."""

    source_columns: int
    input_candidates: int
    reason_candidates: int
    no_decision_candidates: int
    auto_accept_candidates: int
    projected_llm_call_reduction: int


@dataclass(frozen=True)
class CandidateRouting:
    """All routing decisions and the exact candidates selected for reasoning."""

    decisions: list[RoutingDecision]
    reasoning_candidates: list[CandidateMapping]
    diagnostics: RoutingDiagnostics


def route_embedding_candidates(
    candidates: list[CandidateMapping],
    retrieval_diagnostics: RetrievalDiagnostics,
    *,
    minimum_score: float = DEFAULT_ROUTING_MINIMUM_SCORE,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> CandidateRouting:
    """Route one top candidate, plus a close runner-up, per source column.

    AUTO_ACCEPT remains available for a later calibrated workflow but is not
    emitted here. Routing therefore never creates a semantic mapping decision.
    """
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1")
    if not 0.0 <= ambiguity_margin <= 1.0:
        raise ValueError("ambiguity_margin must be between 0 and 1")

    evidence_by_pair = {
        (item.source_column, item.target_column): item
        for item in retrieval_diagnostics.pair_evidence
    }
    grouped: dict[str, list[CandidateMapping]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.source_column].append(candidate)

    decisions: list[RoutingDecision] = []
    reasoning_candidates: list[CandidateMapping] = []
    for source_column in sorted(grouped):
        ranked = sorted(grouped[source_column], key=lambda item: (-item.score, item.target_column))
        top_score = ranked[0].score
        top_margin = top_score - ranked[1].score if len(ranked) > 1 else None
        reason_ranks: set[int] = set()
        if top_score >= minimum_score:
            reason_ranks.add(1)
            if top_margin is not None and top_margin <= ambiguity_margin:
                reason_ranks.add(2)

        for rank, candidate in enumerate(ranked, start=1):
            evidence = _evidence_for(candidate, evidence_by_pair)
            if rank in reason_ranks:
                action = RoutingAction.REASON
                reason = _reason_for_selected_candidate(rank, top_margin, ambiguity_margin)
                reasoning_candidates.append(candidate)
            else:
                action = RoutingAction.NO_DECISION
                reason = _reason_for_unselected_candidate(
                    rank, top_score, minimum_score, top_margin, ambiguity_margin
                )
            decisions.append(
                RoutingDecision(
                    candidate=candidate,
                    rank=rank,
                    top_score_margin=top_margin,
                    exact_normalized_name=evidence.exact_normalized_name,
                    type_compatibility=evidence.type_compatibility,
                    action=action,
                    reason=reason,
                )
            )

    reason_count = len(reasoning_candidates)
    return CandidateRouting(
        decisions=decisions,
        reasoning_candidates=reasoning_candidates,
        diagnostics=RoutingDiagnostics(
            source_columns=len(grouped),
            input_candidates=len(candidates),
            reason_candidates=reason_count,
            no_decision_candidates=len(candidates) - reason_count,
            auto_accept_candidates=0,
            projected_llm_call_reduction=len(candidates) - reason_count,
        ),
    )


def _evidence_for(
    candidate: CandidateMapping, evidence_by_pair: dict[tuple[str, str], PairEvidence]
) -> PairEvidence:
    try:
        return evidence_by_pair[(candidate.source_column, candidate.target_column)]
    except KeyError as error:
        raise ValueError(
            "Missing retrieval evidence for candidate: "
            f"{candidate.source_column} -> {candidate.target_column}"
        ) from error


def _reason_for_selected_candidate(
    rank: int, top_margin: float | None, ambiguity_margin: float
) -> str:
    if rank == 1 and top_margin is not None and top_margin <= ambiguity_margin:
        return "Top candidate meets the score threshold but has a close runner-up; reason over both."
    if rank == 2:
        return "Runner-up is within the configured ambiguity margin of the top candidate."
    return "Top candidate meets the configured retrieval score threshold."


def _reason_for_unselected_candidate(
    rank: int,
    top_score: float,
    minimum_score: float,
    top_margin: float | None,
    ambiguity_margin: float,
) -> str:
    if top_score < minimum_score:
        return f"Top score {top_score:.3f} is below routing minimum {minimum_score:.3f}."
    if rank == 2 and top_margin is not None and top_margin > ambiguity_margin:
        return "Runner-up is outside the configured ambiguity margin."
    return "Only the top candidate and an optional close runner-up are routed for reasoning."
