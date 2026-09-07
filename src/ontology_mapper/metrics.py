"""Fixture-limited metrics for candidate retrieval and final mappings."""

from __future__ import annotations

from collections.abc import Mapping

from .batch_reasoning import BatchReasoningDiagnostics
from .retrieval import RetrievalDiagnostics
from .routing import RoutingDiagnostics
from .models import (
    CandidateEvaluation,
    CandidateMapping,
    FunnelEvaluation,
    GroundTruthMapping,
    MappingResult,
    Relationship,
    RelationshipEvaluation,
)


def evaluate_candidates(
    strategy: str,
    candidates: list[CandidateMapping],
    ground_truth: list[GroundTruthMapping],
    top_k: int,
) -> CandidateEvaluation:
    """Report retrieval coverage for expected MATCH fixture pairs."""
    expected = {_ground_truth_key(item) for item in ground_truth if item.expected_relationship == Relationship.MATCH}
    retrieved = expected & {_mapping_key(item) for item in candidates}
    return CandidateEvaluation(
        strategy=strategy,
        top_k=top_k,
        expected_matches=len(expected),
        retrieved_matches=len(retrieved),
        recall_at_k=len(retrieved) / len(expected) if expected else 0.0,
    )



def evaluate_funnel(
    retrieval: RetrievalDiagnostics,
    routing: RoutingDiagnostics,
    batch_plan: BatchReasoningDiagnostics,
    routed_evaluation: CandidateEvaluation,
    ground_truth: list[GroundTruthMapping],
    routed_candidates: list[CandidateMapping],
    cache_diagnostics: Mapping[str, Mapping[str, int]] | None = None,
) -> FunnelEvaluation:
    """Summarize existing funnel diagnostics without changing their evidence."""
    cache_hits = dict(cache_diagnostics["hits"]) if cache_diagnostics else None
    routed_keys = {_mapping_key(candidate) for candidate in routed_candidates}
    missing_expected_matches = [
        item for item in ground_truth
        if item.expected_relationship == Relationship.MATCH and _ground_truth_key(item) not in routed_keys
    ]
    cache_misses = dict(cache_diagnostics["misses"]) if cache_diagnostics else None
    return FunnelEvaluation(
        total_possible_pairs=retrieval.total_possible_pairs,
        pairs_retained=retrieval.pairs_retained,
        filter_reduction_rate=_reduction(retrieval.total_possible_pairs, retrieval.pairs_retained),
        shortlisted_candidates=retrieval.shortlisted_candidates,
        shortlist_reduction_rate=_reduction(retrieval.embedding_ranked_pairs, retrieval.shortlisted_candidates),
        routed_candidates=routing.reason_candidates,
        routing_reduction_rate=_reduction(routing.input_candidates, routing.reason_candidates),
        planned_batches=batch_plan.batches,
        planned_provider_requests=batch_plan.provider_requests,
        planned_request_reduction_rate=_reduction(
            retrieval.shortlisted_candidates, batch_plan.provider_requests
        ),
        routed_match_recall=routed_evaluation.recall_at_k,
        routed_missing_expected_matches=missing_expected_matches,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def _reduction(before: int, after: int) -> float:
    return (before - after) / before if before else 0.0


def evaluate_relationships(
    mappings: list[MappingResult], ground_truth: list[GroundTruthMapping]
) -> RelationshipEvaluation:
    """Evaluate final labels only against explicitly labeled fixture pairs.

    Missing predictions become NO_MATCH for a known fixture pair. Predictions not
    present in the fixture are counted separately and do not change its metrics.
    """
    expected = {_ground_truth_key(item): item.expected_relationship for item in ground_truth}
    predictions = {_mapping_key(item): item.relationship for item in mappings}
    correct = true_positive = false_positive = false_negative = missing = 0
    for pair, expected_relationship in expected.items():
        predicted = predictions.get(pair, Relationship.NO_MATCH)
        if pair not in predictions:
            missing += 1
        if predicted == expected_relationship:
            correct += 1
        if predicted == Relationship.MATCH and expected_relationship == Relationship.MATCH:
            true_positive += 1
        elif predicted == Relationship.MATCH:
            false_positive += 1
        elif expected_relationship == Relationship.MATCH:
            false_negative += 1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    non_matches = sum(value != Relationship.MATCH for value in expected.values())
    return RelationshipEvaluation(
        evaluated_pairs=len(expected),
        correct_relationships=correct,
        relationship_accuracy=correct / len(expected) if expected else 0.0,
        match_true_positives=true_positive,
        match_false_positives=false_positive,
        match_false_negatives=false_negative,
        match_precision=precision,
        match_recall=recall,
        match_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        false_positive_rate=false_positive / non_matches if non_matches else 0.0,
        missing_predictions=missing,
        unlabeled_predictions=len(set(predictions) - set(expected)),
    )


def _ground_truth_key(item: GroundTruthMapping) -> tuple[str, str, str, str]:
    return (item.source_dataset, item.source_column, item.target_dataset, item.target_column)


def _mapping_key(item: CandidateMapping | MappingResult) -> tuple[str, str, str, str]:
    return (item.source_dataset, item.source_column, item.target_dataset, item.target_column)
