import unittest

from ontology_mapper.metrics import evaluate_candidates, evaluate_relationships
from ontology_mapper.models import (
    CandidateEvaluation,
    CandidateMapping,
    DecisionStatus,
    GroundTruthMapping,
    MappingResult,
    MatchMethod,
    Relationship,
)


def final_mapping(source: str, target: str, relationship: Relationship) -> MappingResult:
    return MappingResult(
        "a", source, "b", target, 0.9, MatchMethod.EMBEDDING, relationship, 0.9,
        "evaluation fixture", [], [], relationship, 0.9, DecisionStatus.ACCEPT, "fixture"
    )


class EvaluationMetricTests(unittest.TestCase):
    def test_candidate_evaluation_reports_strategy_and_counts(self) -> None:
        candidates = [
            CandidateMapping("a", "one", "b", "uno", 1.0, MatchMethod.EMBEDDING, "one", "uno")
        ]
        truth = [
            GroundTruthMapping("a", "one", "b", "uno", Relationship.MATCH),
            GroundTruthMapping("a", "two", "b", "dos", Relationship.MATCH),
        ]

        result = evaluate_candidates("embedding", candidates, truth, top_k=3)

        self.assertIsInstance(result, CandidateEvaluation)
        self.assertEqual((result.expected_matches, result.retrieved_matches), (2, 1))
        self.assertEqual(result.recall_at_k, 0.5)

    def test_relationship_metrics_handle_labels_missing_and_unlabeled_predictions(self) -> None:
        truth = [
            GroundTruthMapping("a", "match", "b", "match", Relationship.MATCH),
            GroundTruthMapping("a", "related", "b", "related", Relationship.RELATED),
            GroundTruthMapping("a", "conflict", "b", "conflict", Relationship.CONFLICT),
            GroundTruthMapping("a", "none", "b", "none", Relationship.NO_MATCH),
        ]
        mappings = [
            final_mapping("match", "match", Relationship.MATCH),
            final_mapping("related", "related", Relationship.MATCH),
            final_mapping("conflict", "conflict", Relationship.CONFLICT),
            final_mapping("extra", "extra", Relationship.MATCH),
        ]

        result = evaluate_relationships(mappings, truth)

        self.assertEqual((result.evaluated_pairs, result.correct_relationships), (4, 3))
        self.assertEqual((result.match_true_positives, result.match_false_positives), (1, 1))
        self.assertEqual(result.match_false_negatives, 0)
        self.assertEqual(result.match_precision, 0.5)
        self.assertEqual(result.match_recall, 1.0)
        self.assertEqual(result.false_positive_rate, 1 / 3)
        self.assertEqual((result.missing_predictions, result.unlabeled_predictions), (1, 1))

    def test_empty_fixture_returns_stable_zero_metrics(self) -> None:
        result = evaluate_relationships([], [])

        self.assertEqual(result.relationship_accuracy, 0.0)
        self.assertEqual(result.match_f1, 0.0)
        self.assertEqual(result.false_positive_rate, 0.0)
