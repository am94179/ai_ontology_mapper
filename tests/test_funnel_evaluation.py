import unittest

from ontology_mapper.batch_reasoning import BatchReasoningDiagnostics
from ontology_mapper.metrics import evaluate_funnel
from ontology_mapper.models import CandidateEvaluation, GroundTruthMapping, Relationship
from ontology_mapper.retrieval import RetrievalDiagnostics
from ontology_mapper.routing import RoutingDiagnostics


class FunnelEvaluationTests(unittest.TestCase):
    def test_funnel_summary_reports_stage_reductions_and_cache_counts(self) -> None:
        result = evaluate_funnel(
            RetrievalDiagnostics(100, 60, 60, 6, []),
            RoutingDiagnostics(2, 6, 3, 3, 0, 3),
            BatchReasoningDiagnostics(3, 2, 2),
            CandidateEvaluation("routed_embedding", 3, 5, 4, 0.8),
            [GroundTruthMapping("source", "missing", "target", "missing_target", Relationship.MATCH)],
            [],
            {"hits": {"profiles": 2}, "misses": {"routing": 1}},
        )

        self.assertEqual((result.total_possible_pairs, result.pairs_retained), (100, 60))
        self.assertEqual(result.shortlisted_candidates, 6)
        self.assertEqual((result.routed_candidates, result.planned_provider_requests), (3, 2))
        self.assertAlmostEqual(result.filter_reduction_rate, 0.4)
        self.assertAlmostEqual(result.shortlist_reduction_rate, 0.9)
        self.assertAlmostEqual(result.routing_reduction_rate, 0.5)
        self.assertAlmostEqual(result.planned_request_reduction_rate, 2 / 3)
        self.assertEqual(result.routed_match_recall, 0.8)
        self.assertEqual(result.routed_missing_expected_matches[0].source_column, "missing")
        self.assertEqual(result.cache_hits, {"profiles": 2})
        self.assertEqual(result.cache_misses, {"routing": 1})

    def test_empty_funnel_has_stable_zero_reductions_and_no_cache_data(self) -> None:
        result = evaluate_funnel(
            RetrievalDiagnostics(0, 0, 0, 0, []),
            RoutingDiagnostics(0, 0, 0, 0, 0, 0),
            BatchReasoningDiagnostics(0, 0, 0),
            CandidateEvaluation("routed_embedding", 3, 0, 0, 0.0),
            [],
            [],
        )

        self.assertEqual(result.filter_reduction_rate, 0.0)
        self.assertEqual(result.shortlist_reduction_rate, 0.0)
        self.assertEqual(result.routing_reduction_rate, 0.0)
        self.assertEqual(result.planned_request_reduction_rate, 0.0)
        self.assertIsNone(result.cache_hits)
        self.assertIsNone(result.cache_misses)

