import unittest

from ontology_mapper.models import CandidateMapping, MatchMethod
from ontology_mapper.reasoning import reason_candidates
from ontology_mapper.retrieval import (
    ObservedTypeCompatibility,
    PairEvidence,
    RetrievalDiagnostics,
)
from ontology_mapper.routing import RoutingAction, route_embedding_candidates


def candidate(source: str, target: str, score: float) -> CandidateMapping:
    return CandidateMapping("source", source, "target", target, score, MatchMethod.EMBEDDING, source, target)


def diagnostics(*candidates: CandidateMapping) -> RetrievalDiagnostics:
    evidence = [
        PairEvidence(
            candidate.source_column,
            candidate.target_column,
            "string",
            "string",
            candidate.source_normalized == candidate.target_normalized,
            ObservedTypeCompatibility.COMPATIBLE,
            True,
        )
        for candidate in candidates
    ]
    return RetrievalDiagnostics(len(evidence), len(evidence), len(evidence), len(candidates), evidence)


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def reason(self, prompt: str) -> dict[str, object]:
        self.calls += 1
        return {"relationship": "MATCH", "confidence": 0.9, "explanation": "test", "conflicts": []}


class RoutingTests(unittest.TestCase):
    def test_top_candidate_routes_when_it_meets_the_minimum_score(self) -> None:
        candidates = [candidate("income", "salary", 0.90), candidate("income", "postal_code", 0.70)]
        routing = route_embedding_candidates(candidates, diagnostics(*candidates), minimum_score=0.80, ambiguity_margin=0.03)

        self.assertEqual([item.action for item in routing.decisions], [RoutingAction.REASON, RoutingAction.NO_DECISION])
        self.assertEqual(routing.diagnostics.projected_llm_call_reduction, 1)

    def test_close_runner_up_is_also_routed(self) -> None:
        candidates = [candidate("name", "first_name", 0.90), candidate("name", "given_name", 0.88)]
        routing = route_embedding_candidates(candidates, diagnostics(*candidates), minimum_score=0.80, ambiguity_margin=0.03)

        self.assertEqual([item.action for item in routing.decisions], [RoutingAction.REASON, RoutingAction.REASON])
        self.assertEqual(routing.decisions[1].rank, 2)

    def test_low_score_candidates_have_no_decision_and_do_not_invoke_reasoning(self) -> None:
        candidates = [candidate("postal_code", "revenue", 0.50), candidate("postal_code", "amount", 0.45)]
        routing = route_embedding_candidates(candidates, diagnostics(*candidates), minimum_score=0.80)
        client = FakeClient()

        results = reason_candidates({}, {}, routing.reasoning_candidates, client)

        self.assertEqual([item.action for item in routing.decisions], [RoutingAction.NO_DECISION, RoutingAction.NO_DECISION])
        self.assertEqual(results, [])
        self.assertEqual(client.calls, 0)
