import unittest

from ontology_mapper.models import CandidateMapping, MatchMethod
from ontology_mapper.retrieval import (
    ObservedTypeCompatibility,
    PairEvidence,
    RetrievalDiagnostics,
)
from ontology_mapper.routing import route_embedding_candidates


def candidate(source: str, target: str, score: float) -> CandidateMapping:
    return CandidateMapping("source", source, "target", target, score, MatchMethod.EMBEDDING, source, target)


def diagnostics(*candidates: CandidateMapping) -> RetrievalDiagnostics:
    evidence = [
        PairEvidence(
            item.source_column,
            item.target_column,
            "string",
            "string",
            False,
            ObservedTypeCompatibility.COMPATIBLE,
            True,
        )
        for item in candidates
    ]
    return RetrievalDiagnostics(len(evidence), len(evidence), len(evidence), len(candidates), evidence)


class RoutingConfigurationTests(unittest.TestCase):
    def test_decisions_have_stable_source_order(self) -> None:
        candidates = [candidate("zeta", "target_z", 0.9), candidate("alpha", "target_a", 0.9)]

        routing = route_embedding_candidates(candidates, diagnostics(*candidates))

        self.assertEqual([item.candidate.source_column for item in routing.decisions], ["alpha", "zeta"])

    def test_invalid_thresholds_are_rejected(self) -> None:
        candidates = [candidate("alpha", "target_a", 0.9)]
        evidence = diagnostics(*candidates)

        with self.assertRaises(ValueError):
            route_embedding_candidates(candidates, evidence, minimum_score=1.1)
        with self.assertRaises(ValueError):
            route_embedding_candidates(candidates, evidence, ambiguity_margin=-0.1)
