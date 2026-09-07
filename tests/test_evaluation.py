import unittest

from ontology_mapper.evaluation import candidate_recall_at_k
from ontology_mapper.models import CandidateMapping, GroundTruthMapping, MatchMethod, Relationship


class CandidateEvaluationTests(unittest.TestCase):
    def test_recall_only_uses_expected_matches(self) -> None:
        candidates = [
            CandidateMapping(
                "a", "email_address", "b", "emailAddress", 1.0,
                MatchMethod.EXACT_NORMALIZED, "emailaddress", "emailaddress"
            )
        ]
        truth = [
            GroundTruthMapping("a", "email_address", "b", "emailAddress", Relationship.MATCH),
            GroundTruthMapping("a", "postal_code", "b", "id", Relationship.NO_MATCH),
        ]

        self.assertEqual(candidate_recall_at_k(candidates, truth), 1.0)
