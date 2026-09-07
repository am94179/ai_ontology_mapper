import unittest

from ontology_mapper.batch_reasoning import reason_candidate_batches
from ontology_mapper.models import CandidateMapping, ColumnProfile, MatchMethod
from ontology_mapper.reasoning import ReasoningError


def profile(name: str) -> ColumnProfile:
    return ColumnProfile(name, "string", 1, 0, 0.0, 1, 1.0, ["sample"])


def candidate() -> CandidateMapping:
    return CandidateMapping("source", "income", "target", "salary", 0.9, MatchMethod.EMBEDDING, "income", "salary")


class StaticBatchClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def reason_batch(self, prompt: str) -> dict[str, object]:
        return self.payload


class BatchReasoningValidationTests(unittest.TestCase):
    def test_malformed_assessment_fields_are_rejected(self) -> None:
        payload = {"assessments": [{"target_column": "salary", "relationship": "MATCH"}]}

        with self.assertRaises(ReasoningError):
            reason_candidate_batches({"income": profile("income")}, {"salary": profile("salary")}, [candidate()], StaticBatchClient(payload))

    def test_out_of_range_confidence_is_rejected(self) -> None:
        payload = {"assessments": [{
            "target_column": "salary", "relationship": "MATCH", "confidence": 1.1,
            "explanation": "invalid", "conflicts": [],
        }]}

        with self.assertRaises(ReasoningError):
            reason_candidate_batches({"income": profile("income")}, {"salary": profile("salary")}, [candidate()], StaticBatchClient(payload))
