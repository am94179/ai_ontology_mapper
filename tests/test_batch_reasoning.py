import json
import unittest

from ontology_mapper.batch_reasoning import (
    build_batch_reasoning_prompt,
    group_candidate_batches,
    reason_candidate_batches,
)
from ontology_mapper.models import CandidateMapping, ColumnProfile, MatchMethod
from ontology_mapper.reasoning import ReasoningError


def profile(name: str) -> ColumnProfile:
    return ColumnProfile(name, "string", 2, 0, 0.0, 2, 1.0, ["one", "two"])


def candidate(source: str, target: str, score: float) -> CandidateMapping:
    return CandidateMapping("source", source, "target", target, score, MatchMethod.EMBEDDING, source, target)


class FakeBatchClient:
    def __init__(self, mode: str = "valid") -> None:
        self.calls: list[list[str]] = []
        self.mode = mode

    def reason_batch(self, prompt: str) -> dict[str, object]:
        evidence = json.loads(prompt.split("Evidence:\n", 1)[1])
        targets = [item["candidate"]["target_column"] for item in evidence["candidates"]]
        self.calls.append(targets)
        assessments = [
            {"target_column": target, "relationship": "MATCH", "confidence": 0.9, "explanation": "test", "conflicts": []}
            for target in targets
        ]
        if self.mode == "missing":
            assessments.pop()
        if self.mode == "duplicate":
            assessments.append(assessments[0])
        if self.mode == "unexpected":
            assessments[0]["target_column"] = "unknown"
        return {"assessments": assessments}


class BatchReasoningTests(unittest.TestCase):
    def test_batches_group_and_chunk_by_source_deterministically(self) -> None:
        candidates = [
            candidate("zeta", "z_target", 0.9),
            candidate("alpha", "target_b", 0.8),
            candidate("alpha", "target_a", 0.9),
            candidate("alpha", "target_c", 0.7),
        ]

        batches = group_candidate_batches(candidates, maximum_batch_size=2)

        self.assertEqual([[item.target_column for item in batch] for batch in batches], [["target_a", "target_b"], ["target_c"], ["z_target"]])

    def test_prompt_includes_one_source_and_only_requested_targets(self) -> None:
        source = profile("income")
        targets = {"salary": profile("salary"), "wages": profile("wages")}
        prompt = build_batch_reasoning_prompt(source, targets, [candidate("income", "salary", 0.9), candidate("income", "wages", 0.88)])

        self.assertIn('"source_column_profile"', prompt)
        self.assertIn('"salary"', prompt)
        self.assertIn('"wages"', prompt)
        self.assertNotIn("postal_code", prompt)

    def test_valid_batch_results_preserve_candidate_order_and_reduce_calls(self) -> None:
        candidates = [candidate("income", "salary", 0.9), candidate("income", "wages", 0.88), candidate("name", "first_name", 0.9)]
        client = FakeBatchClient()
        run = reason_candidate_batches(
            {"income": profile("income"), "name": profile("name")},
            {"salary": profile("salary"), "wages": profile("wages"), "first_name": profile("first_name")},
            candidates,
            client,
            maximum_batch_size=2,
        )

        self.assertEqual([item.target_column for item in run.results], ["salary", "wages", "first_name"])
        self.assertEqual(run.diagnostics.provider_requests, 2)
        self.assertEqual(client.calls, [["salary", "wages"], ["first_name"]])

    def test_incomplete_duplicate_and_unexpected_responses_are_rejected(self) -> None:
        candidates = [candidate("income", "salary", 0.9), candidate("income", "wages", 0.88)]
        source_columns = {"income": profile("income")}
        target_columns = {"salary": profile("salary"), "wages": profile("wages")}
        for mode in ("missing", "duplicate", "unexpected"):
            with self.subTest(mode=mode), self.assertRaises(ReasoningError):
                reason_candidate_batches(source_columns, target_columns, candidates, FakeBatchClient(mode), maximum_batch_size=2)
