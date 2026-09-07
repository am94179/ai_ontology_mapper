import tempfile
import unittest
from pathlib import Path

from ontology_mapper.batch_reasoning import reason_candidate_batches
from ontology_mapper.cache import FileCache, file_fingerprint, fingerprint
from ontology_mapper.models import CandidateMapping, ColumnProfile, DatasetRef, MatchMethod
from ontology_mapper.pipeline_cache import PipelineCache
from ontology_mapper.retrieval import ObservedTypeCompatibility, PairEvidence, RetrievalDiagnostics


class _BatchClient:
    def __init__(self) -> None:
        self.calls = 0

    def reason_batch(self, prompt: str) -> dict:
        self.calls += 1
        return {
            "assessments": [
                {
                    "target_column": "client_id",
                    "relationship": "MATCH",
                    "confidence": 0.9,
                    "explanation": "Both are identifiers.",
                    "conflicts": [],
                }
            ]
        }


class CacheTests(unittest.TestCase):
    def test_fingerprints_use_stable_values_and_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("id\n1\n", encoding="utf-8")
            first = file_fingerprint(path)
            self.assertEqual(fingerprint({"b": 2, "a": 1}), fingerprint({"a": 1, "b": 2}))
            path.write_text("id\n2\n", encoding="utf-8")
            self.assertNotEqual(first, file_fingerprint(path))

    def test_malformed_entry_is_a_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = FileCache(directory)
            key = fingerprint({"example": 1})
            path = Path(directory) / "profiles" / f"{key}.json"
            path.parent.mkdir()
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(cache.get("profiles", key))

    def test_profile_cache_hits_and_invalidates_for_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "people.csv"
            path.write_text("id,name\n1,Ada\n", encoding="utf-8")
            pipeline = PipelineCache(FileCache(Path(directory) / "cache"))
            reference = DatasetRef.from_path(path)

            first = pipeline.profile(reference, 5)
            second = pipeline.profile(reference, 5)
            self.assertEqual(first, second)
            self.assertEqual(pipeline.diagnostics()["misses"]["profiles"], 1)
            self.assertEqual(pipeline.diagnostics()["hits"]["profiles"], 1)

            path.write_text("id,name\n1,Ada\n2,Grace\n", encoding="utf-8")
            changed = pipeline.profile(reference, 5)
            self.assertEqual(changed.row_count, 2)
            self.assertEqual(pipeline.diagnostics()["misses"]["profiles"], 2)

    def test_routing_cache_key_includes_configuration(self) -> None:
        candidate = CandidateMapping(
            "source", "customer_id", "target", "client_id", 0.9,
            MatchMethod.EMBEDDING, "customerid", "clientid",
        )
        diagnostics = RetrievalDiagnostics(
            1, 1, 1, 1, [PairEvidence(
                "customer_id", "client_id", "string", "string", False,
                ObservedTypeCompatibility.COMPATIBLE, True,
            )]
        )
        with tempfile.TemporaryDirectory() as directory:
            pipeline = PipelineCache(FileCache(directory))
            pipeline.routing([candidate], diagnostics, minimum_score=0.8, ambiguity_margin=0.03)
            pipeline.routing([candidate], diagnostics, minimum_score=0.8, ambiguity_margin=0.03)
            pipeline.routing([candidate], diagnostics, minimum_score=0.95, ambiguity_margin=0.03)
            self.assertEqual(pipeline.diagnostics()["hits"]["routing"], 1)
            self.assertEqual(pipeline.diagnostics()["misses"]["routing"], 2)

    def test_batch_reasoning_reuses_validated_response_and_context_invalidates(self) -> None:
        candidate = CandidateMapping(
            "source", "customer_id", "target", "client_id", 0.9,
            MatchMethod.EMBEDDING, "customerid", "clientid",
        )

        source = ColumnProfile("customer_id", "string", 1, 0, 0.0, 1, 1.0, ["C1"])
        target = ColumnProfile("client_id", "string", 1, 0, 0.0, 1, 1.0, ["C1"])
        with tempfile.TemporaryDirectory() as directory:
            client = _BatchClient()
            cache = FileCache(directory)
            first = reason_candidate_batches(
                {"customer_id": source}, {"client_id": target}, [candidate], client,
                cache=cache, cache_context={"model": "test-model", "base_url": "https://example.test"},
            )
            second = reason_candidate_batches(
                {"customer_id": source}, {"client_id": target}, [candidate], client,
                cache=cache, cache_context={"model": "test-model", "base_url": "https://example.test"},
            )
            changed = reason_candidate_batches(
                {"customer_id": source}, {"client_id": target}, [candidate], client,
                cache=cache, cache_context={"model": "other-model", "base_url": "https://example.test"},
            )
            self.assertEqual(client.calls, 2)
            self.assertEqual(first.diagnostics.provider_requests, 1)
            self.assertEqual(second.diagnostics.provider_requests, 0)
            self.assertEqual(second.diagnostics.cache_hits, 1)
            self.assertEqual(changed.diagnostics.cache_misses, 1)

