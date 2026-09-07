import unittest

from ontology_mapper.models import ColumnProfile, DatasetProfile, DatasetRef
from ontology_mapper.retrieval import (
    ObservedTypeCompatibility,
    generate_embedding_retrieval,
    observed_type_compatibility,
)


class FixedEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def profile(name: str, dtype: str) -> ColumnProfile:
    return ColumnProfile(name, dtype, 2, 0, 0.0, 2, 1.0, ["one", "two"])


def dataset(identifier: str, *columns: ColumnProfile) -> DatasetProfile:
    return DatasetProfile(DatasetRef(identifier, f"{identifier}.csv"), 2, list(columns))


class RetrievalTests(unittest.TestCase):
    def test_identifier_and_related_type_pairs_are_retained(self) -> None:
        self.assertEqual(
            observed_type_compatibility(profile("customer_id", "string"), profile("external_id", "integer")),
            ObservedTypeCompatibility.IDENTIFIER_COMPATIBLE,
        )
        self.assertEqual(
            observed_type_compatibility(profile("age_years", "integer"), profile("date_of_birth", "date")),
            ObservedTypeCompatibility.UNKNOWN,
        )

    def test_clear_text_numeric_and_boolean_date_pairs_are_excluded(self) -> None:
        self.assertEqual(
            observed_type_compatibility(profile("description", "string"), profile("amount", "integer")),
            ObservedTypeCompatibility.INCOMPATIBLE,
        )
        self.assertEqual(
            observed_type_compatibility(profile("enabled", "boolean"), profile("event_date", "date")),
            ObservedTypeCompatibility.INCOMPATIBLE,
        )

    def test_diagnostics_cover_all_pairs_and_shortlist_only_retained_pairs(self) -> None:
        retrieval = generate_embedding_retrieval(
            dataset(
                "source",
                profile("customer_id", "string"),
                profile("age_years", "integer"),
                profile("description", "string"),
            ),
            dataset(
                "target",
                profile("external_id", "integer"),
                profile("date_of_birth", "date"),
                profile("amount", "integer"),
            ),
            FixedEncoder(),
            top_k=3,
            minimum_score=0.0,
        )

        self.assertEqual(retrieval.diagnostics.total_possible_pairs, 9)
        self.assertEqual(retrieval.diagnostics.pairs_retained, 6)
        self.assertEqual(retrieval.diagnostics.embedding_ranked_pairs, 6)
        self.assertEqual(retrieval.diagnostics.shortlisted_candidates, len(retrieval.candidates))
        self.assertEqual(len(retrieval.diagnostics.pair_evidence), 9)
        self.assertTrue(
            next(
                evidence
                for evidence in retrieval.diagnostics.pair_evidence
                if evidence.source_column == "customer_id" and evidence.target_column == "external_id"
            ).retained
        )
        self.assertFalse(
            next(
                evidence
                for evidence in retrieval.diagnostics.pair_evidence
                if evidence.source_column == "description" and evidence.target_column == "amount"
            ).retained
        )
