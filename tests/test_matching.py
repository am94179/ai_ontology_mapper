import unittest

from ontology_mapper.matching import generate_candidates, normalize_column_name
from ontology_mapper.models import ColumnProfile, DatasetProfile, DatasetRef, MatchMethod


def dataset(identifier: str, *column_names: str) -> DatasetProfile:
    columns = [
        ColumnProfile(name, "string", 1, 0, 0.0, 1, 1.0, ["value"])
        for name in column_names
    ]
    return DatasetProfile(DatasetRef(identifier, f"{identifier}.csv"), 1, columns)


class CandidateMatchingTests(unittest.TestCase):
    def test_normalization_handles_case_and_separator_conventions(self) -> None:
        self.assertEqual(normalize_column_name("email_address"), "emailaddress")
        self.assertEqual(normalize_column_name("emailAddress"), "emailaddress")

    def test_normalized_equal_names_are_exact_candidates(self) -> None:
        candidates = generate_candidates(
            dataset("source", "email_address"), dataset("target", "emailAddress")
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].score, 1.0)
        self.assertEqual(candidates[0].method, MatchMethod.EXACT_NORMALIZED)

    def test_fuzzy_names_rank_ahead_of_unrelated_names(self) -> None:
        candidates = generate_candidates(
            dataset("source", "first_name"),
            dataset("target", "fname", "postal_code"),
            top_k=1,
            minimum_score=0.0,
        )

        self.assertEqual(candidates[0].target_column, "fname")
        self.assertEqual(candidates[0].method, MatchMethod.FUZZY_NAME)

    def test_low_scoring_pairs_are_not_candidates(self) -> None:
        candidates = generate_candidates(
            dataset("source", "revenue"), dataset("target", "postal_code"), minimum_score=0.8
        )

        self.assertEqual(candidates, [])

    def test_top_k_is_limited_per_source_column(self) -> None:
        candidates = generate_candidates(
            dataset("source", "customer_id"),
            dataset("target", "customer_identifier", "client_identifier"),
            top_k=1,
            minimum_score=0.0,
        )

        self.assertEqual(len(candidates), 1)
