import unittest

from ontology_mapper.embeddings import (
    column_representation,
    cosine_similarity,
    generate_embedding_candidates,
)
from ontology_mapper.models import ColumnProfile, DatasetProfile, DatasetRef, MatchMethod


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "annual income": [1.0, 0.0],
            "salary": [0.9, 0.1],
            "postal code": [0.0, 1.0],
        }
        return [next(vector for key, vector in vectors.items() if key in text) for text in texts]


def profile(name: str, dtype: str = "string") -> ColumnProfile:
    return ColumnProfile(name, dtype, 2, 0, 0.0, 2, 1.0, ["one", "two"])


def dataset(identifier: str, *columns: ColumnProfile) -> DatasetProfile:
    return DatasetProfile(DatasetRef(identifier, f"{identifier}.csv"), 2, list(columns))


class EmbeddingCandidateTests(unittest.TestCase):
    def test_representation_includes_name_type_and_samples(self) -> None:
        representation = column_representation(profile("annual_income", "integer"))

        self.assertIn("Column name: annual income.", representation)
        self.assertIn("Observed data type: integer.", representation)
        self.assertIn("Sample values: one, two.", representation)

    def test_embedding_ranking_uses_highest_cosine_similarity(self) -> None:
        candidates = generate_embedding_candidates(
            dataset("source", profile("annual_income")),
            dataset("target", profile("postal_code"), profile("salary")),
            FakeEncoder(),
            top_k=1,
        )

        self.assertEqual(candidates[0].target_column, "salary")
        self.assertEqual(candidates[0].method, MatchMethod.EMBEDDING)

    def test_embedding_score_threshold_and_vector_validation(self) -> None:
        self.assertEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 0.0])
