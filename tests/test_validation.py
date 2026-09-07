import unittest

from ontology_mapper.models import (
    CandidateMapping, ColumnProfile, DecisionStatus, MatchMethod, ReasoningResult, Relationship,
)
from ontology_mapper.validation import finalize_mapping


def profile(name: str, dtype: str) -> ColumnProfile:
    return ColumnProfile(name, dtype, 2, 0, 0.0, 2, 1.0, ["one", "two"])


def reasoning(relationship: Relationship, confidence: float = 0.9, score: float = 0.9) -> ReasoningResult:
    return ReasoningResult(
        "source", "source_column", "target", "target_column", score, MatchMethod.EMBEDDING,
        relationship, confidence, "Provisional assessment.", [],
    )


class ValidationTests(unittest.TestCase):
    def test_currency_conflict_overrides_provisional_match(self) -> None:
        result = finalize_mapping(profile("price_usd", "float"), profile("price_eur", "float"), reasoning(Relationship.MATCH))
        self.assertEqual(result.relationship, Relationship.CONFLICT)
        self.assertEqual(result.status, DecisionStatus.REJECT)
        self.assertIn("EXPLICIT_CURRENCY_CONFLICT", [item.code for item in result.validation_findings])

    def test_identifier_type_difference_is_warning_not_conflict(self) -> None:
        result = finalize_mapping(profile("customer_id", "string"), profile("client_identifier", "integer"), reasoning(Relationship.MATCH, 1.0, 1.0))
        self.assertEqual(result.relationship, Relationship.MATCH)
        self.assertEqual(result.status, DecisionStatus.ACCEPT)
        self.assertIn("OBSERVED_TYPE_DIFFER", [item.code for item in result.validation_findings])

    def test_numeric_types_are_compatible(self) -> None:
        result = finalize_mapping(profile("annual_income", "integer"), profile("salary", "float"), reasoning(Relationship.MATCH, 1.0, 1.0))
        self.assertEqual(result.status, DecisionStatus.ACCEPT)
        self.assertIn("NUMERIC_TYPES_COMPATIBLE", [item.code for item in result.validation_findings])

    def test_related_is_review_and_no_match_conflict_are_rejected(self) -> None:
        for relationship, status in (
            (Relationship.RELATED, DecisionStatus.REVIEW),
            (Relationship.NO_MATCH, DecisionStatus.REJECT),
            (Relationship.CONFLICT, DecisionStatus.REJECT),
        ):
            result = finalize_mapping(profile("source", "string"), profile("target", "string"), reasoning(relationship, 1.0, 1.0))
            self.assertEqual(result.status, status)

    def test_match_score_boundaries_drive_status(self) -> None:
        for confidence, expected in ((0.9, DecisionStatus.ACCEPT), (0.7, DecisionStatus.REVIEW), (0.69, DecisionStatus.REJECT)):
            result = finalize_mapping(profile("source", "string"), profile("target", "string"), reasoning(Relationship.MATCH, confidence, confidence))
            self.assertEqual(result.status, expected)
