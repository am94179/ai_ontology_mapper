import json
from dataclasses import asdict
import unittest

from ontology_mapper.models import (
    CandidateMapping,
    DecisionStatus,
    MappingResult,
    MatchMethod,
    Relationship,
)
from ontology_mapper.ontology import build_canonical_schema


def mapping(
    source: str,
    target: str,
    relationship: Relationship = Relationship.MATCH,
    status: DecisionStatus = DecisionStatus.ACCEPT,
) -> MappingResult:
    return MappingResult(
        "customers_a", source, "clients_b", target, 0.95, MatchMethod.EMBEDDING,
        relationship, 0.95, "Finalized mapping.", [], [], relationship, 0.95, status, "Finalized."
    )


class CanonicalSchemaTests(unittest.TestCase):
    def test_accepted_match_creates_a_source_named_concept(self) -> None:
        schema = build_canonical_schema([mapping("customer_id", "client_identifier")])

        self.assertEqual(len(schema.concepts), 1)
        concept = schema.concepts[0]
        self.assertEqual(concept.identifier, "customers_a.customer_id")
        self.assertEqual(concept.canonical_name, "customer_id")
        self.assertEqual([(field.dataset, field.column) for field in concept.fields], [
            ("customers_a", "customer_id"), ("clients_b", "client_identifier"),
        ])

    def test_review_and_rejected_mappings_are_not_canonical_members(self) -> None:
        schema = build_canonical_schema([
            mapping("age_years", "date_of_birth", Relationship.RELATED, DecisionStatus.REVIEW),
            mapping("price_usd", "price_eur", Relationship.CONFLICT, DecisionStatus.REJECT),
        ])

        self.assertEqual(schema.concepts, [])
        self.assertEqual(len(schema.review_mappings), 1)
        self.assertEqual(len(schema.rejected_mappings), 1)

    def test_ambiguous_accepted_mappings_are_excluded(self) -> None:
        schema = build_canonical_schema([
            mapping("customer_id", "client_identifier"),
            mapping("customer_id", "external_identifier"),
        ])

        self.assertEqual(schema.concepts, [])
        self.assertEqual([item.target_column for item in schema.ambiguous_mappings], [
            "client_identifier", "external_identifier",
        ])

    def test_output_order_is_deterministic(self) -> None:
        schema = build_canonical_schema([
            mapping("first_name", "fname"),
            mapping("customer_id", "client_identifier"),
        ])

        self.assertEqual([item.canonical_name for item in schema.concepts], ["customer_id", "first_name"])

    def test_target_ambiguity_is_excluded_and_output_is_json_serializable(self) -> None:
        schema = build_canonical_schema([
            mapping("customer_id", "identifier"),
            mapping("external_id", "identifier"),
        ])

        self.assertEqual(schema.concepts, [])
        self.assertEqual(len(schema.ambiguous_mappings), 2)
        json.dumps(asdict(schema), default=str)
