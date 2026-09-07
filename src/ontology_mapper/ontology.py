"""Small, deterministic canonical-schema construction from final mappings."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .models import (
    CanonicalConcept,
    CanonicalField,
    CanonicalSchema,
    DecisionStatus,
    MappingResult,
    Relationship,
)


def build_canonical_schema(mappings: Sequence[MappingResult]) -> CanonicalSchema:
    """Build concepts from unambiguous accepted matches and report everything else.

    This function deliberately does not resolve global mapping ambiguity. A field
    participating in multiple accepted matches is returned for review instead of
    being automatically merged into an unreliable canonical concept.
    """
    ordered = sorted(mappings, key=_mapping_key)
    accepted_matches = [
        item
        for item in ordered
        if item.status == DecisionStatus.ACCEPT and item.relationship == Relationship.MATCH
    ]
    source_counts = Counter((item.source_dataset, item.source_column) for item in accepted_matches)
    target_counts = Counter((item.target_dataset, item.target_column) for item in accepted_matches)
    ambiguous = [
        item
        for item in accepted_matches
        if source_counts[(item.source_dataset, item.source_column)] > 1
        or target_counts[(item.target_dataset, item.target_column)] > 1
    ]
    ambiguous_keys = {_mapping_key(item) for item in ambiguous}
    concepts = [
        _concept_from_mapping(item)
        for item in accepted_matches
        if _mapping_key(item) not in ambiguous_keys
    ]
    return CanonicalSchema(
        concepts=concepts,
        review_mappings=[item for item in ordered if item.status == DecisionStatus.REVIEW],
        rejected_mappings=[item for item in ordered if item.status == DecisionStatus.REJECT],
        ambiguous_mappings=ambiguous,
    )


def _concept_from_mapping(mapping: MappingResult) -> CanonicalConcept:
    source_field = CanonicalField(mapping.source_dataset, mapping.source_column)
    target_field = CanonicalField(mapping.target_dataset, mapping.target_column)
    return CanonicalConcept(
        identifier=f"{mapping.source_dataset}.{mapping.source_column}",
        canonical_name=mapping.source_column,
        fields=[source_field, target_field],
        mapping_results=[mapping],
    )


def _mapping_key(mapping: MappingResult) -> tuple[str, str, str, str]:
    return (
        mapping.source_dataset,
        mapping.source_column,
        mapping.target_dataset,
        mapping.target_column,
    )
