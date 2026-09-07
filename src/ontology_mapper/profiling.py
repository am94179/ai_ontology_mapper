"""Deterministic profiling of CSV values into observed column metadata."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from statistics import mean, median

from .loaders import load_csv
from .models import ColumnProfile, DatasetProfile, DatasetRef

NULL_VALUES = {"", "null", "none", "na", "n/a"}


def profile_dataset(dataset: DatasetRef, sample_size: int = 5) -> DatasetProfile:
    """Profile one CSV dataset without making semantic claims about its fields."""
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")

    headers, rows = load_csv(dataset)
    columns = [
        profile_column(header, [row.get(header, "") for row in rows], sample_size)
        for header in headers
    ]
    return DatasetProfile(dataset=dataset, row_count=len(rows), columns=columns)


def profile_column(name: str, values: list[str], sample_size: int = 5) -> ColumnProfile:
    """Compute deterministic statistics for one sequence of CSV values."""
    non_null = [value.strip() for value in values if value.strip().lower() not in NULL_VALUES]
    row_count = len(values)
    distinct_count = len(set(non_null))
    dtype = _observed_dtype(non_null)
    samples = list(dict.fromkeys(non_null))[:sample_size]
    profile = ColumnProfile(
        name=name,
        observed_dtype=dtype,
        row_count=row_count,
        null_count=row_count - len(non_null),
        null_rate=_rate(row_count - len(non_null), row_count),
        distinct_count=distinct_count,
        unique_rate=_rate(distinct_count, len(non_null)),
        sample_values=samples,
    )
    if dtype in {"integer", "float"}:
        numbers = [float(value) for value in non_null]
        return replace(
            profile,

            minimum=_clean_number(min(numbers)),
            maximum=_clean_number(max(numbers)),
            mean=mean(numbers),
            median=median(numbers),
        )
    if non_null:
        lengths = [len(value) for value in non_null]
        return replace(
            profile,

            min_length=min(lengths),
            max_length=max(lengths),
            mean_length=mean(lengths),
        )
    return profile


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _clean_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _observed_dtype(values: list[str]) -> str:
    if not values:
        return "empty"
    lowered = [value.lower() for value in values]
    if all(value in {"true", "false"} for value in lowered):
        return "boolean"
    if all(_is_integer(value) for value in values):
        return "integer"
    if all(_is_float(value) for value in values):
        return "float"
    if all(_is_iso_date(value) for value in values):
        return "date"
    return "string"


def _is_integer(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False

