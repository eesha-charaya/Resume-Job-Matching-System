"""Metadata-filter helpers for resume-job retrieval."""

from typing import Any, Optional

from .constants import (
    DEGREE_UNKNOWN,
    SENIORITY_UNKNOWN,
    YEARS_UNKNOWN,
)

MetadataFilter = dict[str, Any]


def build_metadata_filter(
    user_degree: int,
    user_seniority: int,
    user_years: int,
) -> Optional[MetadataFilter]:
    """Build a Chroma-compatible metadata filter from candidate attributes."""
    conditions: list[MetadataFilter] = []

    if user_seniority != SENIORITY_UNKNOWN:
        conditions.append(
            {
                "$or": [
                    {"seniority_level": {"$eq": user_seniority}},
                    {"seniority_level": {"$eq": SENIORITY_UNKNOWN}},
                ]
            }
        )

    if user_degree != DEGREE_UNKNOWN:
        conditions.append(
            {
                "$or": [
                    {"required_degree": {"$lte": user_degree}},
                    {"required_degree": {"$eq": DEGREE_UNKNOWN}},
                ]
            }
        )

    if user_years != YEARS_UNKNOWN:
        conditions.append(
            {
                "$or": [
                    {"min_years_experience": {"$lte": user_years}},
                    {"min_years_experience": {"$eq": YEARS_UNKNOWN}},
                ]
            }
        )

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
