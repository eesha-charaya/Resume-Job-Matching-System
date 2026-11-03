from resume_job_matcher import (
    DEGREE_MASTER,
    DEGREE_UNKNOWN,
    SENIORITY_ENTRY,
    SENIORITY_UNKNOWN,
    YEARS_UNKNOWN,
    build_metadata_filter,
)


def test_build_metadata_filter_returns_none_for_unknown_profile():
    assert build_metadata_filter(DEGREE_UNKNOWN, SENIORITY_UNKNOWN, YEARS_UNKNOWN) is None


def test_build_metadata_filter_builds_combined_conditions():
    result = build_metadata_filter(
        user_degree=DEGREE_MASTER,
        user_seniority=SENIORITY_ENTRY,
        user_years=3,
    )
    assert result is not None
    assert "$and" in result
    assert len(result["$and"]) == 3


def test_experience_filter_allows_unknown_or_supported_years():
    result = build_metadata_filter(
        user_degree=DEGREE_UNKNOWN,
        user_seniority=SENIORITY_UNKNOWN,
        user_years=3,
    )
    assert result == {
        "$or": [
            {"min_years_experience": {"$lte": 3}},
            {"min_years_experience": {"$eq": YEARS_UNKNOWN}},
        ]
    }
