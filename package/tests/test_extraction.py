from resume_job_matcher import (
    DEGREE_BACHELOR,
    DEGREE_MASTER,
    SENIORITY_ENTRY,
    SENIORITY_SENIOR,
    YEARS_UNKNOWN,
    extract_degree_requirement,
    extract_job_requirements,
    extract_seniority_level,
    extract_years_experience,
)


def test_extract_degree_requirement():
    assert extract_degree_requirement("Bachelor's degree required") == DEGREE_BACHELOR
    assert extract_degree_requirement("Master's degree preferred") == DEGREE_MASTER


def test_extract_seniority_level():
    assert extract_seniority_level("Senior Data Scientist") == SENIORITY_SENIOR
    assert extract_seniority_level("New graduate role") == SENIORITY_ENTRY


def test_extract_years_experience():
    assert extract_years_experience("Requires 3+ years of experience") == 3
    assert extract_years_experience("No explicit experience requirement") == YEARS_UNKNOWN


def test_extract_job_requirements_uses_title_fallback():
    result = extract_job_requirements(
        "Bachelor's degree and at least 2 years of experience",
        title="Senior Data Scientist",
    )
    assert result == {
        "required_degree": DEGREE_BACHELOR,
        "seniority_level": SENIORITY_SENIOR,
        "min_years_experience": 2,
    }
