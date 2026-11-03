"""Reusable utilities for the Resume-Job Matching System."""

from .constants import (
    DEGREE_BACHELOR,
    DEGREE_MASTER,
    DEGREE_PHD,
    DEGREE_UNKNOWN,
    SENIORITY_ENTRY,
    SENIORITY_MID,
    SENIORITY_SENIOR,
    SENIORITY_UNKNOWN,
    YEARS_UNKNOWN,
)
from .extraction import (
    JobRequirements,
    extract_degree_requirement,
    extract_job_requirements,
    extract_seniority_level,
    extract_years_experience,
)
from .filtering import build_metadata_filter
from .metrics import compute_metrics_at_k, precision_at_k, recall_at_k
from .preprocessing import preprocess_description

__all__ = [
    "DEGREE_BACHELOR",
    "DEGREE_MASTER",
    "DEGREE_PHD",
    "DEGREE_UNKNOWN",
    "JobRequirements",
    "SENIORITY_ENTRY",
    "SENIORITY_MID",
    "SENIORITY_SENIOR",
    "SENIORITY_UNKNOWN",
    "YEARS_UNKNOWN",
    "build_metadata_filter",
    "compute_metrics_at_k",
    "extract_degree_requirement",
    "extract_job_requirements",
    "extract_seniority_level",
    "extract_years_experience",
    "precision_at_k",
    "preprocess_description",
    "recall_at_k",
]

__version__ = "0.2.0"
