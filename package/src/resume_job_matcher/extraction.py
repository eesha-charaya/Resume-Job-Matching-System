"""Regex-based extraction of structured job requirements."""

import re
from typing import TypedDict

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

_DEGREE_PHD_RE = re.compile(
    r"\b(ph\.d\.?|phd|doctorate|doctoral\s+degree)\b", re.IGNORECASE
)
_DEGREE_MASTER_RE = re.compile(
    r"(?:master(?:'?s)?(?:\s+degree)?|m\.s\.(?:\s|$)|m\.eng\.(?:\s|$)|mba)(?:\s|$|\b)",
    re.IGNORECASE,
)
_DEGREE_BACHELOR_RE = re.compile(
    r"(?:bachelor(?:'?s)?(?:\s+degree)?|b\.s\.(?:\s|$)|b\.a\.(?:\s|$)|"
    r"undergraduate(?:\s+degree)?)(?:\s|$|\b)",
    re.IGNORECASE,
)

_SENIORITY_SENIOR_RE = re.compile(
    r"\b(senior|sr\.\s|lead|principal|director|manager)\b", re.IGNORECASE
)
_SENIORITY_MID_RE = re.compile(r"\b(mid-level|mid\s+level|intermediate)\b", re.IGNORECASE)
_SENIORITY_ENTRY_RE = re.compile(
    r"\b(entry-level|entry\s+level|junior|new\s+grad(?:uate)?)\b", re.IGNORECASE
)

_YEARS_PATTERNS = (
    re.compile(
        r"(?<!\d)(\d{1,2})\+\s*years?\b(?:\s+(?:of\s+)?experience)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:at\s+least|minimum\s+of|minimum)\s+(?<!\d)(\d{1,2})\s+years?\b"
        r"(?:\s+(?:of\s+)?experience)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\d)(\d{1,2})\s+or\s+more\s+years?\b(?:\s+(?:of\s+)?experience)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\d)(\d{1,2})\s*[-–]\s*\d{1,2}\s+years?\b"
        r"(?:\s+(?:of\s+)?(?:in\s+)?(?:[\w\s]+?\s+)?experience)?",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\d)(\d{1,2})\s+years?\s+of\s+[\w\s]+?experience\b", re.IGNORECASE),
)


class JobRequirements(TypedDict):
    """Structured requirements extracted from a job description."""

    required_degree: int
    seniority_level: int
    min_years_experience: int


def extract_degree_requirement(text: str) -> int:
    """Extract the highest degree requirement from text."""
    if not text:
        return DEGREE_UNKNOWN
    if _DEGREE_PHD_RE.search(text):
        return DEGREE_PHD
    if _DEGREE_MASTER_RE.search(text):
        return DEGREE_MASTER
    if _DEGREE_BACHELOR_RE.search(text):
        return DEGREE_BACHELOR
    return DEGREE_UNKNOWN


def extract_seniority_level(text: str) -> int:
    """Extract seniority level from free text."""
    if not text:
        return SENIORITY_UNKNOWN
    if _SENIORITY_SENIOR_RE.search(text):
        return SENIORITY_SENIOR
    if _SENIORITY_MID_RE.search(text):
        return SENIORITY_MID
    if _SENIORITY_ENTRY_RE.search(text):
        return SENIORITY_ENTRY
    return SENIORITY_UNKNOWN


def extract_years_experience(text: str) -> int:
    """Extract the minimum years of experience stated in text."""
    if not text:
        return YEARS_UNKNOWN

    found: list[int] = []
    for pattern in _YEARS_PATTERNS:
        found.extend(int(match.group(1)) for match in pattern.finditer(text))
    return min(found) if found else YEARS_UNKNOWN


def extract_job_requirements(text: str, title: str = "") -> JobRequirements:
    """Extract degree, seniority, and experience requirements for a job."""
    seniority = extract_seniority_level(text)
    if seniority == SENIORITY_UNKNOWN and title:
        seniority = extract_seniority_level(title)

    return JobRequirements(
        required_degree=extract_degree_requirement(text),
        seniority_level=seniority,
        min_years_experience=extract_years_experience(text),
    )
