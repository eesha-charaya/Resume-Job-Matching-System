"""
Validation for synthetic positive job skeletons.

Rules that can be checked exactly are validated deterministically. Ollama is
used only for genuinely semantic checks. This avoids rejecting valid values
because a small LLM misread an enum, numeric range, or skill count.
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

from eval.eval_utils import call_ollama_validate
from eval.types import JobSkeleton
from src.config import OLLAMA_MODEL

logger = logging.getLogger(__name__)


class ValidationResult(TypedDict):
    passed: bool
    reason: str | None


class ResumeInfo(TypedDict):
    seniority: str
    years_experience: int
    primary_skills: list[str]
    domain: str
    resume_text: str


_VALID_SENIORITIES = {"Junior", "Mid", "Senior", "Staff"}
_VALID_DOMAINS = {"backend", "frontend", "fullstack", "data"}

_SENIORITY_ALIASES: dict[str, str] = {
    "junior": "Junior",
    "mid": "Mid",
    "mid-level": "Mid",
    "midlevel": "Mid",
    "mid level": "Mid",
    "senior": "Senior",
    "staff": "Staff",
}

_DOMAIN_ALIASES: dict[str, str] = {
    "backend": "backend",
    "frontend": "frontend",
    "front-end": "frontend",
    "front end": "frontend",
    "fullstack": "fullstack",
    "full-stack": "fullstack",
    "full stack": "fullstack",
    "data": "data",
    "data science": "data",
    "data engineering": "data",
}


def _parse_year_parts(years_str: str) -> list[int]:
    text = str(years_str).strip()
    if not text:
        return []
    if not re.fullmatch(r"\d+\s*(?:-\s*\d+)?", text):
        return []
    return [int(x.strip()) for x in text.split("-")]


def _parse_years_required(years_str: str) -> int:
    parts = _parse_year_parts(years_str)
    return max(parts) if parts else 0


def _parse_years_min(years_str: str) -> int:
    parts = _parse_year_parts(years_str)
    return min(parts) if parts else 0


def _parse_validation_response(raw_response: str) -> ValidationResult:
    stripped = raw_response.strip()
    if stripped.upper().startswith("PASS"):
        return {"passed": True, "reason": None}
    if stripped.upper().startswith("FAIL:"):
        return {"passed": False, "reason": stripped[5:].strip()}
    return {"passed": False, "reason": f"Unparseable LLM response: {raw_response}"}


def _normalize_skeleton(job: JobSkeleton) -> JobSkeleton:
    normalized = dict(job)
    seniority_raw = str(job.get("seniority", "")).strip()
    domain_raw = str(job.get("domain", "")).strip()
    normalized["seniority"] = _SENIORITY_ALIASES.get(
        seniority_raw.lower(), seniority_raw.title()
    )
    normalized["domain"] = _DOMAIN_ALIASES.get(domain_raw.lower(), domain_raw.lower())
    return normalized  # type: ignore[return-value]


def _normalize_skill(skill: str) -> str:
    return re.sub(r"[^a-z0-9+#.]", "", skill.lower())


def _build_structural_prompt(job_text: str) -> str:
    return f"""Generated job:\n{job_text}\n\nThe deterministic schema checks have already passed.
Judge ONLY whether the Title is a plausible real software/data job title and is not gibberish.
Do not re-check seniority, years, domain, skill counts, or list lengths.

Respond ONLY with:
PASS
or
FAIL: [specific title issue]"""


def _build_seniority_years_prompt(seniority: str, years: str) -> str:
    return f"""Generated job:
Seniority: {seniority}
YearsRequired: {years}

Check seniority-years alignment:
- Junior: 0-2 years
- Mid: up to 5 years
- Senior: 2-8 years
- Staff: 6+ years

Respond ONLY with PASS or FAIL: [reason]."""


def _build_resume_job_alignment_prompt(
    resume_seniority: str,
    resume_years: int,
    resume_skills: list[str],
    resume_text: str,
    job_seniority: str,
    job_years: str,
    job_skills: list[str],
    job_responsibilities: list[str],
) -> str:
    return f"""Resume text:
{resume_text}

Generated job responsibilities:
- {chr(10).join(job_responsibilities)}

Exact seniority, experience, and skill-overlap checks have already passed.
Judge ONLY whether the responsibilities are plausible for work described by this resume.
Do not re-check seniority, years, skill counts, or domains.

Respond ONLY with:
PASS
or
FAIL: [specific responsibility mismatch]"""


def _build_domain_consistency_prompt(
    resume_domain: str,
    job_domain: str,
    job_title: str,
) -> str:
    return f"""Generated job:
Domain: {job_domain}
Title: {job_title}

The resume/job domain compatibility check has already passed.
Judge ONLY whether the job title is plausible for the stated job domain.
Do not re-check whether the domain enum is valid or compare it to the resume domain.

Respond ONLY with:
PASS
or
FAIL: [specific title/domain inconsistency]"""


def validate_responsibilities(responsibilities: list[str]) -> ValidationResult:
    if not responsibilities:
        reason = "Responsibilities list is empty"
        logger.warning("responsibilities: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    if not 3 <= len(responsibilities) <= 5:
        reason = f"Must have 3–5 responsibilities, got {len(responsibilities)}"
        logger.warning("responsibilities: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    for i, responsibility in enumerate(responsibilities):
        text = str(responsibility).strip()
        if not text or text == ".":
            reason = f"Responsibility {i} is empty or a literal period"
            logger.warning("responsibilities: FAIL — %s", reason)
            return {"passed": False, "reason": reason}
        words = len(text.split())
        if words < 10:
            reason = f"Responsibility {i} has {words} words, need ≥10: {text!r}"
            logger.warning("responsibilities: FAIL — %s", reason)
            return {"passed": False, "reason": reason}

    logger.info("responsibilities: PASS — %d items, all ≥10 words", len(responsibilities))
    return {"passed": True, "reason": None}


def validate_structural(job: JobSkeleton, model: str = OLLAMA_MODEL) -> ValidationResult:
    required = (
        "title",
        "seniority",
        "years_required",
        "domain",
        "primary_skills",
        "secondary_skills",
        "responsibilities",
    )
    missing = [key for key in required if key not in job]
    if missing:
        reason = f"Missing required fields: {', '.join(missing)}"
        logger.warning("structural: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    title = str(job["title"]).strip()
    if not title:
        reason = "Title is empty"
        logger.warning("structural: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    if job["seniority"] not in _VALID_SENIORITIES:
        reason = f"Invalid seniority: {job['seniority']!r}"
        logger.warning("structural: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    if job["domain"] not in _VALID_DOMAINS:
        reason = f"Invalid domain: {job['domain']!r}"
        logger.warning("structural: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    parts = _parse_year_parts(str(job["years_required"]))
    if not parts or any(value < 0 or value > 20 for value in parts):
        reason = (
            "YearsRequired must be an integer or numeric range within 0–20, "
            f"got {job['years_required']!r}"
        )
        logger.warning("structural: FAIL — %s", reason)
        return {"passed": False, "reason": reason}
    if len(parts) == 2 and parts[0] > parts[1]:
        reason = f"YearsRequired range is reversed: {job['years_required']!r}"
        logger.warning("structural: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    primary = list(job["primary_skills"])
    if not 2 <= len(primary) <= 4:
        reason = f"PrimarySkills must contain 2–4 items, got {len(primary)}"
        logger.warning("structural: FAIL — %s", reason)
        return {"passed": False, "reason": reason}
    if any(not str(skill).strip() for skill in primary):
        reason = "PrimarySkills contains an empty item"
        logger.warning("structural: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    job_text = (
        f"Title: {job['title']}\n"
        f"Seniority: {job['seniority']}\n"
        f"YearsRequired: {job['years_required']}\n"
        f"Domain: {job['domain']}\n"
        f"PrimarySkills: {', '.join(job['primary_skills'])}\n"
        f"SecondarySkills: {', '.join(job['secondary_skills'])}\n"
        f"Responsibilities: {'; '.join(job['responsibilities'])}"
    )
    result = _parse_validation_response(
        call_ollama_validate(_build_structural_prompt(job_text), model)
    )
    if result["passed"]:
        logger.info("structural: PASS — title: %s", job["title"])
    else:
        logger.warning("structural: FAIL — %s", result["reason"])
    return result


def validate_seniority_years(job: JobSkeleton, model: str = OLLAMA_MODEL) -> ValidationResult:
    del model
    seniority = job["seniority"]
    parts = _parse_year_parts(str(job["years_required"]))
    if not parts:
        reason = f"Unparseable YearsRequired: {job['years_required']!r}"
        logger.warning("seniority_years: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    min_years = min(parts)
    max_years = max(parts)
    brackets: dict[str, tuple[int, int]] = {
        "Junior": (0, 2),
        "Mid": (0, 5),
        "Senior": (2, 8),
        "Staff": (6, 20),
    }
    if seniority not in brackets:
        reason = f"Unknown seniority: {seniority!r}"
        logger.warning("seniority_years: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    lo, hi = brackets[seniority]
    if min_years < lo or max_years > hi:
        reason = (
            f"{seniority} requires a range within {lo}–{hi} years, "
            f"got {job['years_required']!r}"
        )
        logger.warning("seniority_years: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    logger.info("seniority_years: PASS — %s / %s years", seniority, job["years_required"])
    return {"passed": True, "reason": None}


def validate_resume_job_alignment(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> ValidationResult:
    resume_seniority = _SENIORITY_ALIASES.get(
        resume_info["seniority"].strip().lower(),
        resume_info["seniority"].strip().title(),
    )
    if job["seniority"] != resume_seniority:
        reason = (
            f"Seniority mismatch: job is {job['seniority']!r}, "
            f"resume is {resume_seniority!r}"
        )
        logger.warning("resume_job_alignment: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    min_years = _parse_years_min(str(job["years_required"]))
    if min_years > resume_info["years_experience"]:
        reason = (
            f"Years required ({job['years_required']}) exceeds resume experience "
            f"({resume_info['years_experience']})"
        )
        logger.warning("resume_job_alignment: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    resume_skills = {
        _normalize_skill(skill) for skill in resume_info["primary_skills"] if skill.strip()
    }
    job_skills = {
        _normalize_skill(skill) for skill in job["primary_skills"] if str(skill).strip()
    }
    overlap = resume_skills & job_skills
    required_overlap = min(2, len(resume_skills))
    if len(overlap) < required_overlap:
        reason = (
            f"Only {len(overlap)} resume primary skill(s) overlap with the job; "
            f"need {required_overlap}"
        )
        logger.warning("resume_job_alignment: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    result = _parse_validation_response(
        call_ollama_validate(
            _build_resume_job_alignment_prompt(
                resume_seniority,
                resume_info["years_experience"],
                resume_info["primary_skills"],
                resume_info["resume_text"],
                job["seniority"],
                job["years_required"],
                job["primary_skills"],
                job["responsibilities"],
            ),
            model,
        )
    )
    if result["passed"]:
        logger.info("resume_job_alignment: PASS — exact checks and responsibilities align")
    else:
        logger.warning("resume_job_alignment: FAIL — %s", result["reason"])
    return result


def validate_domain_consistency(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> ValidationResult:
    resume_domain = _DOMAIN_ALIASES.get(
        resume_info["domain"].strip().lower(),
        resume_info["domain"].strip().lower(),
    )
    job_domain = job["domain"]

    adjacent = {
        "backend": {"backend", "fullstack"},
        "frontend": {"frontend", "fullstack"},
        "fullstack": {"backend", "frontend", "fullstack"},
        "data": {"data"},
    }
    if resume_domain not in adjacent or job_domain not in adjacent[resume_domain]:
        reason = f"Domain mismatch: resume is {resume_domain!r}, job is {job_domain!r}"
        logger.warning("domain_consistency: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    result = _parse_validation_response(
        call_ollama_validate(
            _build_domain_consistency_prompt(resume_domain, job_domain, job["title"]),
            model,
        )
    )
    if result["passed"]:
        logger.info(
            "domain_consistency: PASS — %s compatible with resume domain %s",
            job_domain,
            resume_domain,
        )
    else:
        logger.warning("domain_consistency: FAIL — %s", result["reason"])
    return result


def validate_job_skeleton(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> dict[str, bool | str | None]:
    normalized = _normalize_skeleton(job)
    checks = [
        ("responsibilities", lambda: validate_responsibilities(normalized["responsibilities"])),
        ("structural", lambda: validate_structural(normalized, model)),
        ("seniority_years", lambda: validate_seniority_years(normalized, model)),
        (
            "resume_job_alignment",
            lambda: validate_resume_job_alignment(normalized, resume_info, model),
        ),
        (
            "domain_consistency",
            lambda: validate_domain_consistency(normalized, resume_info, model),
        ),
    ]

    for check_name, check_fn in checks:
        result = check_fn()
        if not result["passed"]:
            logger.warning(
                "validate_job_skeleton: FAIL at %s — %s",
                check_name,
                result["reason"],
            )
            return {
                "passed": False,
                "failed_check": check_name,
                "reason": result["reason"],
            }

    logger.info("validate_job_skeleton: PASS — all checks passed")
    return {"passed": True, "failed_check": None, "reason": None}
