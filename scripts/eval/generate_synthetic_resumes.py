#!/usr/bin/env python3
"""Generate the 50-resume synthetic corpus expected by the eval pipeline.

The downstream evaluation code expects ``data/eval/synthetic_resume.csv`` with
columns: ``id,resume,seniority,domain``.  This script creates that input
reproducibly so the existing positive/negative generation and 30/20
stratified split can run from a clean checkout.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOTAL_RESUMES = 50

# These totals mirror the distribution used by the existing split tests.
SENIORITY_COUNTS = {
    "junior": 15,
    "mid": 20,
    "senior": 15,
}
DOMAIN_COUNTS = {
    "backend": 20,
    "frontend": 15,
    "devops": 15,
}

YEARS_BY_SENIORITY = {
    "junior": 1,
    "mid": 4,
    "senior": 7,
}

SKILLS_BY_DOMAIN = {
    "backend": ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "frontend": ["TypeScript", "React", "JavaScript", "CSS"],
    "devops": ["AWS", "Kubernetes", "Terraform", "Docker"],
}

ROLE_BY_DOMAIN = {
    "backend": "Backend Software Engineer",
    "frontend": "Frontend Software Engineer",
    "devops": "DevOps Engineer",
}


def _expand_counts(counts: dict[str, int]) -> list[str]:
    values: list[str] = []
    for label, count in counts.items():
        values.extend([label] * count)
    return values


def build_rows() -> list[dict[str, str | int]]:
    """Build the deterministic 50-row synthetic resume corpus."""
    seniorities = _expand_counts(SENIORITY_COUNTS)
    domains = _expand_counts(DOMAIN_COUNTS)

    if len(seniorities) != TOTAL_RESUMES or len(domains) != TOTAL_RESUMES:
        raise ValueError("Synthetic resume distributions must total 50 rows")

    rows: list[dict[str, str | int]] = []
    for idx, (seniority, domain) in enumerate(zip(seniorities, domains), start=1):
        years = YEARS_BY_SENIORITY[seniority]
        skills = SKILLS_BY_DOMAIN[domain]
        role = ROLE_BY_DOMAIN[domain]
        skills_csv = ", ".join(skills)

        resume = (
            f"{role}\n"
            f"Professional Summary: {seniority.title()} {role.lower()} with {years} years "
            f"of experience building and operating production software.\n"
            f"Skills: {skills_csv}\n"
            f"Experience: Delivered project {idx:02d} using {skills[0]} and {skills[1]}, "
            f"collaborated with cross-functional teams, reviewed code, and improved system "
            f"reliability and maintainability.\n"
            f"Education: Bachelor's degree in Computer Science."
        )

        rows.append(
            {
                "id": idx,
                "resume": resume,
                "seniority": seniority,
                "domain": domain,
            }
        )

    return rows


def write_csv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    """Write synthetic resumes using the schema expected downstream."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "resume", "seniority", "domain"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation resumes")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path (defaults to data/eval/synthetic_resume.csv)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = (
        Path(args.output)
        if args.output is not None
        else project_root / "data" / "eval" / "synthetic_resume.csv"
    )

    rows = build_rows()
    write_csv(rows, output_path)
    logger.info("Wrote %d synthetic resumes to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
