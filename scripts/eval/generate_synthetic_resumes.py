#!/usr/bin/env python3
"""Generate a deterministic 50-resume corpus for retrieval evaluation.

The downstream eval pipeline expects ``data/eval/synthetic_resume.csv`` with
columns ``id,resume,seniority,domain``.  The generated resumes intentionally
vary by domain, seniority, skills, projects, years, and role wording while
remaining reproducible.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOTAL_RESUMES = 50

SENIORITIES = ["junior", "mid", "senior"]
DOMAINS = ["backend", "frontend", "fullstack", "data"]

YEARS = {
    "junior": [1, 2],
    "mid": [3, 4, 5],
    "senior": [6, 7, 8],
}

DOMAIN_PROFILES = {
    "backend": {
        "titles": ["Backend Software Engineer", "API Engineer", "Platform Backend Engineer"],
        "skills": [
            ["Python", "FastAPI", "PostgreSQL", "Docker"],
            ["Java", "Spring Boot", "PostgreSQL", "Kafka"],
            ["Go", "gRPC", "Redis", "Kubernetes"],
            ["Python", "Django", "MySQL", "AWS"],
        ],
        "projects": [
            "designed REST APIs for a customer-facing service and improved request reliability",
            "built an event-driven service that processed asynchronous application workflows",
            "implemented caching and database indexing to reduce API latency under production load",
            "developed authentication and authorization services used by multiple internal applications",
        ],
    },
    "frontend": {
        "titles": ["Frontend Software Engineer", "UI Engineer", "Web Application Engineer"],
        "skills": [
            ["TypeScript", "React", "JavaScript", "CSS"],
            ["React", "Redux", "TypeScript", "Jest"],
            ["JavaScript", "Next.js", "HTML", "CSS"],
            ["TypeScript", "React", "GraphQL", "Playwright"],
        ],
        "projects": [
            "built responsive product workflows and reusable UI components for a web application",
            "migrated a client application to typed state management and improved maintainability",
            "implemented accessible dashboards with client-side caching and API integrations",
            "developed end-to-end tested user flows and improved page performance for core screens",
        ],
    },
    "fullstack": {
        "titles": ["Full Stack Software Engineer", "Product Engineer", "Full Stack Web Engineer"],
        "skills": [
            ["Python", "React", "PostgreSQL", "Docker"],
            ["Java", "Spring Boot", "React", "SQL"],
            ["TypeScript", "Node.js", "React", "MongoDB"],
            ["Python", "FastAPI", "TypeScript", "AWS"],
        ],
        "projects": [
            "delivered an end-to-end product feature spanning APIs, persistence, and interactive UI components",
            "built internal workflow tooling with backend services and a browser-based management interface",
            "implemented a user-facing application with authentication, APIs, database models, and deployment automation",
            "developed full-stack analytics features connecting data services to interactive product experiences",
        ],
    },
    "data": {
        "titles": ["Data Engineer", "Analytics Engineer", "Machine Learning Data Engineer"],
        "skills": [
            ["Python", "SQL", "Airflow", "Spark"],
            ["Python", "Pandas", "PostgreSQL", "dbt"],
            ["SQL", "Spark", "AWS", "Kafka"],
            ["Python", "PyTorch", "SQL", "MLflow"],
        ],
        "projects": [
            "built batch data pipelines that transformed raw event data into reliable analytical datasets",
            "developed data quality checks and scheduled transformations for reporting and experimentation",
            "implemented streaming ingestion and scalable processing for high-volume application events",
            "prepared feature pipelines and reproducible datasets for machine learning model development",
        ],
    },
}

IMPACT_PHRASES = [
    "improved reliability for production users",
    "reduced manual operational work for the team",
    "improved maintainability through testing and code review",
    "collaborated with product and engineering partners to ship the work",
    "added monitoring and documentation to support production ownership",
]


def build_rows() -> list[dict[str, str | int]]:
    """Build 50 deterministic but meaningfully varied synthetic resumes."""
    rows: list[dict[str, str | int]] = []

    for idx in range(1, TOTAL_RESUMES + 1):
        # Rotate independently so each seniority appears across every domain.
        seniority = SENIORITIES[(idx - 1) % len(SENIORITIES)]
        domain = DOMAINS[((idx - 1) * 3) % len(DOMAINS)]
        profile = DOMAIN_PROFILES[domain]

        years_options = YEARS[seniority]
        years = years_options[((idx - 1) // len(SENIORITIES)) % len(years_options)]
        title = profile["titles"][(idx - 1) % len(profile["titles"])]
        skills = profile["skills"][(idx - 1) % len(profile["skills"])]
        project = profile["projects"][((idx - 1) // 2) % len(profile["projects"])]
        impact = IMPACT_PHRASES[(idx - 1) % len(IMPACT_PHRASES)]

        leadership = ""
        if seniority == "mid":
            leadership = " Owned features independently and participated in design reviews."
        elif seniority == "senior":
            leadership = " Led technical design, mentored teammates, and coordinated cross-team delivery."

        resume = (
            f"{title}\n"
            f"Professional Summary: {seniority.title()} {domain} engineer with {years} "
            f"{'year' if years == 1 else 'years'} of professional software experience.\n"
            f"Skills: {', '.join(skills)}\n"
            f"Experience: {project.capitalize()}; {impact}.{leadership}\n"
            f"Education: Bachelor's degree in Computer Science or a related technical field."
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "resume", "seniority", "domain"])
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
