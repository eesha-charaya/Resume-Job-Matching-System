#!/usr/bin/env python3
"""Generate synthetic positive job descriptions for evaluation resumes.

The normal run processes every resume and targets five validated positives per
resume.  ``--limit`` and ``--target-count`` make it possible to smoke-test the
pipeline before paying the cost of a full Ollama run.
"""

import argparse
import csv
import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.positive_gen.positives_pipeline import run_pipeline
from eval.positive_gen.positives_validate import ResumeInfo

logger = logging.getLogger(__name__)


def extract_skills_from_resume(resume_text: str) -> list[str]:
    match = re.search(r"Skills\s*:\s*([^\n]+)", resume_text, re.IGNORECASE)
    if not match:
        return []
    return [skill.strip() for skill in match.group(1).split(",") if skill.strip()][:4]


def extract_years_from_resume(resume_text: str, seniority: str) -> int:
    """Read the explicit years value from our deterministic synthetic resumes."""
    match = re.search(r"with\s+(\d+)\s+years?\b", resume_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return {"Junior": 1, "Mid": 3, "Senior": 6, "Staff": 8}.get(seniority, 3)


def build_resume_info(row: dict) -> ResumeInfo:
    resume_text = row["resume"]
    seniority = row["seniority"].strip().title()
    return {
        "seniority": seniority,
        "years_experience": extract_years_from_resume(resume_text, seniority),
        "primary_skills": extract_skills_from_resume(resume_text),
        "domain": row["domain"].strip(),
        "resume_text": resume_text,
    }


def format_job_for_embedding(job: dict) -> str:
    return (
        f"{job['title']}. {job['seniority']} level. Domain: {job['domain']}. "
        f"Requires {job['years_required']} years of experience. "
        f"Skills: {', '.join(job.get('primary_skills', []))}. "
        f"Additional skills: {', '.join(job.get('secondary_skills', []))}. "
        f"Responsibilities: {'; '.join(job.get('responsibilities', []))}."
    )


def write_rows(output_csv: Path, rows: list[dict]) -> None:
    """Persist current progress so an interrupted run does not lose everything."""
    if not rows:
        return
    fieldnames = [
        "id",
        "resume_id",
        "title",
        "seniority",
        "years_required",
        "domain",
        "primary_skills",
        "secondary_skills",
        "responsibilities",
        "resume_seniority",
        "resume_domain",
        "generated_at",
        "job_description",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic positive jobs")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N resumes (useful for smoke tests)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=5,
        help="Validated positives to target per resume (default: 5)",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.target_count < 1:
        parser.error("--target-count must be at least 1")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    input_csv = project_root / "data" / "eval" / "synthetic_resume.csv"
    output_csv = project_root / "data" / "eval" / "synthetic_job_descriptions.csv"

    if not input_csv.exists():
        logger.error("Input file not found: %s", input_csv)
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8") as handle:
        resumes = list(csv.DictReader(handle))
    if args.limit is not None:
        resumes = resumes[: args.limit]

    logger.info(
        "Loaded %d resume(s). Targeting %d positive job(s) per resume.",
        len(resumes),
        args.target_count,
    )

    all_rows: list[dict] = []

    try:
        for idx, row in enumerate(resumes, 1):
            resume_id = row["id"]
            resume_text = row["resume"]
            resume_seniority = row["seniority"].strip()
            resume_domain = row["domain"].strip()

            logger.info(
                "[%d/%d] Processing resume_id=%s, seniority=%s, domain=%s",
                idx,
                len(resumes),
                resume_id,
                resume_seniority,
                resume_domain,
            )

            jobs = run_pipeline(
                resume_text=resume_text,
                resume_info=build_resume_info(row),
                target_count=args.target_count,
                output_path=None,
            )

            for job in jobs:
                all_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "resume_id": resume_id,
                        "title": job.get("title", ""),
                        "seniority": job.get("seniority", ""),
                        "years_required": job.get("years_required", ""),
                        "domain": job.get("domain", ""),
                        "primary_skills": "; ".join(job.get("primary_skills", [])),
                        "secondary_skills": "; ".join(job.get("secondary_skills", [])),
                        "responsibilities": "; ".join(job.get("responsibilities", [])),
                        "resume_seniority": resume_seniority,
                        "resume_domain": resume_domain,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "job_description": format_job_for_embedding(job),
                    }
                )

            # Checkpoint after every resume.
            write_rows(output_csv, all_rows)
            logger.info(
                "[%d/%d] Collected %d job(s) for this resume. Total=%d",
                idx,
                len(resumes),
                len(jobs),
                len(all_rows),
            )
    except KeyboardInterrupt:
        write_rows(output_csv, all_rows)
        logger.warning(
            "Interrupted. Preserved %d completed positive job(s) in %s",
            len(all_rows),
            output_csv,
        )
        raise

    write_rows(output_csv, all_rows)
    logger.info("Generation complete. Wrote %d rows to %s", len(all_rows), output_csv)


if __name__ == "__main__":
    main()
