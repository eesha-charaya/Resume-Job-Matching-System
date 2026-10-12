#!/usr/bin/env python3
"""
Generate synthetic negative job descriptions for all resumes in synthetic_resume.csv.

For each resume, generates intentionally mismatched job descriptions:
- 1 seniority-mismatched job
- 2 responsibility-mismatched jobs

Each job is validated and repaired if needed. Deterministic fields (title, seniority, domain,
years_required) are generated using the same approach as the positives pipeline to ensure
structural consistency. Results are written to a CSV file with:
  - A unique UUID per job
  - Resume ID reference
  - All JobSkeleton fields
  - Denormalized resume metadata (seniority, domain)
  - An embedding-ready plain-text job_description
  - The mismatch_type (seniority/responsibility)
  - ISO 8601 generation timestamp

Output: data/eval/synthetic_negative_job_descriptions.csv
"""

import argparse
import csv
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add the project root so package imports like `src.*`, `eval.*`, and
# `scripts.*` work when this file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ollama

from src.config import OLLAMA_MODEL
from eval.negative_gen.negatives_gen import (
    generate_mismatched_skeleton,
    MismatchType,
)
from eval.negative_gen.negatives_validate import validate_mismatched_skeleton
from eval.negative_gen.negatives_repair import repair_mismatched_skeleton
from eval.positive_gen.positives_validate import ResumeInfo

# Reuse helpers from the positives script
from scripts.eval.generate_synthetic_positives import (
    build_resume_info,
    format_job_for_embedding,
)

logger = logging.getLogger(__name__)


def run_negatives_for_resume(
    resume_info: ResumeInfo,
    mismatch_type: MismatchType,
    target_count: int,
    model: str = OLLAMA_MODEL,
) -> list[dict]:
    """
    Generate mismatched job skeletons for a single resume and mismatch type.

    Runs the generate → validate → repair loop, collecting valid skeletons
    until target_count is reached or max_attempts is exhausted.
    """
    collected: list[dict] = []
    total_attempts = 0
    max_attempts = target_count * 10
    discard_count = 0

    logger.info(
        "[%s] Starting: target=%d, max_attempts=%d, model=%s",
        mismatch_type,
        target_count,
        max_attempts,
        model,
    )

    while len(collected) < target_count:
        if total_attempts >= max_attempts:
            logger.warning(
                "[%s] Safety cap reached (%d attempts). Collected %d/%d jobs.",
                mismatch_type,
                max_attempts,
                len(collected),
                target_count,
            )
            break

        total_attempts += 1
        logger.info(
            "[%s] Attempt %d/%d — generating skeleton...",
            mismatch_type,
            total_attempts,
            max_attempts,
        )

        try:
            job, mismatch_context = generate_mismatched_skeleton(
                resume_info=resume_info,
                model=model,
                mismatch_type=mismatch_type,
            )
        except ValueError as e:
            logger.warning("[%s] Generation parse/value error: %s", mismatch_type, e)
            continue
        except (ollama.RequestError, ollama.ResponseError) as e:
            logger.error("[%s] Ollama error during generation: %s", mismatch_type, e)
            continue

        logger.info(
            "[%s] Generated: title='%s', seniority=%s, domain=%s",
            mismatch_type,
            job["title"],
            job["seniority"],
            job["domain"],
        )

        try:
            validation = validate_mismatched_skeleton(
                job=job,
                resume_info=resume_info,
                model=model,
                mismatch_type=mismatch_type,
            )
        except (ollama.RequestError, ollama.ResponseError) as e:
            logger.error("[%s] Ollama error during validation: %s", mismatch_type, e)
            continue

        if validation["passed"]:
            collected.append(job)
            logger.info(
                "[%s] Validation PASSED — collected %d/%d",
                mismatch_type,
                len(collected),
                target_count,
            )
            continue

        failed_check = validation["failed_check"] or "structural"
        logger.warning(
            "[%s] Validation FAILED at '%s': %s",
            mismatch_type,
            failed_check,
            validation["reason"],
        )
        logger.info("[%s] Attempting repair...", mismatch_type)

        try:
            repair_result = repair_mismatched_skeleton(
                job=job,
                failed_check=failed_check,
                reason=validation["reason"],
                resume_info=resume_info,
                mismatch_context=mismatch_context,
                model=model,
                mismatch_type=mismatch_type,
            )
        except (ollama.RequestError, ollama.ResponseError) as e:
            logger.error("[%s] Ollama error during repair: %s", mismatch_type, e)
            discard_count += 1
            continue

        if repair_result["success"] and repair_result["job"] is not None:
            collected.append(repair_result["job"])
            logger.info(
                "[%s] Repair SUCCEEDED (%d attempt(s)) — collected %d/%d",
                mismatch_type,
                repair_result["attempts"],
                len(collected),
                target_count,
            )
        else:
            discard_count += 1
            logger.warning(
                "[%s] Repair FAILED after %d attempt(s) — discarding. Reason: %s",
                mismatch_type,
                repair_result["attempts"],
                repair_result["discard_reason"],
            )

    logger.info(
        "[%s] Done. collected=%d, total_attempts=%d, discards=%d",
        mismatch_type,
        len(collected),
        total_attempts,
        discard_count,
    )
    return collected


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic negative jobs")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N resumes (useful for smoke tests)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    input_csv = project_root / "data" / "eval" / "synthetic_resume.csv"
    output_csv = project_root / "data" / "eval" / "synthetic_negative_job_descriptions.csv"

    if not input_csv.exists():
        logger.error("Input file not found: %s", input_csv)
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(input_csv, "r", encoding="utf-8") as f:
        resumes = list(csv.DictReader(f))

    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        resumes = resumes[: args.limit]

    logger.info("Loaded %d resumes. Starting negatives generation...", len(resumes))

    all_rows = []
    total_collected = 0

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
        "mismatch_type",
    ]

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

        if not resume_text.strip():
            logger.warning("[%d/%d] Empty resume text; skipping.", idx, len(resumes))
            continue

        resume_info = build_resume_info(row)

        generation_tasks = [
            ("seniority", 1),
            ("responsibility", 2),
        ]

        for mismatch_type, target_count in generation_tasks:
            logger.info(
                "[%d/%d] Running mismatch_type=%s, target=%d",
                idx,
                len(resumes),
                mismatch_type,
                target_count,
            )
            try:
                jobs = run_negatives_for_resume(
                    resume_info=resume_info,
                    mismatch_type=mismatch_type,
                    target_count=target_count,
                )
            except Exception as e:
                logger.error(
                    "[%d/%d] Unexpected error for resume_id=%s, mismatch_type=%s: %s",
                    idx,
                    len(resumes),
                    resume_id,
                    mismatch_type,
                    e,
                )
                continue

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
                        "mismatch_type": mismatch_type,
                    }
                )
                total_collected += 1

            logger.info(
                "[%d/%d] mismatch_type=%s collected %d/%d",
                idx,
                len(resumes),
                mismatch_type,
                len(jobs),
                target_count,
            )

        # Checkpoint after every resume so completed work survives interruption.
        if all_rows:
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

        logger.info(
            "[%d/%d] Resume done. Total so far: %d",
            idx,
            len(resumes),
            total_collected,
        )

    if all_rows:
        logger.info("Generation complete. Wrote %d rows to %s", len(all_rows), output_csv)
    else:
        logger.warning("No jobs collected. CSV was not written.")


if __name__ == "__main__":
    main()
