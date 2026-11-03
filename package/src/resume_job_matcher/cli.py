"""Command-line interface for resume-job-matcher."""

import argparse
import json
from typing import List, Optional

from .extraction import extract_job_requirements
from .metrics import compute_metrics_at_k
from .preprocessing import preprocess_description


def _csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(item) for item in _csv_list(value)]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="resume-job-matcher",
        description="Utilities for preprocessing and evaluating resume-job matching data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean", help="Clean an HTML job description")
    clean.add_argument("text")

    extract = subparsers.add_parser("extract", help="Extract structured job requirements")
    extract.add_argument("text")
    extract.add_argument("--title", default="")

    metrics = subparsers.add_parser("metrics", help="Compute precision/recall at k")
    metrics.add_argument("--retrieved", required=True, help="Comma-separated ranked job IDs")
    metrics.add_argument("--relevant", required=True, help="Comma-separated relevant job IDs")
    metrics.add_argument("--k", default="1,5,10", help="Comma-separated k values")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)

    if args.command == "clean":
        print(preprocess_description(args.text))
        return 0

    if args.command == "extract":
        print(json.dumps(extract_job_requirements(args.text, title=args.title), indent=2))
        return 0

    result = compute_metrics_at_k(
        retrieved_ids=_csv_list(args.retrieved),
        relevant_ids=set(_csv_list(args.relevant)),
        k_values=_csv_ints(args.k),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
