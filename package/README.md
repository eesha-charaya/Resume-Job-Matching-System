# Resume Job Matcher Python Package

An installable Python package extracted from the larger Resume-Job Matching System. It exposes reusable utilities for cleaning job-description text, extracting structured requirements, building metadata filters, and evaluating ranked job retrieval.

## Features

- HTML and text preprocessing for job descriptions
- Degree, seniority, and years-of-experience extraction
- Chroma-compatible metadata filter construction
- Precision@k and recall@k evaluation
- Command-line interface
- Typed public API
- Automated tests, coverage, linting, type checking, and package build validation

## Install

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install from a local clone:

```bash
python -m pip install -e "./package"
```

Install directly from the GitHub `package` branch:

```bash
python -m pip install "git+https://github.com/eesha-charaya/Resume-Job-Matching-System.git@package#subdirectory=package"
```

For development tools:

```bash
python -m pip install -e "./package[dev]"
```

## Python API

### Clean a job description

```python
from resume_job_matcher import preprocess_description

cleaned = preprocess_description(
    "<p>Python &amp; Data Science</p><img src='tracking.png'>"
)
print(cleaned)
# Python & Data Science
```

### Extract structured requirements

```python
from resume_job_matcher import extract_job_requirements

requirements = extract_job_requirements(
    "Bachelor's degree and 3+ years of experience required.",
    title="Senior Data Scientist",
)

print(requirements)
# {
#   'required_degree': 1,
#   'seniority_level': 3,
#   'min_years_experience': 3
# }
```

### Build a metadata filter

```python
from resume_job_matcher import (
    DEGREE_MASTER,
    SENIORITY_ENTRY,
    build_metadata_filter,
)

where = build_metadata_filter(
    user_degree=DEGREE_MASTER,
    user_seniority=SENIORITY_ENTRY,
    user_years=3,
)
```

### Evaluate retrieval quality

```python
from resume_job_matcher import compute_metrics_at_k

metrics = compute_metrics_at_k(
    retrieved_ids=["job-1", "job-2", "job-3"],
    relevant_ids={"job-1", "job-3"},
    k_values=[1, 3],
)
```

## Command-line interface

The package installs a `resume-job-matcher` command.

Clean HTML:

```bash
resume-job-matcher clean "<p>Python &amp; Data Science</p>"
```

Extract requirements:

```bash
resume-job-matcher extract   "Bachelor's degree and 3+ years of experience required"   --title "Senior Data Scientist"
```

Evaluate ranked results:

```bash
resume-job-matcher metrics   --retrieved "job1,job2,job3,job4"   --relevant "job1,job3"   --k "1,2,4"
```

The CLI can also be invoked with:

```bash
python -m resume_job_matcher --help
```

## Package structure

```text
package/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE
├── src/
│   └── resume_job_matcher/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── constants.py
│       ├── extraction.py
│       ├── filtering.py
│       ├── metrics.py
│       ├── preprocessing.py
│       └── py.typed
└── tests/
    ├── test_cli.py
    ├── test_extraction.py
    ├── test_filtering.py
    ├── test_metrics.py
    └── test_preprocessing.py
```

## Development and verification

From the repository root:

```bash
python -m pip install -e "./package[dev]"

pytest package/tests --cov=resume_job_matcher --cov-report=term-missing
ruff check package/src package/tests
mypy package/src
python -m build package
python -m twine check package/dist/*
```

CI runs these checks on Python 3.9, 3.10, 3.11, and 3.12.

## Versioning

The package follows semantic versioning. Current version: **0.2.0**.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT License. See [LICENSE](LICENSE).

This package is maintained as part of the Resume-Job Matching System repository.
