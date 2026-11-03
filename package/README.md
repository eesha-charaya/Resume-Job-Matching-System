# Resume Job Matcher Python Package

A small installable Python package extracted from the larger Resume-Job Matching System. It exposes reusable utilities for cleaning job-description text and evaluating retrieval quality.

## Install

From the repository root:

```bash
python -m pip install -e ./package
```

Or directly from the `package` branch on GitHub:

```bash
python -m pip install "git+https://github.com/eesha-charaya/Resume-Job-Matching-System.git@package#subdirectory=package"
```

## Usage

```python
from resume_job_matcher import (
    compute_metrics_at_k,
    preprocess_description,
)

cleaned = preprocess_description("<p>Python &amp; data science</p>")

metrics = compute_metrics_at_k(
    retrieved_ids=["job-1", "job-2", "job-3"],
    relevant_ids={"job-1", "job-3"},
    k_values=[1, 3],
)
```

## What is packaged

- Job-description HTML/text preprocessing
- Precision@k and recall@k retrieval metrics
- A conventional `src/` package layout
- `pyproject.toml` package metadata
- Automated tests with `pytest`

## Development

```bash
python -m pip install -e "./package[test]"
pytest package/tests
```

This package is maintained as part of the Resume-Job Matching System repository.
