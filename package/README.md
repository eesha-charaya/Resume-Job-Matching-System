# Resume Job Matcher Python Package

A small installable Python package extracted from the larger Resume-Job Matching System. It exposes reusable utilities for cleaning job-description text and evaluating retrieval quality.

## Install

Create and activate a virtual environment, then upgrade the packaging tools first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

From the repository root:

```bash
python -m pip install -e "./package"
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
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "./package[test]"
python -c "import resume_job_matcher; print(resume_job_matcher.__version__)"
pytest package/tests
```

If editable install is not needed, a regular install also works:

```bash
python -m pip install "./package[test]"
```

This package is maintained as part of the Resume-Job Matching System repository.
