import pytest

from resume_job_matcher import compute_metrics_at_k, precision_at_k, recall_at_k


RETRIEVED = ["j1", "j2", "j3", "j4", "j5"]
RELEVANT = {"j1", "j3", "j5"}


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(2 / 3)


def test_compute_metrics_at_k():
    result = compute_metrics_at_k(RETRIEVED, RELEVANT, [1, 3, 5])
    assert result["precision"][1] == 1.0
    assert result["recall"][5] == 1.0


def test_invalid_k():
    with pytest.raises(ValueError, match="k must be >= 1"):
        precision_at_k(RETRIEVED, RELEVANT, 0)
