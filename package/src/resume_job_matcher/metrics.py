"""Ranking metrics used to evaluate resume-job retrieval quality."""

from typing import TypedDict


class MetricsAtK(TypedDict):
    k_values: list[int]
    precision: dict[int, float]
    recall: dict[int, float]


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute precision@k for a ranked retrieval result."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(item_id in relevant_ids for item_id in top_k)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute recall@k for a ranked retrieval result."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(item_id in relevant_ids for item_id in top_k)
    return hits / len(relevant_ids)


def compute_metrics_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k_values: list[int]
) -> MetricsAtK:
    """Compute precision and recall over several k cutoffs."""
    if not k_values:
        raise ValueError("k_values must not be empty")

    return MetricsAtK(
        k_values=k_values,
        precision={k: precision_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
        recall={k: recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
    )
