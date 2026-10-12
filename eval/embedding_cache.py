"""
Cached embedding utilities for the evaluation pipeline.

Provides a generic embed_with_cache function and wrappers for positives and resumes.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src import embedding
from eval import data_loading, eval_config

logger = logging.getLogger(__name__)


def embed_with_cache(
    embedding_client,
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    cache_path: str,
    hash_path: str,
    model: str | None = None,
    skip_empty: bool = False,
) -> dict:
    """Embed a DataFrame column with .npz/.hash caching."""
    if model is None:
        model = config.EMBEDDING_MODEL

    cache_p = Path(cache_path)
    hash_p = Path(hash_path)

    sorted_texts = df.sort_values(id_col)[text_col].values.astype(str)
    cache_fingerprint = f"{model}|" + "|".join(sorted_texts)
    current_hash = data_loading.compute_hash(cache_fingerprint.encode("utf-8"))

    if cache_p.exists() and hash_p.exists():
        try:
            with open(hash_p) as f:
                cached_hash = f.read().strip()
            if cached_hash == current_hash:
                logger.info("Loading embeddings from cache: %s", cache_path)
                cached = np.load(cache_p)
                result = {k: cached[k] for k in cached.files}
                logger.info("Loaded %d embeddings from cache", len(result))
                return result
        except Exception as exc:
            logger.warning("Cache load failed: %s; will re-embed", exc)

    logger.info("Embedding from column: %s", text_col)
    embeddings_dict = {}
    empty_count = 0

    df = df.sort_values(id_col)
    for idx in range(0, len(df), config.EMBEDDING_BATCH_SIZE):
        batch = df.iloc[idx : idx + config.EMBEDDING_BATCH_SIZE]
        texts = []
        ids = []

        for _, row in batch.iterrows():
            text = row[text_col]
            if not text or not str(text).strip():
                if skip_empty:
                    logger.warning(
                        "Skipping empty %s for %s=%s",
                        text_col,
                        id_col,
                        row[id_col],
                    )
                    empty_count += 1
                    continue
                text = ""

            texts.append(str(text))
            ids.append(row[id_col])

        if texts:
            embeddings = embedding.embed_batch(embedding_client, texts, model=model)
            for id_val, emb in zip(ids, embeddings):
                embeddings_dict[id_val] = emb

        logger.info("Embedded %d/%d items", len(embeddings_dict), len(df))

    if empty_count > 0:
        logger.warning("Skipped %d items with empty %s", empty_count, text_col)

    cache_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_p, **{str(k): v for k, v in embeddings_dict.items()})
    with open(hash_p, "w") as f:
        f.write(current_hash)

    logger.info("Cached %d embeddings to %s", len(embeddings_dict), cache_path)
    return embeddings_dict


def embed_positives(
    embedding_client,
    positives_df: pd.DataFrame,
    model: str | None = None,
    cache_path: str | None = None,
    hash_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Embed synthetic positives, using cache when the model/text hash matches."""
    if cache_path is None:
        cache_path = eval_config.TUNE_POSITIVE_EMBEDDINGS_CACHE
    if hash_path is None:
        hash_path = eval_config.TUNE_POSITIVE_EMBEDDINGS_HASH

    embeddings = embed_with_cache(
        embedding_client,
        positives_df,
        id_col="id",
        text_col="job_description",
        cache_path=cache_path,
        hash_path=hash_path,
        model=model,
        skip_empty=True,
    )
    return {str(k): v for k, v in embeddings.items()}


def embed_resumes(
    embedding_client,
    resumes_df: pd.DataFrame,
    model: str | None = None,
    cache_path: str | None = None,
    hash_path: str | None = None,
) -> dict[int, np.ndarray]:
    """Embed resumes, using cache when the model/text hash matches."""
    if cache_path is None:
        cache_path = eval_config.TUNE_RESUME_EMBEDDINGS_CACHE
    if hash_path is None:
        hash_path = eval_config.TUNE_RESUME_EMBEDDINGS_HASH

    embeddings = embed_with_cache(
        embedding_client,
        resumes_df,
        id_col="id",
        text_col="resume",
        cache_path=cache_path,
        hash_path=hash_path,
        model=model,
        skip_empty=False,
    )
    return {int(k): v for k, v in embeddings.items()}
