"""
ChromaDB collection management for evaluation pipeline.

Handles building, caching, and updating collections used in retrieval evaluation.
"""

import logging
import sqlite3
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

from src import config
from src import embedding
from src import regex_extraction
from eval import data_loading, eval_config

logger = logging.getLogger(__name__)


def get_or_build_tune_collection(
    tune_jobs_df: pd.DataFrame,
    db_path: str,
    chroma_dir: str = eval_config.CHROMA_TUNE_EVAL_DIR,
    hash_path: str = eval_config.TUNE_SAMPLED_JOBS_HASH,
    collection_name: str = eval_config.CHROMA_TUNE_EVAL_COLLECTION,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """Get or build an eval Chroma collection with model-aware cache invalidation."""
    hash_p = Path(hash_path)

    sorted_ids = np.sort(tune_jobs_df["job_id"].values)  # type: ignore[arg-type]
    fingerprint = (
        f"{config.EMBEDDING_MODEL}|{config.EMBEDDING_DIM}|"
        + "|".join(map(str, sorted_ids))
    )
    current_hash = data_loading.compute_hash(fingerprint.encode("utf-8"))

    if not force_rebuild and hash_p.exists():
        try:
            with open(hash_p) as f:
                cached_hash = f.read().strip()
            if cached_hash == current_hash:
                chroma_client = chromadb.PersistentClient(path=chroma_dir)
                try:
                    collection = chroma_client.get_collection(name=collection_name)
                    logger.info("Reusing existing eval collection")
                    return collection
                except Exception:
                    pass
        except Exception:
            pass

    logger.info("Building eval ChromaDB collection")
    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    try:
        chroma_client.delete_collection(name=collection_name)
        logger.info("Deleted existing collection")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw_construction": config.HNSW_EF_CONSTRUCTION},
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    job_ids = tune_jobs_df["job_id"].tolist()
    ids_to_upsert = []
    embeddings_to_upsert = []
    metadatas_to_upsert = []

    rows = data_loading.chunked_select(
        conn,
        """
        SELECT id, title, location, source_url,
               source_board AS board_token,
               cleaned_description, embedding
        FROM jobs
        WHERE id IN ({}) AND embedded = 1 AND embedding IS NOT NULL
        """,
        job_ids,
    )

    for row in rows:
        ids_to_upsert.append(f"job_{row[0]}")
        embeddings_to_upsert.append(
            embedding.deserialize_embedding(row[6], dim=config.EMBEDDING_DIM).tolist()
        )
        description = row[5] or ""
        metadatas_to_upsert.append(
            {
                "title": row[1] or "",
                "location": row[2] or "",
                "source_url": row[3] or "",
                "board_token": row[4] or "",
                "cleaned_description": description,
                "required_degree": regex_extraction.extract_degree_requirement(description),
                "seniority_level": (
                    regex_extraction.extract_seniority_level(description)
                    or regex_extraction.extract_seniority_from_title(row[1] or "")
                ),
                "min_years_experience": regex_extraction.extract_years_experience(description),
            }
        )

    conn.close()

    if not ids_to_upsert:
        raise ValueError(
            "No embedded sampled jobs were found. Run scripts/pipeline/embed_jobs.py first."
        )

    collection.upsert(
        ids=ids_to_upsert,
        embeddings=embeddings_to_upsert,
        documents=[m["cleaned_description"] for m in metadatas_to_upsert],
        metadatas=metadatas_to_upsert,
    )

    logger.info("Upserted %d jobs to eval collection", len(ids_to_upsert))

    hash_p.parent.mkdir(parents=True, exist_ok=True)
    with open(hash_p, "w") as f:
        f.write(current_hash)

    return collection


def swap_positives(
    collection: chromadb.Collection,
    prev_positive_ids: list[str],
    current_positives: pd.DataFrame,
    positive_embeddings: dict[str, np.ndarray],
) -> list[str]:
    """Delete previous positives and upsert current positives to the collection."""
    if prev_positive_ids:
        collection.delete(ids=prev_positive_ids)

    ids_to_upsert = []
    embeddings_to_upsert = []
    metadatas_to_upsert = []

    for _, row in current_positives.iterrows():
        pos_id = row["id"]
        if pos_id not in positive_embeddings:
            logger.warning("No embedding for positive %s; skipping", pos_id)
            continue

        ids_to_upsert.append(f"pos_{pos_id}")
        embeddings_to_upsert.append(positive_embeddings[pos_id].tolist())
        description = str(row["job_description"])
        title = str(row["title"]) if row["title"] else ""  # type: ignore[arg-type]
        metadatas_to_upsert.append(
            {
                "title": title,
                "location": "",
                "source_url": "",
                "board_token": "",
                "cleaned_description": description,
                "required_degree": regex_extraction.extract_degree_requirement(description),
                "seniority_level": (
                    regex_extraction.extract_seniority_level(description)
                    or regex_extraction.extract_seniority_from_title(title)
                ),
                "min_years_experience": regex_extraction.extract_years_experience(description),
            }
        )

    if ids_to_upsert:
        collection.upsert(
            ids=ids_to_upsert,
            embeddings=embeddings_to_upsert,
            documents=[m["cleaned_description"] for m in metadatas_to_upsert],
            metadatas=metadatas_to_upsert,
        )

    return ids_to_upsert
