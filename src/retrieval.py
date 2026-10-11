"""
Retrieval module: Chroma collection management and dense vector search.

All functions are stateless and side-effect-free beyond Chroma persistence.
No environment loading, path resolution, or DB connection management is performed here.
"""

import logging
import sqlite3
from typing import Optional, TypedDict

import chromadb
from chromadb.api import ClientAPI
import numpy as np

from src.config import (
    CHROMA_COLLECTION_NAME,
    DEGREE_UNKNOWN,
    EMBEDDING_DIM,
    SENIORITY_UNKNOWN,
    YEARS_UNKNOWN,
    OLLAMA_MODEL,
    USE_LLM_FALLBACK,
)
from src.embedding import deserialize_embedding
from src.regex_extraction import (
    extract_degree_requirement,
    extract_seniority_from_title,
    extract_seniority_level,
    extract_years_experience,
    extract_degree_with_fallback,
    extract_seniority_with_fallback,
    extract_years_with_fallback,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME: str = CHROMA_COLLECTION_NAME
SQLITE_CHUNK_SIZE: int = 500


class JobResult(TypedDict):
    """Result of a vector similarity query."""

    id: str
    distance: float
    title: str
    company_name: str
    location: str
    source_url: str
    board_token: str
    cleaned_description: str
    required_degree: int
    seniority_level: int
    min_years_experience: int


def build_collection(
    conn: sqlite3.Connection,
    chroma_client: ClientAPI,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    ef_construction: int = 100,
    model: str = OLLAMA_MODEL,
    use_llm_fallback: bool = USE_LLM_FALLBACK,
) -> chromadb.Collection:
    """Load embedded jobs from SQLite and upsert them into a Chroma collection."""
    conn.row_factory = sqlite3.Row
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw_construction": ef_construction},
    )

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, title, company_name, location, source_url,
               source_board AS board_token,
               cleaned_description, embedding
        FROM jobs
        WHERE embedded = 1 AND embedding IS NOT NULL
        """
    )

    total_upserted = 0

    while True:
        rows = cursor.fetchmany(SQLITE_CHUNK_SIZE)
        if not rows:
            break

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for row in rows:
            ids.append(str(row["id"]))
            embeddings.append(
                deserialize_embedding(row["embedding"], dim=EMBEDDING_DIM).tolist()
            )
            desc = row["cleaned_description"] or ""
            documents.append(desc)

            if use_llm_fallback:
                degree = extract_degree_with_fallback(desc, model=model)
                seniority = (
                    extract_seniority_with_fallback(desc, model=model)
                    or extract_seniority_from_title(row["title"] or "")
                )
                years = extract_years_with_fallback(desc, model=model)
            else:
                degree = extract_degree_requirement(desc)
                seniority = (
                    extract_seniority_level(desc)
                    or extract_seniority_from_title(row["title"] or "")
                )
                years = extract_years_experience(desc)

            metadatas.append(
                {
                    "title": row["title"] or "",
                    "company_name": row["company_name"] or "",
                    "location": row["location"] or "",
                    "source_url": row["source_url"] or "",
                    "board_token": row["board_token"] or "",
                    "cleaned_description": desc,
                    "required_degree": degree,
                    "seniority_level": seniority,
                    "min_years_experience": years,
                }
            )

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        total_upserted += len(rows)
        logger.info(
            "Upserted %d rows to collection '%s' (batch of %d)",
            total_upserted,
            collection_name,
            len(rows),
        )

    logger.info(
        "Collection '%s' sync complete: %d total jobs indexed",
        collection_name,
        total_upserted,
    )
    return collection


def query_collection(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    top_k: int = 10,
    ef: int = 10,
    where: Optional[dict] = None,
    run_id: str | None = None,
) -> list[JobResult]:
    """Query a Chroma collection for the most similar jobs using dense vectors."""
    if query_embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Expected query_embedding shape ({EMBEDDING_DIM},), got {query_embedding.shape}"
        )
    if query_embedding.dtype != np.float32:
        raise ValueError(
            f"Expected query_embedding dtype float32, got {query_embedding.dtype}"
        )

    collection.modify(metadata={"hnsw:ef": ef})
    query_kwargs: dict = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": top_k,
        "include": ["metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where
    result = collection.query(**query_kwargs)

    assert result["ids"] is not None
    assert result["distances"] is not None
    assert result["metadatas"] is not None

    ids = result["ids"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]

    results = []
    for doc_id, dist, meta in zip(ids, distances, metadatas):
        assert isinstance(meta, dict)

        title_val = meta.get("title")
        title = str(title_val) if title_val else ""
        company_name_val = meta.get("company_name")
        company_name = str(company_name_val) if company_name_val else ""
        location_val = meta.get("location")
        location = str(location_val) if location_val else ""
        source_url_val = meta.get("source_url")
        source_url = str(source_url_val) if source_url_val else ""
        board_token_val = meta.get("board_token")
        board_token = str(board_token_val) if board_token_val else ""
        cleaned_desc_val = meta.get("cleaned_description")
        cleaned_description = str(cleaned_desc_val) if cleaned_desc_val else ""

        degree_val = meta.get("required_degree", DEGREE_UNKNOWN)
        required_degree = int(degree_val) if isinstance(degree_val, (int, str, float)) else DEGREE_UNKNOWN
        seniority_val = meta.get("seniority_level", SENIORITY_UNKNOWN)
        seniority_level = int(seniority_val) if isinstance(seniority_val, (int, str, float)) else SENIORITY_UNKNOWN
        years_val = meta.get("min_years_experience", YEARS_UNKNOWN)
        min_years_experience = int(years_val) if isinstance(years_val, (int, str, float)) else YEARS_UNKNOWN

        results.append(
            JobResult(
                id=doc_id,
                distance=float(dist),
                title=title,
                company_name=company_name,
                location=location,
                source_url=source_url,
                board_token=board_token,
                cleaned_description=cleaned_description,
                required_degree=required_degree,
                seniority_level=seniority_level,
                min_years_experience=min_years_experience,
            )
        )
    return results
