"""
Orchestration script: two-stage pipeline combining dense retrieval and reranking.

Reads resume from data/user_profile.txt, retrieves top 100 candidates via dense search,
reranks them to top 10 using Cohere Rerank 3, and outputs results to matched_jobs.md.
"""

import argparse
import hashlib
import logging
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DEFAULT_DIR,
    CORPUS_LIMITATION_MESSAGE,
    DB_DEFAULT_PATH,
    EMBEDDING_MODEL,
    HNSW_EF,
    HNSW_EF_CONSTRUCTION,
    OLLAMA_MODEL,
    RETRIEVE_TOP_K,
    RERANK_TOP_N,
)
from src.embedding import create_client, embed_batch
from src.llm_extraction import (
    extract_degree_with_llm,
    extract_seniority_with_llm,
    extract_years_with_llm,
)
from src.regex_extraction import (
    build_chroma_where_filter,
    describe_chroma_filter,
    extract_years_experience,
)
from src.retrieval import build_collection, query_collection
from src.reranking import rerank_jobs

import ollama
from src.generation import run_generation_pipeline

logger = logging.getLogger(__name__)


def load_resume(resume_path: str | None = None) -> str:
    """Load and return resume text."""
    if resume_path is None:
        resume_path = str(_PROJECT_ROOT / "data" / "user_profile.txt")

    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_text = f.read().strip()
    except FileNotFoundError:
        logger.error("Resume file not found: %s", resume_path)
        sys.exit(1)

    if not resume_text:
        logger.error("Resume is empty.")
        sys.exit(1)

    return resume_text


def extract_user_filters(resume_text: str):
    """Extract structured criteria from resume text and build a Chroma filter."""
    user_degree = extract_degree_with_llm(resume_text, OLLAMA_MODEL)
    user_seniority = extract_seniority_with_llm(resume_text, OLLAMA_MODEL)
    user_years = extract_years_with_llm(resume_text, OLLAMA_MODEL)

    logger.info(
        "User profile — degree: %d, seniority: %d, years: %d",
        user_degree,
        user_seniority,
        user_years,
    )
    return build_chroma_where_filter(user_degree, user_seniority, user_years)


def _resume_hash(resume_text: str) -> str:
    """Hash both embedding-model identity and resume text for cache invalidation."""
    payload = f"{EMBEDDING_MODEL}:{resume_text}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def load_or_embed_resume(
    client,
    resume_text: str,
    cache_path: str = "data/user_profile_embedding.npy",
    hash_path: str = "data/user_profile_embedding_hash.txt",
) -> np.ndarray:
    """Load a valid cached resume embedding or compute one with the local encoder."""
    current_hash = _resume_hash(resume_text)
    cache = Path(cache_path)
    hash_file = Path(hash_path)

    if cache.exists() and hash_file.exists():
        saved_hash = hash_file.read_text().strip()
        if saved_hash == current_hash:
            logger.info("Loading cached resume embedding...")
            try:
                return np.load(str(cache))
            except Exception as exc:
                logger.warning("Failed to load cached embedding: %s. Re-embedding.", exc)

    logger.info("Embedding resume...")
    embedding = embed_batch(client, [resume_text])[0]

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache), embedding)
    hash_file.write_text(current_hash)
    logger.info("Resume embedding cached to %s", cache_path)
    return embedding


def build_chroma_client(chroma_dir: str, rebuild: bool) -> ClientAPI:
    """Create a persistent Chroma client, optionally clearing the collection first."""
    chroma_path = Path(chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    if rebuild:
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'.", CHROMA_COLLECTION_NAME)
        except Exception as exc:
            # Chroma versions differ on the exact exception for a missing collection.
            if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise

    return chroma_client


def write_results_markdown(results: list[dict] | list, output_path: str = "matched_jobs.md") -> None:
    """Write reranked results to Markdown."""
    lines = ["# Top Matched Jobs (Reranked)\n"]

    if not results:
        lines.append(CORPUS_LIMITATION_MESSAGE)
    else:
        for i, job in enumerate(results, start=1):
            title = job.get("title", "Unknown")
            url = job.get("source_url", "No URL")
            description = job.get("cleaned_description", "")

            lines.append(f"## {i}. {title}")

            min_years = extract_years_experience(description)
            if min_years > 0:
                lines.append(f"- **Min. Years of Experience:** {min_years}")

            lines.append(f"- **URL:** [{url}]({url})\n")

            explanation = job.get("explanation")
            if explanation is not None:
                lines.append(f"- **Fit Summary:** {explanation}\n")
            elif job.get("corpus_warning"):
                lines.append(f"- **Note:** {job['corpus_warning']}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Results written to %s.", output_path)


def run_generation_for_results(
    resume_text: str,
    results: list[dict] | list,
    model: str = OLLAMA_MODEL,
    run_id: str | None = None,
) -> None:
    """Attach grounded fit explanations to reranked jobs in place."""
    logger.info("Running generation pipeline for %d results...", len(results))

    for job in results:
        description = job.get("cleaned_description", "")
        if not description:
            job["explanation"] = None
            continue

        try:
            generation_results, corpus_message = run_generation_pipeline(
                pairs=[(resume_text, description)],
                model=model,
                run_id=run_id,
            )
        except ollama.RequestError as exc:
            logger.warning("Ollama is not reachable: %s. Skipping generation.", exc)
            for remaining in results:
                remaining.setdefault("explanation", None)
            return
        except ollama.ResponseError as exc:
            logger.warning("Ollama model error for job '%s': %s. Skipping.", job.get("title"), exc)
            job["explanation"] = None
            continue
        except Exception as exc:
            logger.warning("Unexpected generation error for job '%s': %s. Skipping.", job.get("title"), exc)
            job["explanation"] = None
            continue

        job["explanation"] = generation_results[0]["explanation"]
        if corpus_message is not None:
            job["corpus_warning"] = corpus_message


def main() -> None:
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Match resume to job listings using dense retrieval and reranking."
    )
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--output-path", default="matched_jobs.md")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the Chroma vector store from scratch.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    load_dotenv()

    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        logger.error("COHERE_API_KEY environment variable is not set.")
        sys.exit(1)

    try:
        ollama.list()
        logger.info("Ollama is reachable.")
    except (ollama.RequestError, ollama.ResponseError) as exc:
        logger.error("Ollama is not reachable: %s. Please start Ollama before running.", exc)
        sys.exit(1)

    run_id = str(uuid.uuid4())
    logger.info("Pipeline run_id: %s", run_id)

    db_path = args.db_path or os.getenv("DB_PATH") or DB_DEFAULT_PATH
    if not Path(db_path).exists():
        logger.error("Database not found at %s.", db_path)
        sys.exit(1)

    resume_text = load_resume(args.resume_path)
    logger.info("Loaded resume (%d characters).", len(resume_text))

    query_filter = extract_user_filters(resume_text)
    logger.info("Applying filters: %s", describe_chroma_filter(query_filter))

    chroma_client = build_chroma_client(CHROMA_DEFAULT_DIR, rebuild=args.rebuild)

    try:
        conn = sqlite3.connect(db_path)
        try:
            logger.info("Building ChromaDB collection from embeddings...")
            start_build = time.monotonic()
            collection = build_collection(
                conn,
                chroma_client,
                CHROMA_COLLECTION_NAME,
                ef_construction=HNSW_EF_CONSTRUCTION,
                model=OLLAMA_MODEL,
            )
            elapsed_build = time.monotonic() - start_build
            logger.info(
                "Collection ready with %d jobs indexed in %.2fs.",
                collection.count(),
                elapsed_build,
            )

            if collection.count() == 0:
                logger.error("No embedded jobs found in the database.")
                sys.exit(1)

            embedding_client = create_client()
            start_resume = time.monotonic()
            query_embedding = load_or_embed_resume(embedding_client, resume_text)
            elapsed_resume = time.monotonic() - start_resume
            logger.info(
                "Resume embedding loaded/computed in %.2fs, shape: %s",
                elapsed_resume,
                query_embedding.shape,
            )

            logger.info("Querying for top %d candidates...", RETRIEVE_TOP_K)
            start_query = time.monotonic()
            candidates = query_collection(
                collection,
                query_embedding,
                top_k=RETRIEVE_TOP_K,
                ef=HNSW_EF,
                where=query_filter,
                run_id=run_id,
            )
            logger.info(
                "Queried %d candidates in %.2fs.",
                len(candidates),
                time.monotonic() - start_query,
            )

            logger.info("Reranking %d candidates to top %d...", len(candidates), RERANK_TOP_N)
            start_rerank = time.monotonic()
            results = rerank_jobs(
                resume_text,
                candidates,
                top_n=RERANK_TOP_N,
                api_key=cohere_api_key,
                run_id=run_id,
            )
            logger.info(
                "Reranked to %d results in %.2fs.",
                len(results),
                time.monotonic() - start_rerank,
            )

            if not results:
                logger.warning("No matching jobs found.")
            else:
                logger.info("Running generation pipeline...")
                start_gen = time.monotonic()
                run_generation_for_results(resume_text, results, run_id=run_id)
                logger.info("Generation completed in %.2fs.", time.monotonic() - start_gen)
                write_results_markdown(results, args.output_path)
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        logger.error("Error reading database: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Error building collection (corrupt embeddings?): %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
