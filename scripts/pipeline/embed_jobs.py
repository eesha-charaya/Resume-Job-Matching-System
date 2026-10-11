"""Orchestration script: embed preprocessed job descriptions with Qwen3-Embedding-0.6B."""

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import DB_CHUNK_SIZE, DB_DEFAULT_PATH, EMBEDDING_BATCH_SIZE
from src.db_utils import add_column_if_missing
from src.embedding import create_client, embed_batch, serialize_embedding

logger = logging.getLogger(__name__)


def run_embedding(db_path: str) -> None:
    """Embed all unembedded, preprocessed job descriptions in the database."""
    client = create_client()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        add_column_if_missing(cur, "jobs", "embedding", "BLOB")
        add_column_if_missing(cur, "jobs", "embedded", "INTEGER DEFAULT 0")
        conn.commit()

        cur.execute(
            "SELECT COUNT(*) FROM jobs "
            "WHERE embedded=0 AND cleaned_description IS NOT NULL AND cleaned_description != ''"
        )
        total = cur.fetchone()[0]
        logger.info("Jobs to embed: %d", total)

        embedded_count = 0
        skipped_count = 0
        start = time.monotonic()

        last_id = 0
        while True:
            cur.execute(
                "SELECT id, cleaned_description FROM jobs "
                "WHERE embedded=0 AND cleaned_description IS NOT NULL AND cleaned_description != '' "
                "AND id > ? "
                "ORDER BY id LIMIT ?",
                (last_id, DB_CHUNK_SIZE),
            )
            chunk = cur.fetchall()
            if not chunk:
                break
            last_id = chunk[-1]["id"]

            for batch_start in range(0, len(chunk), EMBEDDING_BATCH_SIZE):
                sub_batch = chunk[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                ids = [row["id"] for row in sub_batch]
                texts = [row["cleaned_description"] for row in sub_batch]
                min_id = min(ids)
                max_id = max(ids)

                sub_batch_start = time.monotonic()
                try:
                    embeddings = embed_batch(client, texts)
                except Exception as exc:
                    logger.error(
                        "embed_batch failed for ids %d-%d — %s: %s",
                        min_id,
                        max_id,
                        type(exc).__name__,
                        exc,
                    )
                    skipped_count += len(sub_batch)
                    continue

                sub_batch_elapsed = time.monotonic() - sub_batch_start
                logger.debug(
                    "Embedded sub-batch of %d jobs (id range %d-%d) in %.2fs",
                    len(sub_batch),
                    min_id,
                    max_id,
                    sub_batch_elapsed,
                )

                updates = [
                    (serialize_embedding(emb), row_id)
                    for emb, row_id in zip(embeddings, ids)
                ]
                cur.executemany(
                    "UPDATE jobs SET embedding=?, embedded=1 WHERE id=?",
                    updates,
                )
                conn.commit()
                embedded_count += len(sub_batch)

            elapsed = time.monotonic() - start
            logger.info(
                "Progress: %d embedded, %d skipped (%.1fs elapsed)",
                embedded_count,
                skipped_count,
                elapsed,
            )

        elapsed_total = time.monotonic() - start
        throughput = embedded_count / elapsed_total if elapsed_total > 0 else 0
        logger.info(
            "Done. %d jobs embedded, %d skipped in %.1fs (%.1f jobs/sec avg)",
            embedded_count,
            skipped_count,
            elapsed_total,
            throughput,
        )
    finally:
        conn.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Embed all preprocessed job descriptions using Qwen3-Embedding-0.6B."
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite database path (default: DB_PATH env var or config default)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    load_dotenv()

    db_path = args.db_path or os.getenv("DB_PATH") or DB_DEFAULT_PATH
    run_embedding(db_path)


if __name__ == "__main__":
    main()
