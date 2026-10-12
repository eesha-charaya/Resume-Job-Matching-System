"""Test suite for embed_jobs.py orchestration script."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "pipeline"))

from embed_jobs import run_embedding
from src.config import EMBEDDING_DIM


@pytest.fixture
def tmp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL,
            board_token TEXT NOT NULL,
            title TEXT,
            location TEXT,
            description TEXT,
            source TEXT,
            source_url TEXT,
            company_name TEXT,
            department TEXT,
            job_type TEXT,
            scraped_at TEXT,
            updated_at TEXT,
            cleaned_description TEXT,
            preprocessed INTEGER DEFAULT 0,
            embedding BLOB,
            embedded INTEGER DEFAULT 0,
            UNIQUE(external_id, board_token)
        )
        """
    )
    conn.commit()
    conn.close()

    yield db_path

    if os.path.exists(db_path):
        os.unlink(db_path)


def _insert_job(db_path, external_id, cleaned_description, embedded=0, embedding=None):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO jobs
        (external_id, board_token, title, cleaned_description, preprocessed, embedded, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (external_id, "test-board", "Test Job", cleaned_description, 1, embedded, embedding),
    )
    conn.commit()
    conn.close()


class TestRunEmbedding:
    @patch("embed_jobs.create_client")
    @patch("embed_jobs.embed_batch")
    def test_run_embedding_adds_columns(self, mock_embed_batch, mock_create_client, tmp_db):
        _insert_job(tmp_db, "job-1", "Software engineer position")
        _insert_job(tmp_db, "job-2", "Data analyst role")

        mock_create_client.return_value = MagicMock()
        mock_embed_batch.return_value = [
            np.ones(EMBEDDING_DIM, dtype=np.float32),
            np.ones(EMBEDDING_DIM, dtype=np.float32),
        ]

        run_embedding(tmp_db)

        conn = sqlite3.connect(tmp_db)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        conn.close()
        assert "embedding" in columns
        assert "embedded" in columns

    @patch("embed_jobs.create_client")
    @patch("embed_jobs.embed_batch")
    def test_run_embedding_skips_empty_descriptions(self, mock_embed_batch, mock_create_client, tmp_db):
        _insert_job(tmp_db, "job-1", "Valid job description")
        _insert_job(tmp_db, "job-2", "")
        _insert_job(tmp_db, "job-3", None)

        mock_create_client.return_value = MagicMock()
        mock_embed_batch.return_value = [np.ones(EMBEDDING_DIM, dtype=np.float32)]

        run_embedding(tmp_db)

        texts = mock_embed_batch.call_args[0][1]
        assert texts == ["Valid job description"]

    @patch("embed_jobs.EMBEDDING_BATCH_SIZE", 2)
    @patch("embed_jobs.create_client")
    @patch("embed_jobs.embed_batch")
    def test_run_embedding_bad_batch_skipped(self, mock_embed_batch, mock_create_client, tmp_db, caplog):
        _insert_job(tmp_db, "job-1", "First job")
        _insert_job(tmp_db, "job-2", "Second job")
        _insert_job(tmp_db, "job-3", "Third job")

        mock_create_client.return_value = MagicMock()
        mock_embed_batch.side_effect = [
            RuntimeError("embedding error"),
            [np.ones(EMBEDDING_DIM, dtype=np.float32)],
        ]

        run_embedding(tmp_db)

        assert "embed_batch failed" in caplog.text
        conn = sqlite3.connect(tmp_db)
        embedded_count = conn.execute("SELECT COUNT(*) FROM jobs WHERE embedded=1").fetchone()[0]
        conn.close()
        assert embedded_count == 1

    @patch("embed_jobs.create_client")
    @patch("embed_jobs.embed_batch")
    def test_run_embedding_idempotent(self, mock_embed_batch, mock_create_client, tmp_db):
        _insert_job(tmp_db, "job-1", "Test job")
        mock_create_client.return_value = MagicMock()
        mock_embed_batch.return_value = [np.ones(EMBEDDING_DIM, dtype=np.float32)]

        run_embedding(tmp_db)
        mock_embed_batch.reset_mock()
        run_embedding(tmp_db)

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        embedded = conn.execute("SELECT COUNT(*) FROM jobs WHERE embedded=1").fetchone()[0]
        conn.close()
        assert count == 1
        assert embedded == 1
        mock_embed_batch.assert_not_called()


class TestMainFunction:
    @patch("embed_jobs.load_dotenv")
    @patch("embed_jobs.run_embedding")
    @patch("sys.argv", ["embed_jobs.py", "--db-path", "/tmp/test.db"])
    def test_main_does_not_require_external_embedding_api_key(self, mock_run_embedding, mock_load_dotenv):
        """Local Qwen embeddings should run without an embedding API key."""
        from embed_jobs import main

        main()
        mock_run_embedding.assert_called_once_with("/tmp/test.db")
