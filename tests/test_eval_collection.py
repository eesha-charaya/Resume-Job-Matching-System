"""Tests for eval.collection module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from eval.collection import get_or_build_tune_collection, swap_positives
from src.config import EMBEDDING_DIM


def _mock_job_row(job_id: int = 1):
    embedding_blob = np.ones(EMBEDDING_DIM, dtype=np.float32).tobytes()
    return (
        job_id,
        "Software Engineer",
        "Seattle, WA",
        "https://example.com/job",
        "example-board",
        "Build Python systems. 3+ years of experience required.",
        embedding_blob,
    )


class TestGetOrBuildTuneCollection:
    @patch("eval.collection.chromadb.PersistentClient")
    @patch("eval.collection.data_loading.compute_hash")
    @patch("eval.collection.Path.exists")
    def test_reuse_existing_collection_on_hash_match(
        self, mock_exists, mock_hash, mock_chroma
    ) -> None:
        mock_exists.return_value = True
        mock_hash.return_value = "matching_hash"
        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        df = pd.DataFrame({"job_id": [1, 2, 3]})

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "matching_hash"
            result = get_or_build_tune_collection(df, "dummy.db", force_rebuild=False)

        mock_client.get_collection.assert_called_once()
        assert result == mock_collection

    @patch("eval.collection.sqlite3.connect")
    @patch("eval.collection.chromadb.PersistentClient")
    @patch("eval.collection.data_loading.chunked_select")
    @patch("eval.collection.data_loading.compute_hash")
    @patch("eval.collection.Path.exists")
    def test_rebuild_collection_on_hash_mismatch(
        self, mock_exists, mock_hash, mock_chunked, mock_chroma, mock_connect
    ) -> None:
        mock_exists.return_value = True
        mock_hash.return_value = "new_hash"
        mock_chunked.return_value = [_mock_job_row()]
        mock_connect.return_value = MagicMock()

        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        df = pd.DataFrame({"job_id": [1, 2, 3]})
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "old_hash"
            result = get_or_build_tune_collection(df, "dummy.db", force_rebuild=False)

        mock_client.get_or_create_collection.assert_called_once()
        mock_collection.upsert.assert_called_once()
        assert result == mock_collection

    @patch("eval.collection.sqlite3.connect")
    @patch("eval.collection.chromadb.PersistentClient")
    @patch("eval.collection.data_loading.chunked_select")
    @patch("eval.collection.data_loading.compute_hash")
    def test_force_rebuild_creates_new_collection(
        self, mock_hash, mock_chunked, mock_chroma, mock_connect
    ) -> None:
        mock_hash.return_value = "current_hash"
        mock_chunked.return_value = [_mock_job_row()]
        mock_connect.return_value = MagicMock()

        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        df = pd.DataFrame({"job_id": [1, 2, 3]})
        with patch("builtins.open", create=True):
            get_or_build_tune_collection(df, "dummy.db", force_rebuild=True)

        mock_client.delete_collection.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()
        mock_collection.upsert.assert_called_once()

    @patch("eval.collection.data_loading.compute_hash")
    def test_cache_fingerprint_includes_embedding_model(self, mock_hash) -> None:
        """Changing embedding model identity must invalidate an eval collection cache."""
        mock_hash.return_value = "hash"
        df = pd.DataFrame({"job_id": [1]})

        with patch("eval.collection.Path.exists", return_value=True), \
             patch("builtins.open", create=True) as mock_open, \
             patch("eval.collection.chromadb.PersistentClient") as mock_chroma:
            mock_open.return_value.__enter__.return_value.read.return_value = "hash"
            mock_chroma.return_value.get_collection.return_value = MagicMock()
            get_or_build_tune_collection(df, "dummy.db")

        fingerprint_bytes = mock_hash.call_args[0][0]
        assert b"Qwen/Qwen3-Embedding-0.6B" in fingerprint_bytes


class TestSwapPositives:
    def test_swap_positives_deletes_previous(self) -> None:
        mock_collection = MagicMock()
        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1", "uuid-2"],
                "title": ["Title 1", "Title 2"],
                "job_description": ["Desc 1", "Desc 2"],
            }
        )
        positive_embeddings = {
            "uuid-1": np.array([1.0, 2.0]),
            "uuid-2": np.array([3.0, 4.0]),
        }
        prev_ids = ["pos_old-uuid-1", "pos_old-uuid-2"]

        swap_positives(mock_collection, prev_ids, positives_df, positive_embeddings)
        mock_collection.delete.assert_called_once_with(ids=prev_ids)

    def test_swap_positives_upserts_current(self) -> None:
        mock_collection = MagicMock()
        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1"],
                "title": ["Software Engineer"],
                "job_description": ["Build systems..."],
            }
        )
        positive_embeddings = {"uuid-1": np.array([1.0, 2.0, 3.0])}

        swap_positives(mock_collection, [], positives_df, positive_embeddings)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]
        assert "pos_uuid-1" in call_kwargs["ids"]
        assert len(call_kwargs["embeddings"]) == 1
        assert len(call_kwargs["metadatas"]) == 1

    def test_swap_positives_returns_current_ids(self) -> None:
        mock_collection = MagicMock()
        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1", "uuid-2"],
                "title": ["Title 1", "Title 2"],
                "job_description": ["Desc 1", "Desc 2"],
            }
        )
        positive_embeddings = {
            "uuid-1": np.array([1.0, 2.0]),
            "uuid-2": np.array([3.0, 4.0]),
        }

        result = swap_positives(mock_collection, [], positives_df, positive_embeddings)
        assert result == ["pos_uuid-1", "pos_uuid-2"]

    def test_swap_positives_skips_missing_embeddings(self) -> None:
        mock_collection = MagicMock()
        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1", "uuid-2"],
                "title": ["Title 1", "Title 2"],
                "job_description": ["Desc 1", "Desc 2"],
            }
        )
        positive_embeddings = {"uuid-1": np.array([1.0, 2.0])}

        result = swap_positives(mock_collection, [], positives_df, positive_embeddings)
        assert result == ["pos_uuid-1"]
