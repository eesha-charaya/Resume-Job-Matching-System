"""
Tests for eval.embedding_cache module.

Tests embed_with_cache and wrapper functions for caching behavior.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.embedding_cache import embed_positives, embed_resumes, embed_with_cache


class TestEmbedWithCache:
    def test_embed_with_cache_hit(self, tmp_path) -> None:
        cache_path = tmp_path / "embeddings.npz"
        hash_path = tmp_path / "embeddings.hash"
        test_data = {"id1": np.array([1.0, 2.0, 3.0])}
        np.savez(cache_path, **test_data)
        hash_path.write_text("test_hash")
        df = pd.DataFrame({"id": ["id1"], "text": ["sample text"]})

        with patch("eval.embedding_cache.data_loading.compute_hash") as mock_hash:
            mock_hash.return_value = "test_hash"
            embedding_client = MagicMock()
            result = embed_with_cache(
                embedding_client,
                df,
                id_col="id",
                text_col="text",
                cache_path=str(cache_path),
                hash_path=str(hash_path),
            )

        assert "id1" in result
        np.testing.assert_array_equal(result["id1"], test_data["id1"])

    def test_embed_with_cache_miss_hash_mismatch(self, tmp_path) -> None:
        cache_path = tmp_path / "embeddings.npz"
        hash_path = tmp_path / "embeddings.hash"
        np.savez(cache_path, id1=np.array([1.0, 2.0, 3.0]))
        hash_path.write_text("old_hash")
        df = pd.DataFrame({"id": ["id1"], "text": ["new text"]})

        with patch("eval.embedding_cache.data_loading.compute_hash") as mock_hash:
            mock_hash.return_value = "new_hash"
            with patch("eval.embedding_cache.embedding.embed_batch") as mock_embed:
                mock_embed.return_value = [np.array([4.0, 5.0, 6.0])]
                embedding_client = MagicMock()
                result = embed_with_cache(
                    embedding_client,
                    df,
                    id_col="id",
                    text_col="text",
                    cache_path=str(cache_path),
                    hash_path=str(hash_path),
                )

                mock_embed.assert_called_once()

        np.testing.assert_array_equal(result["id1"], np.array([4.0, 5.0, 6.0]))

    def test_embed_with_cache_skip_empty(self, tmp_path) -> None:
        cache_path = tmp_path / "embeddings.npz"
        hash_path = tmp_path / "embeddings.hash"
        df = pd.DataFrame({"id": ["id1", "id2"], "text": ["valid text", ""]})

        with patch("eval.embedding_cache.data_loading.compute_hash") as mock_hash:
            mock_hash.return_value = "test_hash"
            with patch("eval.embedding_cache.embedding.embed_batch") as mock_embed:
                mock_embed.return_value = [np.array([1.0, 2.0])]
                embedding_client = MagicMock()
                result = embed_with_cache(
                    embedding_client,
                    df,
                    id_col="id",
                    text_col="text",
                    cache_path=str(cache_path),
                    hash_path=str(hash_path),
                    skip_empty=True,
                )

                mock_embed.assert_called_once()
                assert len(mock_embed.call_args[0][1]) == 1

        assert "id1" in result
        assert "id2" not in result


class TestEmbedPositives:
    def test_embed_positives_returns_string_keys(self, tmp_path) -> None:
        cache_path = tmp_path / "positives.npz"
        hash_path = tmp_path / "positives.hash"
        df = pd.DataFrame(
            {
                "id": ["uuid-1", "uuid-2"],
                "job_description": ["desc1", "desc2"],
            }
        )

        with patch("eval.embedding_cache.embed_with_cache") as mock_cache:
            mock_cache.return_value = {"uuid-1": np.array([1.0, 2.0])}
            embedding_client = MagicMock()
            result = embed_positives(
                embedding_client,
                df,
                cache_path=str(cache_path),
                hash_path=str(hash_path),
            )

        assert all(isinstance(k, str) for k in result.keys())


class TestEmbedResumes:
    def test_embed_resumes_returns_int_keys(self, tmp_path) -> None:
        cache_path = tmp_path / "resumes.npz"
        hash_path = tmp_path / "resumes.hash"
        df = pd.DataFrame({"id": [1, 2], "resume": ["resume1", "resume2"]})

        with patch("eval.embedding_cache.embed_with_cache") as mock_cache:
            mock_cache.return_value = {
                "1": np.array([1.0, 2.0]),
                "2": np.array([3.0, 4.0]),
            }
            embedding_client = MagicMock()
            result = embed_resumes(
                embedding_client,
                df,
                cache_path=str(cache_path),
                hash_path=str(hash_path),
            )

        assert all(isinstance(k, int) for k in result.keys())
        assert 1 in result
        assert 2 in result
