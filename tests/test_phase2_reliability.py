"""Tests for Phase 2 reliability improvements.

Validates:
1. _call_ollama() retry logic with exponential backoff
2. call_ollama_validate/repair retry logic
3. local Qwen embedding behavior
4. Jitter in reranking/scraping backoff
5. Ollama pre-flight check in match_jobs
6. SQLite connection cleanup in data_loading
7. sample_jobs raises ValueError on size reduction
"""

import sys
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import ollama
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generation import _call_ollama
from embedding import embed_batch
from reranking import rerank_jobs

# Delay imports to avoid circular dependency issues
eval_utils = None
sample_jobs = None
load_sampled_job_embeddings = None


def lazy_import_eval_utils():
    global eval_utils
    if eval_utils is None:
        from eval import eval_utils as _eval_utils
        eval_utils = _eval_utils
    return eval_utils


def lazy_import_data_loading():
    global sample_jobs, load_sampled_job_embeddings
    if sample_jobs is None:
        from eval.data_loading import sample_jobs as _sample_jobs
        from eval.data_loading import load_sampled_job_embeddings as _load_sampled
        sample_jobs = _sample_jobs
        load_sampled_job_embeddings = _load_sampled
    return sample_jobs, load_sampled_job_embeddings


class TestCallOllamaRetry:
    """Test _call_ollama() retry logic with exponential backoff."""

    def make_ollama_response(self, content: str) -> dict:
        return {"message": {"content": content}}

    @patch("generation.ollama.chat")
    def test_succeeds_on_first_attempt(self, mock_chat: MagicMock) -> None:
        mock_chat.return_value = self.make_ollama_response("test response")
        result = _call_ollama("test prompt")
        assert result == "test response"
        mock_chat.assert_called_once()

    @patch("generation.time.sleep")
    @patch("generation.ollama.chat")
    def test_retries_once_on_request_error(
        self, mock_chat: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_chat.side_effect = [
            ollama.RequestError("Connection failed"),
            self.make_ollama_response("success on retry"),
        ]
        result = _call_ollama("test prompt")
        assert result == "success on retry"
        assert mock_chat.call_count == 2
        mock_sleep.assert_called_once()

    @patch("generation.ollama.chat")
    def test_raises_after_max_retries(self, mock_chat: MagicMock) -> None:
        mock_chat.side_effect = ollama.RequestError("Connection failed")
        with pytest.raises(ollama.RequestError):
            _call_ollama("test prompt", max_retries=1)
        assert mock_chat.call_count == 2

    @patch("generation.ollama.chat")
    def test_passes_options_dict(self, mock_chat: MagicMock) -> None:
        mock_chat.return_value = self.make_ollama_response("response")
        _call_ollama("test prompt")
        call_kwargs = mock_chat.call_args[1]
        assert "options" in call_kwargs
        assert "temperature" in call_kwargs["options"]

    @patch("generation.ollama.chat")
    def test_does_not_retry_response_error(self, mock_chat: MagicMock) -> None:
        mock_chat.side_effect = ollama.ResponseError("Model error")
        with pytest.raises(ollama.ResponseError):
            _call_ollama("test prompt")
        assert mock_chat.call_count == 1


class TestCallOllamaValidateRetry:
    """Test call_ollama_validate retry logic."""

    def test_validate_has_retry_logic(self) -> None:
        eval_utils_path = Path(__file__).parent.parent / "eval" / "eval_utils.py"
        source = eval_utils_path.read_text()
        func_start = source.index("def call_ollama_validate")
        func_section = source[func_start : source.index("\ndef ", func_start + 1)]
        assert "ollama.RequestError" in func_section
        assert "attempt" in func_section
        assert "max_retries" in func_section
        assert "2 **" in func_section


class TestCallOllamaRepairRetry:
    """Test call_ollama_repair retry logic."""

    def test_repair_has_retry_logic_with_2_attempts(self) -> None:
        eval_utils_path = Path(__file__).parent.parent / "eval" / "eval_utils.py"
        source = eval_utils_path.read_text()
        func_start = source.index("def call_ollama_repair")
        next_func = source.index("\ndef ", func_start + 1)
        func_section = source[func_start:next_func]
        assert "ollama.RequestError" in func_section
        assert "attempt" in func_section
        assert "max_retries" in func_section
        assert "= 2" in func_section


class TestEmbedBatchLocalModel:
    """Test the local SentenceTransformer-based embedding path."""

    def test_uses_encode_and_returns_float32_vectors(self) -> None:
        client = MagicMock()
        client.encode.return_value = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

        result = embed_batch(client, ["test"])

        client.encode.assert_called_once()
        assert len(result) == 1
        assert result[0].dtype == np.float32
        np.testing.assert_array_equal(result[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_passes_expected_encode_options(self) -> None:
        client = MagicMock()
        client.encode.return_value = np.array([[1.0, 2.0]], dtype=np.float32)

        embed_batch(client, ["test"])

        _, kwargs = client.encode.call_args
        assert kwargs["convert_to_numpy"] is True
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["show_progress_bar"] is False

    def test_local_model_errors_propagate(self) -> None:
        """Local encoder errors are not treated like remote HTTP failures/retries."""
        client = MagicMock()
        client.encode.side_effect = RuntimeError("local inference failed")

        with pytest.raises(RuntimeError, match="local inference failed"):
            embed_batch(client, ["test"])

        client.encode.assert_called_once()

    def test_empty_batch_rejected(self) -> None:
        client = MagicMock()
        with pytest.raises(ValueError, match="texts must be non-empty"):
            embed_batch(client, [])
        client.encode.assert_not_called()


class TestBackoffJitter:
    """Test networked components that still use retry jitter."""

    def test_rerank_jobs_includes_jitter(self) -> None:
        import inspect
        source = inspect.getsource(rerank_jobs)
        assert "random.uniform" in source
        assert "delay * 0.1" in source or "0.1" in source

    def test_scrape_jobs_applies_jitter(self) -> None:
        scrape_path = Path(__file__).parent.parent / "scripts" / "pipeline" / "scrape_jobs.py"
        source = scrape_path.read_text()
        assert "import random" in source
        assert "random.uniform(0, delay * 0.1)" in source


class TestOllamaPreflightCheck:
    """Test Ollama pre-flight check in match_jobs.py."""

    def test_preflights_ollama_list_call(self) -> None:
        match_path = Path(__file__).parent.parent / "scripts" / "pipeline" / "match_jobs.py"
        source = match_path.read_text()
        assert "ollama.list()" in source
        main_start = source.index("def main()")
        main_section = source[main_start:]
        ollama_list_pos = main_section.find("ollama.list()")
        gen_pos = main_section.find("run_generation_for_results")
        assert ollama_list_pos > -1, "ollama.list() not found in main()"
        assert gen_pos > -1, "run_generation_for_results not found in main()"
        assert ollama_list_pos < gen_pos, "ollama.list() should be called before run_generation_for_results"


class TestSQLiteConnectionCleanup:
    """Test SQLite connection cleanup with context managers."""

    def test_load_sampled_job_embeddings_uses_context_manager(self) -> None:
        import inspect
        _sample_jobs, _load_sampled = lazy_import_data_loading()
        source = inspect.getsource(_load_sampled)
        assert "with sqlite3.connect" in source
        assert "conn.close()" not in source or source.count("with sqlite3.connect") > 0

    def test_sample_jobs_uses_context_manager(self) -> None:
        import inspect
        _sample_jobs, _load_sampled = lazy_import_data_loading()
        source = inspect.getsource(_sample_jobs)
        assert source.count("with sqlite3.connect") >= 2


class TestSampleJobsValueError:
    """Test sample_jobs raises ValueError when reducing sizes."""

    @patch("eval.data_loading.Path.exists")
    @patch("eval.data_loading.sqlite3.connect")
    def test_raises_value_error_insufficient_jobs(
        self, mock_connect: MagicMock, mock_exists: MagicMock
    ) -> None:
        _sample_jobs, _load_sampled = lazy_import_data_loading()
        mock_exists.return_value = False

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = False
        mock_connect.return_value = mock_conn
        mock_cursor.fetchall.return_value = [(i,) for i in range(5)]

        with pytest.raises(ValueError, match="Insufficient embedded jobs"):
            _sample_jobs("test.db", tune_n=10, test_n=10, force=True)

    @patch("eval.data_loading.Path.exists")
    @patch("eval.data_loading.sqlite3.connect")
    @patch("eval.data_loading.np.random.default_rng")
    @patch("eval.data_loading.pd.DataFrame.to_csv")
    def test_succeeds_with_sufficient_jobs(
        self,
        mock_to_csv: MagicMock,
        mock_rng: MagicMock,
        mock_connect: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        _sample_jobs, _load_sampled = lazy_import_data_loading()
        mock_exists.return_value = False

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = False
        mock_connect.return_value = mock_conn

        job_ids = list(range(1, 26))
        mock_cursor.fetchall.side_effect = [
            [(id,) for id in job_ids],
            [(id, "description") for id in job_ids[:10]],
            [(id, "description") for id in job_ids[10:20]],
        ]

        mock_rng_instance = MagicMock()
        mock_rng.return_value = mock_rng_instance
        mock_rng_instance.choice.return_value = np.arange(20)

        _sample_jobs("test.db", tune_n=10, test_n=10, force=True)
        mock_to_csv.assert_called()
