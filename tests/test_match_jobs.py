"""Test suite for match_jobs.py orchestration script."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "pipeline"))

from src.config import EMBEDDING_DIM, CORPUS_LIMITATION_MESSAGE
from match_jobs import (
    _resume_hash,
    load_resume,
    load_or_embed_resume,
    write_results_markdown,
)


@pytest.fixture
def tmp_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def mock_embedding_client():
    """Create a mock local embedding client."""
    return MagicMock()


class TestLoadResume:
    def test_load_resume_resolved_from_project_root(self, tmp_dir):
        resume_path = os.path.join(tmp_dir, "resume.txt")
        with open(resume_path, "w") as f:
            f.write("5 years Python experience\nBachelor's degree")

        original_cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            resume_text = load_resume(resume_path)
            assert "Python" in resume_text
        finally:
            os.chdir(original_cwd)

    def test_load_resume_empty_file(self, tmp_dir):
        resume_path = os.path.join(tmp_dir, "empty_resume.txt")
        Path(resume_path).write_text("")
        with pytest.raises(SystemExit):
            load_resume(resume_path)

    def test_load_resume_whitespace_only(self, tmp_dir):
        resume_path = os.path.join(tmp_dir, "whitespace_resume.txt")
        Path(resume_path).write_text("   \n\t\n   ")
        with pytest.raises(SystemExit):
            load_resume(resume_path)

    def test_load_resume_file_not_found(self):
        with pytest.raises(SystemExit):
            load_resume("/nonexistent/path/resume.txt")

    def test_load_resume_valid_content(self, tmp_dir):
        resume_path = os.path.join(tmp_dir, "valid_resume.txt")
        content = "Senior Software Engineer with 10 years experience"
        Path(resume_path).write_text(content)
        assert load_resume(resume_path) == content


class TestLoadOrEmbedResume:
    @patch("match_jobs.embed_batch")
    def test_load_or_embed_resume_cache_hit(self, mock_embed_batch, tmp_dir, mock_embedding_client):
        cache_path = os.path.join(tmp_dir, "embedding.npy")
        hash_path = os.path.join(tmp_dir, "hash.txt")
        resume_text = "Test resume content"

        test_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        np.save(cache_path, test_embedding)
        Path(hash_path).write_text(_resume_hash(resume_text))

        result = load_or_embed_resume(
            mock_embedding_client, resume_text, cache_path, hash_path
        )

        assert np.allclose(result, test_embedding)
        mock_embed_batch.assert_not_called()

    @patch("match_jobs.embed_batch")
    def test_load_or_embed_resume_corrupted_cache(self, mock_embed_batch, tmp_dir, mock_embedding_client):
        cache_path = os.path.join(tmp_dir, "embedding.npy")
        hash_path = os.path.join(tmp_dir, "hash.txt")
        resume_text = "Test resume content"

        Path(cache_path).write_text("This is not a valid numpy file")
        Path(hash_path).write_text(_resume_hash(resume_text))
        test_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        mock_embed_batch.return_value = [test_embedding]

        result = load_or_embed_resume(
            mock_embedding_client, resume_text, cache_path, hash_path
        )

        assert np.allclose(result, test_embedding)
        mock_embed_batch.assert_called_once()

    @patch("match_jobs.embed_batch")
    def test_load_or_embed_resume_hash_mismatch(self, mock_embed_batch, tmp_dir, mock_embedding_client):
        cache_path = os.path.join(tmp_dir, "embedding.npy")
        hash_path = os.path.join(tmp_dir, "hash.txt")
        resume_text = "New resume content"

        np.save(cache_path, np.zeros(EMBEDDING_DIM, dtype=np.float32))
        Path(hash_path).write_text("old_hash_value")
        new_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        mock_embed_batch.return_value = [new_embedding]

        result = load_or_embed_resume(
            mock_embedding_client, resume_text, cache_path, hash_path
        )

        assert np.allclose(result, new_embedding)
        mock_embed_batch.assert_called_once()


class TestApiKeyValidation:
    @patch("match_jobs.load_dotenv")
    @patch("sys.argv", ["match_jobs.py", "--db-path", "/tmp/test.db"])
    def test_cohere_key_validated_before_build_collection(self, mock_load_dotenv):
        """Cohere is the only external model API key required by match_jobs."""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "COHERE_API_KEY": None,
                "DB_PATH": "/tmp/test.db",
            }.get(key, default)

            with patch("match_jobs.build_collection") as mock_build:
                with pytest.raises(SystemExit):
                    from match_jobs import main
                    main()
                mock_build.assert_not_called()


class TestWriteResultsMarkdown:
    def test_write_results_markdown_no_explanations(self, tmp_dir):
        output_path = os.path.join(tmp_dir, "results.md")
        results = [
            {
                "id": 1,
                "title": "Job A",
                "source_url": "https://example.com/job-a",
                "cleaned_description": "Some description",
                "explanation": None,
                "corpus_warning": CORPUS_LIMITATION_MESSAGE,
            },
            {
                "id": 2,
                "title": "Job B",
                "source_url": "https://example.com/job-b",
                "cleaned_description": "Another description",
                "explanation": None,
                "corpus_warning": CORPUS_LIMITATION_MESSAGE,
            },
        ]

        write_results_markdown(results, output_path)
        assert os.path.exists(output_path)
        assert CORPUS_LIMITATION_MESSAGE in Path(output_path).read_text()

    def test_write_results_markdown_with_explanations(self, tmp_dir):
        output_path = os.path.join(tmp_dir, "results.md")
        results = [
            {
                "id": 1,
                "title": "Senior Engineer",
                "source_url": "https://example.com/job-1",
                "cleaned_description": "5+ years Python required",
                "explanation": "Great fit: you have 10 years experience.",
            },
            {
                "id": 2,
                "title": "Data Scientist",
                "source_url": "https://example.com/job-2",
                "cleaned_description": "ML expertise needed",
                "explanation": "Good fit: ML background matches.",
            },
        ]

        write_results_markdown(results, output_path)
        content = Path(output_path).read_text()
        assert "Senior Engineer" in content
        assert "Data Scientist" in content
        assert "Great fit" in content
        assert "Good fit" in content
        assert CORPUS_LIMITATION_MESSAGE not in content

    def test_write_results_markdown_empty_results(self, tmp_dir):
        output_path = os.path.join(tmp_dir, "results.md")
        write_results_markdown([], output_path)
        assert os.path.exists(output_path)
        assert CORPUS_LIMITATION_MESSAGE in Path(output_path).read_text()
