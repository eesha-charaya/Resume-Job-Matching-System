"""
Embedding utilities backed by the local Qwen3-Embedding-0.6B model.

All functions are stateless with respect to the database and environment.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def create_client(api_key: str | None = None) -> SentenceTransformer:
    """Create and return the local embedding model.

    ``api_key`` is accepted for compatibility with older callers and is ignored.
    The model runs locally and does not require an embedding API key.
    """
    logger.info("Loading local embedding model: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def embed_batch(
    client: SentenceTransformer,
    texts: list[str],
    model: str | None = None,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
    run_id: str | None = None,
) -> list[np.ndarray]:
    """Embed a batch of texts with Qwen3-Embedding-0.6B.

    Extra arguments are retained for compatibility with existing pipeline and
    evaluation interfaces; local inference does not use API retries.
    """
    if not texts:
        raise ValueError("texts must be non-empty")

    embeddings = client.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [np.asarray(vec, dtype=np.float32) for vec in embeddings]


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize a float32 embedding to raw bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes, dim: int = 1024) -> np.ndarray:
    """Deserialize a SQLite BLOB into a float32 embedding of the expected size."""
    expected_bytes = dim * 4
    if len(blob) != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} bytes for dim={dim}, got {len(blob)}"
        )
    return np.frombuffer(blob, dtype=np.float32).copy()
