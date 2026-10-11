import os
import sqlite3
from functools import lru_cache

import chromadb
import cohere
import requests

from src.embedding import create_client


@lru_cache
def get_embedding_client():
    return create_client()


@lru_cache
def get_chroma_collection():
    client = chromadb.PersistentClient(path=os.environ["CHROMA_DIR"])
    return client.get_collection(os.environ["CHROMA_COLLECTION"])


@lru_cache
def get_cohere_client():
    return cohere.Client(os.environ["COHERE_API_KEY"])


@lru_cache
def get_ollama_base_url():
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def get_db():
    conn = sqlite3.connect(os.environ["DB_PATH"], check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def check_ollama_health(base_url: str) -> tuple[bool, str]:
    """Check whether Ollama is reachable and the required model is installed."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])
        required_model = "llama3.2:3b-instruct-q4_K_M"

        if not any(model.get("name") == required_model for model in models):
            return False, f"Required model '{required_model}' not found in Ollama"

        return True, "Ollama is healthy"
    except requests.exceptions.ConnectionError as exc:
        return False, f"Ollama connection failed: {exc}"
    except requests.exceptions.Timeout as exc:
        return False, f"Ollama request timed out: {exc}"
    except requests.exceptions.HTTPError as exc:
        return False, f"Ollama server error: {exc}"
    except Exception as exc:
        return False, f"Unexpected error checking Ollama: {exc}"
