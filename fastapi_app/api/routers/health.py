from fastapi import APIRouter, Depends, HTTPException

from fastapi_app.api.dependencies import (
    get_embedding_client,
    get_chroma_collection,
    get_db,
    get_ollama_base_url,
    check_ollama_health,
)
from src.embedding import embed_batch

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(
    embedding_client=Depends(get_embedding_client),
    collection=Depends(get_chroma_collection),
    db=Depends(get_db),
    ollama_base_url=Depends(get_ollama_base_url),
):
    checks = {}
    all_ok = True

    try:
        cursor = db.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        checks["database"] = {"healthy": True, "message": "ok"}
    except Exception as exc:
        checks["database"] = {"healthy": False, "message": f"error: {exc}"}
        all_ok = False

    try:
        collection.count()
        checks["chroma"] = {"healthy": True, "message": "ok"}
    except Exception as exc:
        checks["chroma"] = {"healthy": False, "message": f"error: {exc}"}
        all_ok = False

    try:
        embedding = embed_batch(embedding_client, ["test"])[0]
        healthy = embedding.shape == (1024,)
        checks["embedding_model"] = {
            "healthy": healthy,
            "message": "ok" if healthy else f"unexpected shape: {embedding.shape}",
        }
        if not healthy:
            all_ok = False
    except Exception as exc:
        checks["embedding_model"] = {"healthy": False, "message": f"error: {exc}"}
        all_ok = False

    try:
        ollama_healthy, ollama_message = check_ollama_health(ollama_base_url)
        checks["ollama"] = {"healthy": ollama_healthy, "message": ollama_message}
        if not ollama_healthy:
            all_ok = False
    except Exception as exc:
        checks["ollama"] = {"healthy": False, "message": f"error: {exc}"}
        all_ok = False

    status = "ready" if all_ok else "not ready"
    if not all_ok:
        raise HTTPException(
            status_code=503,
            detail={"status": status, "checks": checks},
        )

    return {"status": status, "checks": checks}
