from fastapi import APIRouter, Depends, HTTPException

from fastapi_app.api.schemas import MatchRequest, MatchResponse, JobMatch
from fastapi_app.api.dependencies import (
    get_embedding_client,
    get_chroma_collection,
    get_cohere_client,
    get_ollama_base_url,
)
from src.embedding import embed_batch
from src.retrieval import query_collection
from src.reranking import rerank_jobs
from src.regex_extraction import build_chroma_where_filter
from src.generation import run_generation_pipeline
from src.config import (
    MAX_BATCH_SIZE,
    CORPUS_LIMITATION_MESSAGE,
    DEGREE_UNKNOWN,
    SENIORITY_UNKNOWN,
    YEARS_UNKNOWN,
)
import logging
import requests

logger = logging.getLogger(__name__)
router = APIRouter()


def _generate_explanation_direct(
    resume: str,
    job_title: str,
    job_content: str,
    ollama_base_url: str,
) -> str | None:
    """Generate explanation using Ollama directly as a fallback."""
    try:
        prompt = f"""You are a career advisor. Explain why the resume is a good match for the job.

Resume summary: {resume[:500]}

Job title: {job_title}
Job description: {job_content[:500]}

Provide a concise explanation (2-3 sentences) highlighting key matches:"""

        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": "llama3.2:3b-instruct-q4_K_M",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "max_tokens": 150},
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as exc:
        logger.error("Failed to generate explanation: %s", exc)
        return None


def generate_explanation_with_pipeline(
    resume: str,
    job_title: str,
    job_content: str,
    ollama_base_url: str,
    model: str = "llama3.2:3b-instruct-q4_K_M",
) -> tuple[str | None, str | None]:
    """Return (explanation, corpus_warning)."""
    try:
        results, corpus_message = run_generation_pipeline(
            [(resume, job_content)],
            model=model,
        )
        if results and results[0]:
            return results[0]["explanation"], corpus_message
        return None, corpus_message
    except Exception as exc:
        logger.error("Generation pipeline failed: %s", exc)
        explanation = _generate_explanation_direct(
            resume,
            job_title,
            job_content,
            ollama_base_url,
        )
        return explanation, None


@router.post("/match", response_model=MatchResponse)
async def match(
    req: MatchRequest,
    embedding_client=Depends(get_embedding_client),
    collection=Depends(get_chroma_collection),
    cohere_client=Depends(get_cohere_client),
    ollama_base_url=Depends(get_ollama_base_url),
):
    try:
        job_count = collection.count()
        if job_count == 0:
            raise HTTPException(status_code=404, detail="no jobs in index")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to check collection count: %s", exc)
        raise HTTPException(status_code=500, detail="failed to check job index")

    user_degree = req.required_degree if req.required_degree is not None else DEGREE_UNKNOWN
    user_seniority = req.seniority_level if req.seniority_level is not None else SENIORITY_UNKNOWN
    user_years = req.min_years_experience if req.min_years_experience is not None else YEARS_UNKNOWN
    where_filter = build_chroma_where_filter(user_degree, user_seniority, user_years)

    logger.info(
        "User filters — degree: %d, seniority: %d, years: %d",
        user_degree,
        user_seniority,
        user_years,
    )
    logger.debug("where_filter: %s", where_filter)

    try:
        embedding = embed_batch(embedding_client, [req.resume])[0]
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        raise HTTPException(status_code=503, detail="embedding service unavailable")

    try:
        retrieved = query_collection(
            collection,
            embedding,
            top_k=100,
            where=where_filter,
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail="retrieval error")

    if not retrieved:
        return MatchResponse(matches=[], resume_id=None, corpus_warning=None)

    try:
        reranked = rerank_jobs(
            query=req.resume,
            jobs=retrieved,
            client=cohere_client,
            top_n=req.top_k,
        )
    except Exception as exc:
        logger.error("Reranking failed: %s", exc)
        raise HTTPException(status_code=503, detail="reranking service unavailable")

    matches = []
    corpus_warning = None

    for job in reranked[: req.top_k]:
        try:
            job_id = int(job["id"])
        except (ValueError, KeyError):
            job_id = 0

        job_content = job.get("document", "") or job.get("cleaned_description", "") or job.get("title", "")

        explanation, warning = generate_explanation_with_pipeline(
            resume=req.resume,
            job_title=job.get("title", ""),
            job_content=job_content,
            ollama_base_url=ollama_base_url,
        )

        if warning is not None and corpus_warning is None:
            corpus_warning = warning

        distance = job.get("distance", 1.0)
        score = max(0.0, 1.0 - distance)
        absolute_url = job.get("source_url") or None

        matches.append(
            JobMatch(
                job_id=job_id,
                title=job.get("title", ""),
                company_name=job.get("company_name", ""),
                score=score,
                explanation=explanation,
                absolute_url=absolute_url,
            )
        )

    return MatchResponse(matches=matches, resume_id=None, corpus_warning=corpus_warning)
