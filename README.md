# Resume-Job Matching System

An end-to-end resume-to-job matching pipeline that retrieves relevant job postings with dense embeddings, reranks them with a cross-encoder, and generates grounded fit explanations with a local LLM.

The current implementation uses **Qwen3-Embedding-0.6B locally** for embeddings, **ChromaDB HNSW** for approximate nearest-neighbor retrieval, **Cohere Rerank** for second-stage ranking, and **Ollama + Llama 3.2 3B** for grounded explanation generation.

## Pipeline

### Offline job pipeline

```text
Job APIs / Greenhouse boards
        ↓
SQLite jobs database
        ↓
HTML/text preprocessing
        ↓
Qwen3-Embedding-0.6B (local)
        ↓
1024-d float32 embeddings
        ↓
ChromaDB HNSW index
```

### Online matching pipeline

```text
Resume
  ↓
Qwen3 embedding
  ↓
metadata filters
  ↓
HNSW dense retrieval (top 100)
  ↓
Cohere cross-encoder reranking (top 10)
  ↓
local Llama requirement/evidence generation
  ↓
grounded fit explanations
```

## Tech Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| Job collection | Greenhouse + multiple job APIs | Build the job corpus |
| Storage | SQLite | Raw jobs, cleaned jobs, serialized embeddings |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` | 1024-dimensional dense vectors |
| Vector retrieval | ChromaDB + HNSW | Approximate nearest-neighbor semantic search |
| Reranking | Cohere `rerank-english-v3.0` | Cross-encoder relevance scoring |
| Local LLM | Ollama + `llama3.2:3b-instruct-q4_K_M` | Requirement extraction and fit explanations |
| API | FastAPI | Matching and health endpoints |
| Evaluation | Precision@5 / Recall@10 scripts + MLflow | Held-out retrieval evaluation |

## Retrieval Configuration

Current defaults from `src/config.py`:

```text
Embedding model: Qwen/Qwen3-Embedding-0.6B
Embedding dimension: 1024
Retrieve top K: 100
HNSW ef_construction: 400
HNSW ef: 400
Rerank top N: 10
```

The system intentionally retrieves a broad candidate set first and then applies a more expensive reranker to the top candidates.

## Evaluation

The repository contains tune/test evaluation scripts that measure:

- **Precision@5** — how many of the top five returned jobs are known relevant positives.
- **Recall@10** — how many known relevant positives appear in the top ten.

The evaluation uses held-out resume/job examples with synthetic positive jobs inserted into a corpus of sampled real jobs. Metrics should be reported only after running the current Qwen-based evaluation; historical values from the previous embedding backend are not treated as current results.

Run the held-out test evaluation with reranking:

```bash
python scripts/eval/run_test_eval.py --db-path data/jobs.db
```

Run dense-retrieval-only evaluation:

```bash
python scripts/eval/run_test_eval.py --db-path data/jobs.db --skip-rerank
```

The scripts write result and diagnostic files under the evaluation results directory and log the aggregate Precision@5 and Recall@10 values.

## Setup

### 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Local Qwen embeddings do **not** require an embedding API key. `COHERE_API_KEY` is required for reranking. Job-collection providers may require their own API keys depending on which collectors you enable.

### 3. Pull the local LLM

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
```

### 4. Collect or provide jobs

The normal job database path is:

```text
data/jobs.db
```

Job collection is orchestrated by:

```bash
python data_pipeline/collect_jobs.py
```

### 5. Preprocess jobs

```bash
python scripts/pipeline/preprocess_jobs.py --db-path data/jobs.db
```

### 6. Embed jobs locally

```bash
python scripts/pipeline/embed_jobs.py --db-path data/jobs.db
```

This loads Qwen3-Embedding-0.6B through Sentence Transformers, embeds unembedded job descriptions, and stores the vectors in SQLite.

### 7. Run resume matching

```bash
python scripts/pipeline/match_jobs.py --db-path data/jobs.db
```

The default resume source is:

```text
data/user_profile.txt
```

The generated result file is:

```text
matched_jobs.md
```

## Important Modules

```text
src/embedding.py
    Local Qwen embedding model loading and batch embedding

src/retrieval.py
    Chroma collection construction and HNSW retrieval

src/reranking.py
    Cohere cross-encoder reranking

src/regex_extraction.py
    Deterministic metadata extraction and filter construction

src/llm_extraction.py
    Optional local LLM fallback for metadata extraction

src/generation.py
    Grounded requirement/evidence matching and explanation generation

scripts/pipeline/embed_jobs.py
    Offline job embedding pipeline

scripts/pipeline/match_jobs.py
    End-to-end resume matching pipeline

scripts/eval/run_tuning_eval.py
    Tune-set retrieval evaluation

scripts/eval/run_test_eval.py
    Held-out test-set retrieval evaluation
```

## API

The FastAPI application exposes:

### `GET /health`

Simple liveness check.

### `GET /ready`

Checks:

- SQLite connectivity
- ChromaDB availability
- local Qwen embedding inference
- Ollama/model availability

### `POST /match`

Embeds the resume locally, retrieves candidate jobs, reranks them, and returns matched jobs with explanations.

## Design Notes

### Why HNSW?

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest-neighbor index. It provides fast high-recall vector retrieval without exhaustively comparing the resume embedding against every job vector.

### Why reranking after embeddings?

Dense retrieval is fast and scalable, but each document is represented independently. A cross-encoder reranker evaluates the resume and candidate job together, which gives a stronger second-stage relevance signal over a much smaller candidate set.

### Why local embeddings and local generation?

Running Qwen embeddings and Llama generation locally reduces external API dependency, avoids per-embedding API cost, and keeps the core semantic representation and explanation generation under local control.

## Current Scale

The pipeline has been run end-to-end over **3.2K+ job postings**, including local Qwen embedding, HNSW indexing, top-100 retrieval, cross-encoder reranking, and local LLM generation.

## Testing

Run the test suite with:

```bash
pytest
```

For evaluation-specific checks:

```bash
pytest tests/test_eval_embedding_cache.py tests/test_metrics.py tests/test_retrieval.py
```

## Output Quality

Retrieval/reranking and generation are intentionally separated. A job may be retrieved correctly even when the local explanation stage produces weak or unvalidated evidence. Retrieval metrics should therefore be measured independently from generation quality.
