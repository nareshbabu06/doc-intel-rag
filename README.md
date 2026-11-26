# DocIntel-RAG: Multi-Modal Document Intelligence & Async RAG Pipeline

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-v1.16%2B-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Celery](https://img.shields.io/badge/Celery-5.4%2B-37814A?logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

A production-grade, layout-aware document intelligence and citation-grounded Retrieval-Augmented Generation (RAG) system engineered for dense technical PDFs, hardware datasheets, and scanned engineering documents.

---

## Problem Statement

Conventional RAG pipelines flatten PDFs into raw character streams using naive fixed-width text splitters. This approach fails catastrophically on technical documents:
- **Table Corruption**: Multi-column electrical characteristics and register maps are shredded across unrelated chunks, stripping headers from numeric values.
- **Duplicate Ingestion**: Naive hybrid extractors index table contents twice—once as raw text and again inside table structures.
- **Blind Scanned Pages**: Native PDF parsers return empty strings on rasterized schematics or scanned annexes.
- **Keyword Blindness**: Pure dense vector similarity often misses exact part numbers (e.g., `LM5176`, `TPS54302`), register mnemonics (`TJ_MAX`), and pin identifiers.
- **Blocking Event Loops**: Running OCR and transformer inference synchronously inside HTTP request handlers starves web workers and triggers gateway timeouts.

**DocIntel-RAG** solves these challenges via spatial bounding-box deduplication, automatic 200 DPI Tesseract OCR fallback, non-blocking Celery/Redis background ingestion, Hybrid BM25 + Dense RRF fusion with Cross-Encoder reranking, and strict `[Source N]` citation grounding.

---

## System Architecture

```text
                          +-----------------------+
                          |     Client / User     |
                          +-----------+-----------+
                                      |
             +------------------------+------------------------+
             | POST /documents/upload                          | POST /query
             | GET  /documents/status/{task_id}                | DELETE /documents/{doc_id}
             v                                                 v
+------------+-------------------------------------------------+------------+
|                         FastAPI Application (api)                         |
|  - Magic-byte (%PDF-) & size validation    - Threadpool offload (to_thread)|
|  - Streaming 64KB chunked upload           - Structured error handling    |
+------------+------------------------+------------------------+------------+
             |                        |                        |
             | Write PDF              | Enqueue Task           | Hybrid Search &
             v                        v                        | Citation Synthesis
+------------+------------+  +--------+--------+               |
|  Shared Upload Volume   |  |  Redis Broker   |               |
|  (/app/data/uploads)    |  |  & Result Store |               |
+------------+------------+  +--------+--------+               |
             ^                        |                        |
             | Read PDF               | Consume                |
             |                        v                        |
+------------+------------------------+------------+           |
|           Celery Worker Pool (worker)            |           |
|  1. pdfplumber: Table extraction -> Markdown     |           |
|  2. PyMuPDF: Text blocks & IoU table dedup       |           |
|  3. Tesseract OCR: 200 DPI fallback (<30 chars)  |           |
|  4. BGE-small-en-v1.5 normalized embeddings      |           |
+------------------------+-------------------------+           |
                         | Upsert PointStructs                 |
                         v                                     v
             +-----------+-------------------------------------+-----------+
             |              Qdrant Vector Database (qdrant)                |
             |  - Collection: doc_chunks (384-d Cosine + doc_id index)     |
             |  - Dense query_points() + BM25Okapi -> RRF -> CrossEncoder  |
             +-------------------------------------------------------------+
```

---

## Engineering Design Decisions

### 1. Why RRF + Cross-Encoder Reranking Beats Naive Cosine Search
- **Lexical Precision vs. Semantic Recall**: Bi-encoder embeddings (`BAAI/bge-small-en-v1.5`) capture high-level intent (e.g., *"over-temperature protection"*) but dilute exact alphanumeric identifiers like `LM5176`, `VCC_CORE`, or `ECE-9021-REV_B`. Pairing Qdrant dense retrieval (`query_points`) with in-document `BM25Okapi` scoring guarantees exact technical token recall.
- **Tuned Reciprocal Rank Fusion ($k=50$)**: Combining ranks via $\text{RRF}(d) = \frac{1}{50 + r_{\text{dense}}} + \frac{1}{50 + r_{\text{BM25}}}$ preserves top-ranked lexical and semantic hits without requiring fragile score normalization across disparate scales.
- **Cross-Encoder Reranking**: Bi-encoders encode query and chunk independently. Passing the top 15 RRF candidates through `cross-encoder/ms-marco-MiniLM-L-6-v2` enables full token-level cross-attention between the query and candidate text (or Markdown table rows), dramatically improving top-1 precision before LLM synthesis.

### 2. Why Layout-Aware Bounding Boxes Prevent Table Corruption
- **Atomic Table Preservation**: `pdfplumber` detects bordered and aligned tables first, serializing them into atomic GitHub-Flavored Markdown (`| Col | ... |`) chunks so column headers remain bound to their corresponding values.
- **Spatial IoU Deduplication**: PyMuPDF (`fitz`) extracts raw narrative text blocks (`block_type == 0`). Every candidate text block is compared against extracted table `BoundingBox` regions on the same page; any block whose IoU or relative intersection area exceeds `0.45` is discarded, eliminating duplicated/garbled table cells from narrative chunks.
- **Traceable Spatial Citations**: Consecutive narrative blocks merge up to `450` estimated tokens while computing enclosing `.union()` bounding boxes. Every RAG citation returns exact `(page_num, bbox[x0, y0, x1, y1], chunk_type)` coordinates for UI PDF highlighting.

---

## Quickstart

### Option A: Docker Compose (Recommended Production Stack)
The multi-stage [`Dockerfile`](Dockerfile) pre-warms both HuggingFace models at build time (`HF_HOME=/app/.cache/huggingface`) and runs as a non-root user (`appuser`).

```bash
# 1. Copy environment configuration (optional: set LLM_API_KEY for OpenAI-compatible synthesis)
cp .env.example .env

# 2. Build and launch Redis, Qdrant, FastAPI, and Celery worker
docker compose up --build -d

# 3. Verify all services are healthy
docker compose ps
```

> **Zero-Cost Local Fallback**: If `LLM_API_KEY` is left empty, the generator automatically executes an extractive summary citing `[Source N]` tags with full bounding boxes—allowing end-to-end testing with zero external API costs.

### Option B: Local Development Setup
Ensure Python 3.11+ and system `tesseract-ocr` are installed on your host machine.

```bash
# 1. Create and activate virtual environment
python -m venv venv
# Windows PowerShell: .\venv\Scripts\Activate.ps1
# Linux / macOS:      source venv/bin/activate

# 2. Install dependencies and configure environment
pip install -r requirements.txt
cp .env.example .env

# 3. Run unit & integration test suite
pytest -v

# 4. Start API (automatically falls back to embedded local Qdrant if remote Qdrant is offline)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. (Optional, requires Redis) Start Celery worker in a separate terminal
celery -A app.workers.celery_app worker --loglevel=info --concurrency=2
```

---

## API Walkthrough (`curl`)

### 1. Health Check
```bash
curl -s http://localhost:8000/api/v1/health | jq .
```

### 2. Upload a Multi-Page PDF (202 Accepted)
```bash
curl -s -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@data/uploads/sample.pdf" | jq .
```
Response:
```json
{
  "doc_id": "9f8c2a1b4e7d40219c3e5a6f1d2b3c4d",
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "sample.pdf",
  "status": "QUEUED"
}
```

### 3. Poll Ingestion Progress
```bash
curl -s "http://localhost:8000/api/v1/documents/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890" | jq .
```
Response during indexing / on completion:
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "state": "SUCCESS",
  "meta": {
    "doc_id": "9f8c2a1b4e7d40219c3e5a6f1d2b3c4d",
    "status": "COMPLETED",
    "pages_processed": 45,
    "chunks_indexed": 104,
    "progress": 100
  },
  "error": null
}
```

### 4. Query with Hybrid Retrieval & Spatial Citations
```bash
curl -s -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "9f8c2a1b4e7d40219c3e5a6f1d2b3c4d",
    "question": "What are the recommended supply voltage limits?",
    "top_k": 5
  }' | jq .
```

### 5. Delete Document & Vectors
```bash
curl -s -X DELETE "http://localhost:8000/api/v1/documents/9f8c2a1b4e7d40219c3e5a6f1d2b3c4d" | jq .
```

---

## Performance Benchmarks

Run the reproducible benchmark suite against synthetic multi-page technical PDFs containing mixed narrative text and 4-column tables:

```bash
python -m scripts.benchmark
```

### Measured Baseline Results

#### Hardware Profile
- **OS**: Windows 11 (AMD64)
- **Processor**: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- **CPU Cores**: 12
- **Python**: 3.14.6
- **Embedding Model**: `BAAI/bge-small-en-v1.5`
- **Reranker Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

#### Document Ingestion & Indexing Benchmark
| Document Size | Extracted Chunks | Parse Time (s) | Parsing Throughput (pages/s) | Total Index Time (ms) | Indexing Latency (ms/chunk) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 5 pages | 10 | 0.12 | 40.46 | 294.2 | 29.42 |
| 15 pages | 30 | 0.46 | 32.63 | 805.5 | 26.85 |
| 30 pages | 60 | 0.57 | 52.55 | 1499.6 | 24.99 |

#### Hybrid Retrieval + Cross-Encoder Reranking Latency (50 Queries @ 30-Page Doc)
| Metric | Latency (ms) |
| :--- | :---: |
| Mean | 237.13 |
| **p50 (Median)** | **237.33** |
| **p95** | **280.61** |
| **p99** | **299.86** |
