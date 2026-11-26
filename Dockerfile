# Stage 1: Build virtual environment and install Python dependencies
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Lean production runtime image
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME="/app/.cache/huggingface"

WORKDIR /app

# Install runtime system dependencies (Tesseract OCR, OpenGL/GLib for PDF rendering, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Build-time ML model warmup to eliminate container cold-start download latency
RUN mkdir -p /app/.cache/huggingface && \
    python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application source
COPY app/ /app/app/
COPY scripts/ /app/scripts/
COPY .env.example /app/.env.example

# Ensure writable data directories and drop root privileges
RUN mkdir -p /app/data/uploads /app/data/qdrant_db && \
    groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
