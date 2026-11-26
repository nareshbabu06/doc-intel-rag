import asyncio
import logging
from pathlib import Path
import time
import uuid

from celery.result import AsyncResult
from fastapi import APIRouter, File, UploadFile, status

from app.api.schemas import (
    DeleteResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TaskStatusResponse,
    UploadResponse,
)
from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    FileSizeExceededError,
    InvalidFileError,
)
from app.services.generator import get_generator
from app.services.retriever import get_retriever
from app.workers.celery_app import celery_app
from app.workers.tasks import process_document_task

logger = logging.getLogger("doc_intel.api.routes")

router = APIRouter()

UPLOAD_CHUNK_BYTES = 64 * 1024  # 64 KB streaming chunks
PDF_MAGIC_PREFIX = b"%PDF-"


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Returns application health status, version, and runtime environment."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Documents"],
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Validates and streams a PDF upload to disk, then dispatches async Celery ingestion."""
    settings = get_settings()
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise InvalidFileError(
            message="Only .pdf files are accepted",
            details={"filename": filename},
        )

    doc_id = uuid.uuid4().hex
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path: Path = settings.UPLOAD_DIR / f"{doc_id}.pdf"
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    total_bytes = 0
    is_first_chunk = True

    try:
        with saved_path.open("wb") as out_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break

                if is_first_chunk:
                    is_first_chunk = False
                    if not chunk.startswith(PDF_MAGIC_PREFIX):
                        raise InvalidFileError(
                            message="Uploaded file does not have a valid %PDF- signature",
                            details={"filename": filename},
                        )

                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise FileSizeExceededError(
                        message=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_MB} MB",
                        details={
                            "filename": filename,
                            "max_bytes": max_bytes,
                        },
                    )

                out_file.write(chunk)

        if total_bytes == 0:
            raise InvalidFileError(
                message="Uploaded PDF file is empty",
                details={"filename": filename},
            )
    except Exception:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    task = process_document_task.delay(doc_id=doc_id, file_path=str(saved_path))
    logger.info(
        "Queued ingestion task_id=%s for doc_id=%s (%s, %d bytes)",
        task.id,
        doc_id,
        filename,
        total_bytes,
    )

    return UploadResponse(
        doc_id=doc_id,
        task_id=str(task.id),
        filename=filename,
        status="QUEUED",
    )


@router.get(
    "/documents/status/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Documents"],
)
async def get_document_status(task_id: str) -> TaskStatusResponse:
    """Inspects Celery AsyncResult state and granular metadata for a background task."""
    result = AsyncResult(task_id, app=celery_app)
    state = result.state
    info = result.info

    return TaskStatusResponse(
        task_id=task_id,
        state=state,
        meta=info if isinstance(info, dict) else None,
        error=str(info) if result.failed() else None,
    )


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_document(request: QueryRequest) -> QueryResponse:
    """Executes hybrid retrieval and citation-grounded generation against an indexed document."""
    retriever = get_retriever()
    generator = get_generator()

    retrieval_start = time.perf_counter()
    ranked_chunks = await asyncio.to_thread(
        retriever.retrieve,
        doc_id=request.doc_id,
        query=request.question,
        top_k=request.top_k,
    )
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

    if not ranked_chunks:
        raise DocumentNotFoundError(
            message=f"No indexed chunks found for doc_id '{request.doc_id}'",
            details={"doc_id": request.doc_id},
        )

    generation_start = time.perf_counter()
    answer, citations = await generator.generate(
        doc_id=request.doc_id,
        question=request.question,
        ranked_chunks=ranked_chunks,
    )
    generation_ms = (time.perf_counter() - generation_start) * 1000.0

    return QueryResponse(
        doc_id=request.doc_id,
        answer=answer,
        citations=citations,
        retrieval_ms=round(retrieval_ms, 2),
        generation_ms=round(generation_ms, 2),
    )


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteResponse,
    tags=["Documents"],
)
async def delete_document(doc_id: str) -> DeleteResponse:
    """Deletes indexed document vectors from Qdrant and removes the stored PDF file."""
    settings = get_settings()
    retriever = get_retriever()
    await asyncio.to_thread(retriever.delete_document, doc_id)

    pdf_path = settings.UPLOAD_DIR / f"{doc_id}.pdf"
    if pdf_path.exists() and pdf_path.is_file():
        pdf_path.unlink()

    return DeleteResponse(
        doc_id=doc_id,
        status="DELETED",
        message=f"Document '{doc_id}' and associated vectors removed.",
    )
