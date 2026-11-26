import logging
from typing import Any

from celery import Task

from app.core.logging import setup_logging
from app.services.parser import DocumentParser
from app.services.retriever import get_retriever
from app.workers.celery_app import celery_app

setup_logging()
logger = logging.getLogger("doc_intel.workers.tasks")


@celery_app.task(bind=True, name="process_document_task")
def process_document_task(
    self: Task,
    doc_id: str,
    file_path: str,
) -> dict[str, Any]:
    """Background task that parses a PDF document and indexes its chunks into Qdrant."""
    logger.info("Starting document processing task for doc_id=%s (%s)", doc_id, file_path)
    try:
        self.update_state(
            state="PARSING",
            meta={"step": "Extracting layout, tables, and OCR", "progress": 25},
        )
        parser = DocumentParser()
        chunks = parser.parse_document(file_path=file_path, doc_id=doc_id)

        self.update_state(
            state="INDEXING",
            meta={
                "step": "Generating embeddings and indexing in Qdrant",
                "progress": 70,
                "chunks_extracted": len(chunks),
            },
        )
        retriever = get_retriever()
        count = retriever.index_chunks(chunks)

        max_page = max((chunk.page_num for chunk in chunks), default=0)
        result: dict[str, Any] = {
            "doc_id": doc_id,
            "status": "COMPLETED",
            "pages_processed": max_page,
            "chunks_indexed": count,
            "progress": 100,
        }
        logger.info(
            "Completed processing doc_id=%s: pages=%d, chunks=%d",
            doc_id,
            max_page,
            count,
        )
        return result
    except Exception:
        logger.exception("Document processing task failed for doc_id=%s", doc_id)
        raise
