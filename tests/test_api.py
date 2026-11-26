from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest
from qdrant_client import QdrantClient

from app.api.schemas import (
    BoundingBox,
    DeleteResponse,
    DocumentChunk,
    QueryResponse,
    TaskStatusResponse,
    UploadResponse,
)
from app.core.config import get_settings
from app.main import app
from app.services.generator import RAGGenerator
from app.services.retriever import HybridRetriever


@pytest.fixture
def isolated_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Overrides UPLOAD_DIR with a temporary directory for isolated API upload tests."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    monkeypatch.setattr(settings, "UPLOAD_DIR", upload_dir)
    return upload_dir


def test_upload_rejects_invalid_extension(isolated_upload_dir: Path) -> None:
    """Verifies that non-PDF file extensions are rejected with 400 Bad Request."""
    settings = get_settings()
    with TestClient(app) as client:
        response = client.post(
            f"{settings.API_V1_PREFIX}/documents/upload",
            files={"file": ("notes.txt", b"plain text content", "text/plain")},
        )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "InvalidFileError"


def test_upload_rejects_invalid_magic_bytes(isolated_upload_dir: Path) -> None:
    """Verifies that a .pdf file lacking %PDF- magic bytes is rejected with 400 Bad Request."""
    settings = get_settings()
    with TestClient(app) as client:
        response = client.post(
            f"{settings.API_V1_PREFIX}/documents/upload",
            files={"file": ("fake.pdf", b"NOT_A_PDF_HEADER_CONTENT", "application/pdf")},
        )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "InvalidFileError"
    assert list(isolated_upload_dir.glob("*.pdf")) == []


def test_upload_rejects_oversized_pdf(
    isolated_upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that uploads exceeding MAX_UPLOAD_MB return 413 Payload Too Large."""
    settings = get_settings()
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    oversized_bytes = b"%PDF-1.4\n" + (b"A" * (1024 * 1024 + 128))

    with TestClient(app) as client:
        response = client.post(
            f"{settings.API_V1_PREFIX}/documents/upload",
            files={"file": ("large.pdf", oversized_bytes, "application/pdf")},
        )
    assert response.status_code == 413
    payload = response.json()
    assert payload["error"] == "FileSizeExceededError"
    assert list(isolated_upload_dir.glob("*.pdf")) == []


def test_upload_valid_pdf_returns_202(isolated_upload_dir: Path) -> None:
    """Verifies valid PDF upload returns 202 Accepted, stores file, and dispatches Celery task."""
    settings = get_settings()
    mock_task = MagicMock()
    mock_task.id = "celery-task-uuid-1234"
    sample_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"

    with patch("app.api.routes.process_document_task.delay", return_value=mock_task) as mock_delay:
        with TestClient(app) as client:
            response = client.post(
                f"{settings.API_V1_PREFIX}/documents/upload",
                files={"file": ("datasheet.pdf", sample_pdf_bytes, "application/pdf")},
            )

    assert response.status_code == 202
    validated = UploadResponse.model_validate(response.json())
    assert validated.task_id == "celery-task-uuid-1234"
    assert validated.filename == "datasheet.pdf"
    assert validated.status == "QUEUED"
    assert len(validated.doc_id) > 0

    expected_saved_path = isolated_upload_dir / f"{validated.doc_id}.pdf"
    assert expected_saved_path.is_file()
    mock_delay.assert_called_once_with(
        doc_id=validated.doc_id,
        file_path=str(expected_saved_path),
    )


def test_document_status_polling_endpoint() -> None:
    """Verifies task status endpoint exposes Celery PARSING/INDEXING meta and FAILURE errors."""
    settings = get_settings()

    mock_async_result = MagicMock()
    mock_async_result.state = "INDEXING"
    mock_async_result.info = {
        "step": "Generating embeddings and indexing in Qdrant",
        "progress": 70,
        "chunks_extracted": 14,
    }
    mock_async_result.failed.return_value = False

    with patch("app.api.routes.AsyncResult", return_value=mock_async_result):
        with TestClient(app) as client:
            response = client.get(f"{settings.API_V1_PREFIX}/documents/status/task-abc-789")

    assert response.status_code == 200
    status_resp = TaskStatusResponse.model_validate(response.json())
    assert status_resp.task_id == "task-abc-789"
    assert status_resp.state == "INDEXING"
    assert status_resp.meta is not None
    assert status_resp.meta["progress"] == 70
    assert status_resp.error is None

    # Verify FAILURE state serialization
    mock_failed_result = MagicMock()
    mock_failed_result.state = "FAILURE"
    mock_failed_result.info = RuntimeError("Corrupt PDF stream")
    mock_failed_result.failed.return_value = True

    with patch("app.api.routes.AsyncResult", return_value=mock_failed_result):
        with TestClient(app) as client:
            fail_response = client.get(f"{settings.API_V1_PREFIX}/documents/status/task-err-001")

    assert fail_response.status_code == 200
    fail_status = TaskStatusResponse.model_validate(fail_response.json())
    assert fail_status.state == "FAILURE"
    assert fail_status.meta is None
    assert fail_status.error == "Corrupt PDF stream"


def test_query_and_delete_contract(
    tmp_path: Path,
    isolated_upload_dir: Path,
) -> None:
    """Verifies /query response schema, 404 on unknown doc_id, and DELETE /documents/{doc_id} cleanup."""
    settings = get_settings()
    qdrant_client = QdrantClient(path=str(tmp_path / "qdrant_api_test"))
    retriever = HybridRetriever(client=qdrant_client, collection_name="api_test_chunks")
    generator = RAGGenerator(api_key=None)

    doc_id = "api_doc_001"
    dummy_pdf = isolated_upload_dir / f"{doc_id}.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n%%EOF")

    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}_p1_c1",
            doc_id=doc_id,
            page_num=1,
            chunk_type="text",
            content="The TPS54302 buck converter supports an input voltage range of 4.5V to 28V with 3A continuous output.",
            bbox=BoundingBox(x0=40.0, y0=100.0, x1=520.0, y1=145.0),
            token_estimate=20,
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_p2_c1",
            doc_id=doc_id,
            page_num=2,
            chunk_type="table",
            content="| Parameter | Min | Typ | Max |\n| --- | --- | --- | --- |\n| Switching Freq (kHz) | 340 | 400 | 460 |",
            bbox=BoundingBox(x0=40.0, y0=200.0, x1=520.0, y1=280.0),
            token_estimate=30,
        ),
    ]
    retriever.index_chunks(chunks)

    with (
        patch("app.api.routes.get_retriever", return_value=retriever),
        patch("app.api.routes.get_generator", return_value=generator),
    ):
        with TestClient(app) as client:
            # 1. Valid query returning QueryResponse with full Citation objects
            query_resp = client.post(
                f"{settings.API_V1_PREFIX}/query",
                json={
                    "doc_id": doc_id,
                    "question": "What is the input voltage range of TPS54302?",
                    "top_k": 2,
                },
            )
            assert query_resp.status_code == 200
            validated_query = QueryResponse.model_validate(query_resp.json())
            assert validated_query.doc_id == doc_id
            assert "[Source 1]" in validated_query.answer
            assert validated_query.retrieval_ms >= 0.0
            assert validated_query.generation_ms >= 0.0
            assert len(validated_query.citations) == 2

            top_citation = validated_query.citations[0]
            assert top_citation.chunk_id == f"{doc_id}_p1_c1"
            assert top_citation.page_num == 1
            assert top_citation.bbox.x0 < top_citation.bbox.x1
            assert "TPS54302" in top_citation.snippet
            assert isinstance(top_citation.relevance_score, float)

            # 2. Unknown doc_id returns 404 Not Found
            missing_resp = client.post(
                f"{settings.API_V1_PREFIX}/query",
                json={
                    "doc_id": "nonexistent_doc",
                    "question": "What is the operating voltage?",
                    "top_k": 3,
                },
            )
            assert missing_resp.status_code == 404
            assert missing_resp.json()["error"] == "DocumentNotFoundError"

            # 3. Delete document removes vectors and PDF file
            del_resp = client.delete(f"{settings.API_V1_PREFIX}/documents/{doc_id}")
            assert del_resp.status_code == 200
            validated_del = DeleteResponse.model_validate(del_resp.json())
            assert validated_del.doc_id == doc_id
            assert validated_del.status == "DELETED"
            assert not dummy_pdf.exists()

            # 4. Subsequent query on deleted doc_id returns 404
            after_del_resp = client.post(
                f"{settings.API_V1_PREFIX}/query",
                json={
                    "doc_id": doc_id,
                    "question": "What is the input voltage range?",
                    "top_k": 2,
                },
            )
            assert after_del_resp.status_code == 404


def test_process_document_task_reports_progress(tmp_path: Path) -> None:
    """Verifies Celery background task reports PARSING and INDEXING states and completes."""
    from app.workers.tasks import process_document_task

    sample_chunks = [
        DocumentChunk(
            chunk_id="task_doc_p1_c1",
            doc_id="task_doc",
            page_num=1,
            chunk_type="text",
            content="Sample telemetry content.",
            bbox=BoundingBox(x0=10.0, y0=10.0, x1=200.0, y1=50.0),
            token_estimate=5,
        ),
        DocumentChunk(
            chunk_id="task_doc_p2_c1",
            doc_id="task_doc",
            page_num=2,
            chunk_type="text",
            content="Second page voltage telemetry.",
            bbox=BoundingBox(x0=10.0, y0=10.0, x1=200.0, y1=50.0),
            token_estimate=6,
        ),
    ]

    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_document.return_value = sample_chunks
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.index_chunks.return_value = 2

    with (
        patch("app.workers.tasks.DocumentParser", return_value=mock_parser_instance),
        patch("app.workers.tasks.get_retriever", return_value=mock_retriever_instance),
        patch.object(process_document_task, "update_state") as mock_update_state,
    ):
        result = process_document_task.run(
            doc_id="task_doc",
            file_path=str(tmp_path / "task_doc.pdf"),
        )

    assert result == {
        "doc_id": "task_doc",
        "status": "COMPLETED",
        "pages_processed": 2,
        "chunks_indexed": 2,
        "progress": 100,
    }
    assert mock_update_state.call_count == 2
    assert mock_update_state.call_args_list[0].kwargs["state"] == "PARSING"
    assert mock_update_state.call_args_list[0].kwargs["meta"]["progress"] == 25
    assert mock_update_state.call_args_list[1].kwargs["state"] == "INDEXING"
    assert mock_update_state.call_args_list[1].kwargs["meta"]["progress"] == 70
    assert mock_update_state.call_args_list[1].kwargs["meta"]["chunks_extracted"] == 2
