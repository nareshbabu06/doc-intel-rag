from __future__ import annotations

import os
import time
from typing import Any

import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")
RENDER_ZOOM = 2.0  # 144 DPI crisp rendering (72 * 2.0)

st.set_page_config(
    page_title="DocIntel RAG — Visual Citation Viewer",
    layout="wide",
)


def init_session_state() -> None:
    """Initializes persistent Streamlit session state keys."""
    defaults: dict[str, Any] = {
        "pdf_bytes": None,
        "total_pages": 0,
        "doc_id": None,
        "filename": None,
        "query_result": None,
        "active_citation": None,
        "page_selector": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Returns the total number of pages in an in-memory PDF buffer."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


def _select_citation(citation: dict[str, Any]) -> None:
    """Callback that activates a citation and jumps the PDF viewer to its page."""
    st.session_state["active_citation"] = citation
    st.session_state["page_selector"] = int(citation["page_num"])


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    outline: tuple[int, int, int, int],
    width: int = 2,
    dash: int = 8,
    gap: int = 5,
) -> None:
    """Draws a crisp dashed rectangle outline onto a PIL RGBA overlay."""
    edges = [
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ]
    step = dash + gap
    for (sx, sy), (ex, ey) in edges:
        dx = ex - sx
        dy = ey - sy
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 0:
            continue
        ux, uy = dx / length, dy / length
        pos = 0.0
        while pos < length:
            seg_end = min(pos + dash, length)
            draw.line(
                [
                    (sx + ux * pos, sy + uy * pos),
                    (sx + ux * seg_end, sy + uy * seg_end),
                ],
                fill=outline,
                width=width,
            )
            pos += step


def render_annotated_page(
    pdf_bytes: bytes,
    page_num: int,
    citations: list[dict[str, Any]],
    active_citation: dict[str, Any] | None,
    zoom: float = RENDER_ZOOM,
) -> Image.Image:
    """Renders a PDF page at high DPI and composites citation bounding-box overlays."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        safe_page = max(1, min(page_num, doc.page_count))
        page = doc[safe_page - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        base_img = Image.frombytes(
            "RGB", (pixmap.width, pixmap.height), pixmap.samples
        ).convert("RGBA")
    finally:
        doc.close()

    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    active_chunk_id = active_citation.get("chunk_id") if active_citation else None

    # 1. Subtle dashed/tinted outlines for all non-active citations on the current page
    for citation in citations:
        if int(citation["page_num"]) != page_num:
            continue
        if active_chunk_id and citation.get("chunk_id") == active_chunk_id:
            continue

        bbox = citation["bbox"]
        x0 = float(bbox["x0"]) * zoom
        y0 = float(bbox["y0"]) * zoom
        x1 = float(bbox["x1"]) * zoom
        y1 = float(bbox["y1"]) * zoom

        draw.rectangle(
            [(x0, y0), (x1, y1)],
            fill=(59, 130, 246, 28),
        )
        _draw_dashed_rect(
            draw,
            x0,
            y0,
            x1,
            y1,
            outline=(37, 99, 235, 190),
            width=2,
        )

    # 2. Bold semi-transparent highlight box for the selected citation
    if active_citation and int(active_citation["page_num"]) == page_num:
        bbox = active_citation["bbox"]
        x0 = float(bbox["x0"]) * zoom
        y0 = float(bbox["y0"]) * zoom
        x1 = float(bbox["x1"]) * zoom
        y1 = float(bbox["y1"]) * zoom

        draw.rectangle(
            [(x0, y0), (x1, y1)],
            fill=(255, 215, 0, 75),
            outline=(255, 107, 0, 255),
            width=3,
        )

    return Image.alpha_composite(base_img, overlay)


def ingest_pdf_with_progress(filename: str, pdf_bytes: bytes) -> None:
    """Uploads PDF to FastAPI and polls Celery worker status until completion."""
    try:
        upload_resp = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files={"file": (filename, pdf_bytes, "application/pdf")},
            timeout=30,
        )
        if upload_resp.status_code != 202:
            st.error(f"Upload rejected ({upload_resp.status_code}): {upload_resp.text}")
            return

        upload_data = upload_resp.json()
        doc_id: str = upload_data["doc_id"]
        task_id: str = upload_data["task_id"]
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_BASE_URL}: {exc}")
        return

    with st.status(f"Processing '{filename}'...", expanded=True) as status_box:
        progress_bar = st.progress(5)
        st.write(f"Task queued (`{task_id}`)")

        while True:
            try:
                status_resp = requests.get(
                    f"{API_BASE_URL}/documents/status/{task_id}",
                    timeout=10,
                )
                status_resp.raise_for_status()
                payload = status_resp.json()
            except requests.RequestException as exc:
                status_box.update(label="Status polling failed", state="error")
                st.error(f"Polling error: {exc}")
                return

            state: str = payload.get("state", "PENDING")
            meta: dict[str, Any] = payload.get("meta") or {}
            error_msg: str | None = payload.get("error")

            if state == "PARSING":
                pct = int(meta.get("progress", 25))
                progress_bar.progress(pct)
                status_box.update(label=f"PARSING ({pct}%): {meta.get('step', 'Parsing PDF')}")
            elif state == "INDEXING":
                pct = int(meta.get("progress", 70))
                chunks_n = meta.get("chunks_extracted", "?")
                progress_bar.progress(pct)
                status_box.update(
                    label=f"INDEXING ({pct}%): {meta.get('step', 'Indexing')} ({chunks_n} chunks)"
                )
            elif state == "SUCCESS" or meta.get("status") == "COMPLETED":
                progress_bar.progress(100)
                chunks_idx = meta.get("chunks_indexed", "?")
                pages_n = meta.get("pages_processed", "?")
                status_box.update(
                    label=f"Completed: {pages_n} pages, {chunks_idx} chunks indexed",
                    state="complete",
                    expanded=False,
                )
                st.session_state["doc_id"] = doc_id
                st.session_state["filename"] = filename
                st.session_state["query_result"] = None
                st.session_state["active_citation"] = None
                st.session_state["page_selector"] = 1
                return
            elif state == "FAILURE":
                status_box.update(label="Ingestion failed", state="error")
                st.error(f"Worker failure: {error_msg or 'Unknown error'}")
                return

            time.sleep(1)


def delete_active_document(doc_id: str) -> None:
    """Deletes the active document from Qdrant and upload storage."""
    try:
        resp = requests.delete(f"{API_BASE_URL}/documents/{doc_id}", timeout=15)
        resp.raise_for_status()
        st.session_state["doc_id"] = None
        st.session_state["filename"] = None
        st.session_state["query_result"] = None
        st.session_state["active_citation"] = None
        st.success(f"Deleted document `{doc_id}`.")
    except requests.RequestException as exc:
        st.error(f"Delete failed: {exc}")


def main() -> None:
    """Main Streamlit application entrypoint."""
    init_session_state()

    st.title("DocIntel RAG — Visual Citation Viewer")
    st.caption(
        "Layout-aware multi-modal PDF intelligence with hybrid BM25 + dense RRF retrieval "
        "and spatial bounding-box citation grounding."
    )

    with st.sidebar:
        st.header("Document Ingestion")
        uploaded_file = st.file_uploader(
            "Upload a technical PDF",
            type=["pdf"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            st.session_state["pdf_bytes"] = raw_bytes
            st.session_state["total_pages"] = get_pdf_page_count(raw_bytes)
            st.caption(
                f"Loaded locally: `{uploaded_file.name}` "
                f"({st.session_state['total_pages']} pages)"
            )

            if st.button("Upload & Index Document", type="primary", use_container_width=True):
                ingest_pdf_with_progress(uploaded_file.name, raw_bytes)

        st.divider()
        st.subheader("Active Index")
        if st.session_state["doc_id"]:
            st.success(f"Ready: `{st.session_state['filename']}`")
            st.code(st.session_state["doc_id"], language="text")
            if st.button("Delete Indexed Document", use_container_width=True):
                delete_active_document(st.session_state["doc_id"])
        else:
            st.info("No document indexed yet. Upload a PDF to begin querying.")

        st.caption(f"Backend: `{API_BASE_URL}`")

    left_col, right_col = st.columns([1.1, 1.0], gap="large")

    # --- Left Column: Chat & Cited Answers ---
    with left_col:
        st.subheader("Ask & Verify")

        with st.form("query_form"):
            question = st.text_area(
                "Ask a question about the indexed document",
                placeholder="e.g., What are the recommended operating supply voltage limits?",
                height=90,
            )
            top_k = st.slider(
                "Top-K Reranked Sources",
                min_value=1,
                max_value=10,
                value=5,
            )
            submitted = st.form_submit_button("Run Hybrid RAG Query", type="primary")

        if submitted:
            if not st.session_state["doc_id"]:
                st.warning("Please upload and index a PDF document in the sidebar first.")
            elif len(question.strip()) < 3:
                st.warning("Please enter a question of at least 3 characters.")
            else:
                with st.spinner("Running hybrid retrieval, RRF fusion & Cross-Encoder reranking..."):
                    try:
                        query_resp = requests.post(
                            f"{API_BASE_URL}/query",
                            json={
                                "doc_id": st.session_state["doc_id"],
                                "question": question.strip(),
                                "top_k": top_k,
                            },
                            timeout=60,
                        )
                        if query_resp.status_code != 200:
                            st.error(
                                f"Query error ({query_resp.status_code}): {query_resp.text}"
                            )
                        else:
                            result_data = query_resp.json()
                            st.session_state["query_result"] = result_data
                            citations = result_data.get("citations", [])
                            if citations:
                                _select_citation(citations[0])
                            else:
                                st.session_state["active_citation"] = None
                    except requests.RequestException as exc:
                        st.error(f"Query request failed: {exc}")

        query_result: dict[str, Any] | None = st.session_state["query_result"]
        if query_result:
            m1, m2 = st.columns(2)
            m1.metric("Retrieval + Rerank", f"{query_result['retrieval_ms']:.1f} ms")
            m2.metric("Answer Generation", f"{query_result['generation_ms']:.1f} ms")

            st.markdown("### Answer")
            st.markdown(query_result["answer"])

            citations: list[dict[str, Any]] = query_result.get("citations", [])
            if citations:
                st.markdown("### Grounded Citations")
                active_id = (
                    st.session_state["active_citation"].get("chunk_id")
                    if st.session_state["active_citation"]
                    else None
                )
                for cit in citations:
                    is_active = active_id == cit["chunk_id"]
                    label = (
                        f"[Source {cit['source_index']} — Page {cit['page_num']} "
                        f"({cit['chunk_type']}) | Score: {cit['relevance_score']:.2f}]"
                    )
                    st.button(
                        label,
                        key=f"cite_btn_{cit['source_index']}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                        on_click=_select_citation,
                        args=(cit,),
                    )
                    with st.expander(
                        f"Preview Source {cit['source_index']} ({cit['chunk_id']})",
                        expanded=is_active,
                    ):
                        st.markdown(cit["snippet"])

    # --- Right Column: Live PDF Page Canvas with Bounding-Box Overlay ---
    with right_col:
        st.subheader("PDF Spatial Evidence Viewer")

        pdf_bytes: bytes | None = st.session_state["pdf_bytes"]
        total_pages: int = st.session_state["total_pages"]

        if pdf_bytes is None or total_pages == 0:
            st.info("Upload a PDF in the sidebar to inspect pages and citation bounding boxes.")
        else:
            selected_page = int(
                st.number_input(
                    f"Page (1 to {total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    step=1,
                    key="page_selector",
                )
            )

            all_citations: list[dict[str, Any]] = (
                st.session_state["query_result"].get("citations", [])
                if st.session_state["query_result"]
                else []
            )
            active_citation: dict[str, Any] | None = st.session_state["active_citation"]

            annotated_img = render_annotated_page(
                pdf_bytes=pdf_bytes,
                page_num=selected_page,
                citations=all_citations,
                active_citation=active_citation,
                zoom=RENDER_ZOOM,
            )
            st.image(
                annotated_img,
                caption=f"Page {selected_page} of {total_pages} (144 DPI / zoom={RENDER_ZOOM}x)",
                use_container_width=True,
            )

            if active_citation:
                bbox = active_citation["bbox"]
                st.markdown(
                    f"**Active Highlight:** Source `{active_citation['source_index']}` · "
                    f"Chunk Type: `{active_citation['chunk_type']}` · "
                    f"Page `{active_citation['page_num']}`"
                )
                st.code(
                    f"BoundingBox(x0={bbox['x0']:.2f}, y0={bbox['y0']:.2f}, "
                    f"x1={bbox['x1']:.2f}, y1={bbox['y1']:.2f})",
                    language="python",
                )


if __name__ == "__main__":
    main()
