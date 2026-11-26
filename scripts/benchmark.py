from __future__ import annotations

import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import fitz  # PyMuPDF
from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.services.parser import DocumentParser
from app.services.retriever import (
    HybridRetriever,
    get_embedding_model,
    get_reranker_model,
)


def generate_synthetic_pdf(output_path: Path, num_pages: int) -> None:
    """Generates a realistic multi-page technical PDF with paragraphs and 4-column tables."""
    doc = fitz.open()
    subsystems = ["PowerStage", "ClockTree", "DMAController", "ADCPipeline", "PCIePhy"]
    ics = ["LM5176", "TPS54302", "STM32H743", "ADS1256", "Si5351A"]

    for page_idx in range(1, num_pages + 1):
        page = doc.new_page(width=595, height=842)
        subsystem = subsystems[(page_idx - 1) % len(subsystems)]
        ic_name = ics[(page_idx - 1) % len(ics)]

        page.insert_text(
            fitz.Point(50, 55),
            f"Section {page_idx}.0: {subsystem} Electrical & Thermal Specification ({ic_name})",
            fontsize=13,
            fontname="helv",
        )

        p1 = (
            f"The {subsystem} module utilizes the {ic_name} controller operating under continuous "
            f"feedback regulation across Page {page_idx}. Nominal bus impedance is maintained at "
            f"50 ohms with phase margin exceeding 58 degrees under full transient load steps."
        )
        page.insert_textbox(fitz.Rect(50, 75, 545, 130), p1, fontsize=10, fontname="helv")

        p2 = (
            f"Thermal derating guidelines specify maximum junction temperature TJ_MAX = {125 + (page_idx % 25)} C "
            f"when mounted on a 4-layer FR4 PCB with 2 oz copper ground planes. Harmonic distortion remains "
            f"below -82 dBc across the entire 100 kHz to 20 MHz diagnostic bandwidth."
        )
        page.insert_textbox(fitz.Rect(50, 140, 545, 195), p2, fontsize=10, fontname="helv")

        # Draw a 4-column x 4-row bordered table
        t_x0, t_y0 = 50.0, 220.0
        col_w, row_h = 120.0, 28.0
        n_rows, n_cols = 4, 4

        for r in range(n_rows + 1):
            y = t_y0 + r * row_h
            page.draw_line(
                fitz.Point(t_x0, y),
                fitz.Point(t_x0 + n_cols * col_w, y),
                width=1.0,
            )
        for c in range(n_cols + 1):
            x = t_x0 + c * col_w
            page.draw_line(
                fitz.Point(x, t_y0),
                fitz.Point(x, t_y0 + n_rows * row_h),
                width=1.0,
            )

        table_rows = [
            ["Parameter", "Min", "Typical", "Max"],
            [f"{ic_name}_VIN_V", "4.50", "12.00", "36.00"],
            [f"IOUT_PEAK_A_P{page_idx}", "0.50", f"{2.0 + (page_idx % 5):.2f}", "8.50"],
            ["EFFICIENCY_PCT", "91.5", "96.2", "98.1"],
        ]
        for r_idx, row in enumerate(table_rows):
            for c_idx, cell in enumerate(row):
                page.insert_text(
                    fitz.Point(
                        t_x0 + c_idx * col_w + 8,
                        t_y0 + (r_idx + 1) * row_h - 9,
                    ),
                    cell,
                    fontsize=9,
                    fontname="helv",
                )

    doc.save(str(output_path))
    doc.close()


def build_diverse_queries() -> list[str]:
    """Builds 50 diverse technical queries mixing acronyms, exact part numbers, and table attributes."""
    templates = [
        "What is the maximum input voltage VIN for LM5176?",
        "Explain thermal derating TJ_MAX guidelines on 4-layer FR4 PCB.",
        "What is the typical conversion efficiency percentage for TPS54302?",
        "Describe phase margin and transient load step performance of PowerStage.",
        "What is the peak output current IOUT_PEAK_A_P3 for the DMAController?",
        "What diagnostic bandwidth is specified for harmonic distortion in ADCPipeline?",
        "What bus impedance is maintained by the ClockTree subsystem?",
        "Compare min and max VIN ratings for STM32H743.",
        "What copper weight is recommended for PCB thermal dissipation?",
        "What is the maximum efficiency percentage of Si5351A?",
    ]
    queries: list[str] = []
    for i in range(50):
        base = templates[i % len(templates)]
        suffix = f" (Section {(i % 30) + 1})" if i >= len(templates) else ""
        queries.append(f"{base}{suffix}")
    return queries


def run_benchmark() -> None:
    """Executes ingestion, indexing, and retrieval latency benchmarks across synthetic PDFs."""
    settings = get_settings()

    print("Warming up ML singletons (SentenceTransformer & CrossEncoder)...")
    encoder = get_embedding_model(settings.EMBEDDING_MODEL)
    reranker = get_reranker_model(settings.RERANKER_MODEL)
    encoder.encode(["warmup query"], normalize_embeddings=True, show_progress_bar=False)
    reranker.predict([("warmup query", "warmup context")])

    parser = DocumentParser()
    page_counts = [5, 15, 30]
    ingestion_results: list[dict[str, float | int]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        qdrant_client = QdrantClient(path=str(tmp_path / "qdrant_bench"))
        try:
            retriever = HybridRetriever(
                client=qdrant_client,
                collection_name="bench_doc_chunks",
            )

            target_doc_id = ""
            for pages in page_counts:
                pdf_path = tmp_path / f"bench_{pages}p.pdf"
                doc_id = f"bench_{pages}p"
                generate_synthetic_pdf(pdf_path, num_pages=pages)

                parse_start = time.perf_counter()
                chunks = parser.parse_document(file_path=pdf_path, doc_id=doc_id)
                parse_elapsed = time.perf_counter() - parse_start
                pages_per_sec = pages / parse_elapsed if parse_elapsed > 0 else 0.0

                index_start = time.perf_counter()
                indexed_count = retriever.index_chunks(chunks)
                index_elapsed_ms = (time.perf_counter() - index_start) * 1000.0
                ms_per_chunk = index_elapsed_ms / indexed_count if indexed_count > 0 else 0.0

                ingestion_results.append(
                    {
                        "pages": pages,
                        "chunks": indexed_count,
                        "parse_sec": parse_elapsed,
                        "pages_per_sec": pages_per_sec,
                        "index_ms": index_elapsed_ms,
                        "ms_per_chunk": ms_per_chunk,
                    }
                )
                if pages == 30:
                    target_doc_id = doc_id

            # Run 50 diverse queries against the 30-page indexed document
            queries = build_diverse_queries()
            query_latencies_ms: list[float] = []
            for query in queries:
                q_start = time.perf_counter()
                results = retriever.retrieve(doc_id=target_doc_id, query=query, top_k=5)
                q_elapsed_ms = (time.perf_counter() - q_start) * 1000.0
                if not results:
                    raise RuntimeError(f"Expected non-empty results for benchmark query: {query}")
                query_latencies_ms.append(q_elapsed_ms)

            q_percentiles = statistics.quantiles(query_latencies_ms, n=100)
            p50_ms = q_percentiles[49]
            p95_ms = q_percentiles[94]
            p99_ms = q_percentiles[98]
            mean_ms = statistics.mean(query_latencies_ms)
        finally:
            qdrant_client.close()

    processor = platform.processor() or platform.machine() or "Unknown CPU"
    cpu_cores = os.cpu_count() or 1

    print("\n## Hardware Profile")
    print(f"- **OS**: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"- **Processor**: {processor}")
    print(f"- **CPU Cores**: {cpu_cores}")
    print(f"- **Python**: {platform.python_version()}")
    print(f"- **Embedding Model**: `{settings.EMBEDDING_MODEL}`")
    print(f"- **Reranker Model**: `{settings.RERANKER_MODEL}`")

    print("\n## Document Ingestion & Indexing Benchmark")
    print("| Document Size | Extracted Chunks | Parse Time (s) | Parsing Throughput (pages/s) | Total Index Time (ms) | Indexing Latency (ms/chunk) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for row in ingestion_results:
        print(
            f"| {row['pages']} pages | {row['chunks']} | {row['parse_sec']:.2f} | "
            f"{row['pages_per_sec']:.2f} | {row['index_ms']:.1f} | {row['ms_per_chunk']:.2f} |"
        )

    print("\n## Hybrid Retrieval + Cross-Encoder Reranking Latency (50 Queries @ 30-Page Doc)")
    print("| Metric | Latency (ms) |")
    print("| :--- | :---: |")
    print(f"| Mean | {mean_ms:.2f} |")
    print(f"| **p50 (Median)** | **{p50_ms:.2f}** |")
    print(f"| **p95** | **{p95_ms:.2f}** |")
    print(f"| **p99** | **{p99_ms:.2f}** |")


if __name__ == "__main__":
    run_benchmark()
