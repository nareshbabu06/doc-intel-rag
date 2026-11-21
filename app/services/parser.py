from collections import defaultdict
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Literal

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract

from app.api.schemas import BoundingBox, DocumentChunk
from app.core.exceptions import DocumentParsingError

logger = logging.getLogger("doc_intel.parser")

# Automatically detect system Tesseract on Windows if not on default system PATH
if shutil.which("tesseract") is None:
    standard_win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if standard_win_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(standard_win_path)


class DocumentParser:
    """Multi-modal, layout-aware document parser with table extraction, deduplication, and OCR fallback."""

    def __init__(
        self,
        max_chunk_tokens: int = 450,
        min_native_chars: int = 30,
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.min_native_chars = min_native_chars

    @staticmethod
    def _format_markdown_table(rows: list[list[str | None]]) -> str:
        """Converts raw 2D extracted table cells into GitHub-Flavored Markdown table syntax."""
        cleaned_rows: list[list[str]] = []
        for row in rows:
            cleaned_row = [
                " ".join((cell or "").replace("\n", " ").split())
                for cell in row
            ]
            if any(cell for cell in cleaned_row):
                cleaned_rows.append(cleaned_row)

        if not cleaned_rows:
            return ""

        num_cols = max(len(row) for row in cleaned_rows)
        if num_cols == 0:
            return ""

        # Normalize column counts across rows
        padded_rows = [
            row + [""] * (num_cols - len(row)) for row in cleaned_rows
        ]

        header = padded_rows[0]
        divider = ["---"] * num_cols
        lines: list[str] = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(divider) + " |",
        ]
        for data_row in padded_rows[1:]:
            lines.append("| " + " | ".join(data_row) + " |")

        return "\n".join(lines)

    @staticmethod
    def _is_table_overlap(
        block_bbox: BoundingBox,
        table_bboxes: list[BoundingBox],
        threshold: float = 0.45,
    ) -> bool:
        """Checks if a text block substantially overlaps with an extracted table."""
        for t_bbox in table_bboxes:
            if block_bbox.iou(t_bbox) > threshold:
                return True
            if block_bbox.area > 0.0:
                overlap_ratio = block_bbox.intersection_area(t_bbox) / block_bbox.area
                if overlap_ratio > threshold:
                    return True
        return False

    def _extract_tables_with_pdfplumber(
        self,
        pdf_path: str,
    ) -> dict[int, list[tuple[str, BoundingBox]]]:
        """Step A: Extract tables per page using pdfplumber, converting to Markdown."""
        page_tables: dict[int, list[tuple[str, BoundingBox]]] = defaultdict(list)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    found_tables = page.find_tables()
                    for t in found_tables:
                        raw_data = t.extract()
                        if not raw_data:
                            continue
                        md_table = self._format_markdown_table(raw_data)
                        if not md_table:
                            continue
                        bbox = BoundingBox(
                            x0=float(t.bbox[0]),
                            y0=float(t.bbox[1]),
                            x1=float(t.bbox[2]),
                            y1=float(t.bbox[3]),
                        )
                        page_tables[page_idx].append((md_table, bbox))
        except Exception as exc:
            logger.error("Failed to extract tables via pdfplumber from %s: %s", pdf_path, exc)
            raise DocumentParsingError(
                message=f"Table extraction failed: {exc}",
                details={"file_path": str(pdf_path), "error": str(exc)},
            ) from exc

        return page_tables

    def _ocr_page(
        self,
        page: fitz.Page,
        page_num: int,
    ) -> list[tuple[str, BoundingBox]]:
        """Step C: Render page at 200 DPI and run Tesseract OCR line reconstruction."""
        dpi = 200.0
        scale = 72.0 / dpi
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        try:
            pixmap = page.get_pixmap(matrix=matrix)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            ocr_dict = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
            )
        except (pytesseract.TesseractNotFoundError, Exception) as exc:
            logger.warning(
                "OCR execution failed or Tesseract is unavailable on page %d: %s",
                page_num,
                exc,
            )
            return []

        lines_dict: dict[tuple[int, int, int], list[tuple[str, BoundingBox]]] = defaultdict(list)
        num_items = len(ocr_dict.get("text", []))

        for idx in range(num_items):
            word_str = ocr_dict["text"][idx].strip()
            if not word_str:
                continue

            left = float(ocr_dict["left"][idx]) * scale
            top = float(ocr_dict["top"][idx]) * scale
            width = float(ocr_dict["width"][idx]) * scale
            height = float(ocr_dict["height"][idx]) * scale

            word_bbox = BoundingBox(
                x0=left,
                y0=top,
                x1=left + width,
                y1=top + height,
            )
            key = (
                int(ocr_dict["block_num"][idx]),
                int(ocr_dict["par_num"][idx]),
                int(ocr_dict["line_num"][idx]),
            )
            lines_dict[key].append((word_str, word_bbox))

        reconstructed_lines: list[tuple[str, BoundingBox]] = []
        for _, words in lines_dict.items():
            if not words:
                continue
            line_text = " ".join(w[0] for w in words)
            line_bbox = words[0][1]
            for _, w_bbox in words[1:]:
                line_bbox = line_bbox.union(w_bbox)
            reconstructed_lines.append((line_text, line_bbox))

        return reconstructed_lines

    def parse_document(
        self,
        file_path: Path | str,
        doc_id: str | None = None,
    ) -> list[DocumentChunk]:
        """Parses a PDF into layout-aware, deduplicated DocumentChunk instances."""
        path = Path(file_path)
        if not path.is_file():
            raise DocumentParsingError(
                message=f"Document file does not exist: {path}",
                details={"file_path": str(path)},
            )

        resolved_doc_id = doc_id or path.stem

        # Step A: Table Extraction with pdfplumber
        tables_per_page = self._extract_tables_with_pdfplumber(str(path))

        all_chunks: list[DocumentChunk] = []

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise DocumentParsingError(
                message=f"PyMuPDF failed to open document: {exc}",
                details={"file_path": str(path), "error": str(exc)},
            ) from exc

        try:
            for page_num in range(1, doc.page_count + 1):
                page = doc[page_num - 1]
                table_entries = tables_per_page.get(page_num, [])
                table_bboxes = [entry[1] for entry in table_entries]

                # Step B: Text Block Extraction & Deduplication
                raw_blocks = page.get_text("blocks")
                native_text_blocks: list[tuple[str, BoundingBox]] = []
                total_native_chars = sum(len(entry[0]) for entry in table_entries)

                for b in raw_blocks:
                    # Filter block_type == 0 (text blocks)
                    if len(b) >= 7 and b[6] != 0:
                        continue
                    text_content = b[4].strip()
                    if not text_content:
                        continue

                    block_bbox = BoundingBox(
                        x0=float(b[0]),
                        y0=float(b[1]),
                        x1=float(b[2]),
                        y1=float(b[3]),
                    )

                    # Deduplicate: drop if overlapping with any extracted table
                    if self._is_table_overlap(block_bbox, table_bboxes, threshold=0.45):
                        continue

                    total_native_chars += len(text_content)
                    native_text_blocks.append((text_content, block_bbox))

                # Step C: OCR Fallback if below min_native_chars threshold
                page_text_items: list[tuple[str, BoundingBox, Literal["text", "ocr_text"]]] = []

                if total_native_chars < self.min_native_chars:
                    logger.info(
                        "Page %d has only %d native characters (< %d). Triggering OCR fallback.",
                        page_num,
                        total_native_chars,
                        self.min_native_chars,
                    )
                    ocr_lines = self._ocr_page(page, page_num)
                    for text, bbox in ocr_lines:
                        page_text_items.append((text, bbox, "ocr_text"))
                else:
                    for text, bbox in native_text_blocks:
                        page_text_items.append((text, bbox, "text"))

                # Step D: Layout-Aware Chunking
                # 1. Convert tables to atomic DocumentChunk candidates
                page_candidates: list[tuple[BoundingBox, Literal["text", "table", "ocr_text"], str]] = []
                for table_md, t_bbox in table_entries:
                    page_candidates.append((t_bbox, "table", table_md))

                # 2. Merge consecutive narrative text/ocr blocks up to max_chunk_tokens
                if page_text_items:
                    # Sort blocks vertically, then horizontally
                    sorted_text_items = sorted(
                        page_text_items,
                        key=lambda item: (item[1].y0, item[1].x0),
                    )

                    current_texts: list[str] = [sorted_text_items[0][0]]
                    current_bbox: BoundingBox = sorted_text_items[0][1]
                    current_type: Literal["text", "ocr_text"] = sorted_text_items[0][2]
                    current_tokens: int = len(sorted_text_items[0][0]) // 4

                    for next_text, next_bbox, next_type in sorted_text_items[1:]:
                        add_tokens = len(next_text) // 4
                        merged_char_len = sum(len(t) for t in current_texts) + len(next_text) + 2
                        projected_tokens = merged_char_len // 4

                        if (
                            projected_tokens <= self.max_chunk_tokens
                            and next_type == current_type
                        ):
                            current_texts.append(next_text)
                            current_bbox = current_bbox.union(next_bbox)
                            current_tokens = projected_tokens
                        else:
                            page_candidates.append(
                                (current_bbox, current_type, "\n\n".join(current_texts))
                            )
                            current_texts = [next_text]
                            current_bbox = next_bbox
                            current_type = next_type
                            current_tokens = add_tokens

                    if current_texts:
                        page_candidates.append(
                            (current_bbox, current_type, "\n\n".join(current_texts))
                        )

                # Sort all candidates on the page top-to-bottom for deterministic layout ordering
                page_candidates.sort(key=lambda item: (item[0].y0, item[0].x0))

                for idx, (c_bbox, c_type, content) in enumerate(page_candidates, start=1):
                    chunk_id = f"{resolved_doc_id}_p{page_num}_c{idx}"
                    token_est = max(1, len(content) // 4)
                    all_chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=resolved_doc_id,
                            page_num=page_num,
                            chunk_type=c_type,
                            content=content,
                            bbox=c_bbox,
                            token_estimate=token_est,
                        )
                    )
        finally:
            doc.close()

        return all_chunks
