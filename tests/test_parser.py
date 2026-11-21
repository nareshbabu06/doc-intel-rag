import io
from pathlib import Path
import shutil

import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import pytest

from app.api.schemas import BoundingBox, DocumentChunk
from app.services.parser import DocumentParser


def is_tesseract_installed() -> bool:
    """Detects whether Tesseract OCR binary is reachable via PATH or standard Windows location."""
    if shutil.which("tesseract") is not None:
        return True
    return Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists()


@pytest.fixture
def synthetic_multimodal_pdf(tmp_path: Path) -> Path:
    """Generates a 2-page synthetic PDF containing structured tables, text, and rasterized OCR content."""
    pdf_path = tmp_path / "synthetic_test.pdf"
    doc = fitz.open()

    # --- Page 1: Heading, 2 narrative paragraphs, drawn 3x3 bordered table ---
    p1 = doc.new_page(width=595, height=842)

    # Heading
    p1.insert_text(
        fitz.Point(50, 60),
        "ECE-402 High-Speed Integrated Circuit Telemetry Report",
        fontsize=14,
        fontname="helv",
    )

    # Narrative Paragraph 1
    p1.insert_textbox(
        fitz.Rect(50, 80, 545, 130),
        "The operational amplifier operates in closed-loop mode with feedback resistance configured for unity gain. "
        "Thermal dissipation remains stable within ambient operational ranges between -40C and 85C.",
        fontsize=10,
        fontname="helv",
    )

    # Narrative Paragraph 2
    p1.insert_textbox(
        fitz.Rect(50, 140, 545, 190),
        "Dynamic compensation networks attenuate secondary ringing at transition edges. "
        "Measured slew rates align directly with high-frequency simulation targets.",
        fontsize=10,
        fontname="helv",
    )

    # Drawn 3x3 Table
    t_x0, t_y0 = 50.0, 220.0
    cell_w, cell_h = 150.0, 30.0
    rows = 3
    cols = 3

    # Draw grid lines for pdfplumber table recognition
    for r in range(rows + 1):
        y = t_y0 + r * cell_h
        p1.draw_line(fitz.Point(t_x0, y), fitz.Point(t_x0 + cols * cell_w, y), width=1.0)
    for c in range(cols + 1):
        x = t_x0 + c * cell_w
        p1.draw_line(fitz.Point(x, t_y0), fitz.Point(x, t_y0 + rows * cell_h), width=1.0)

    # Populate cell contents
    grid_data = [
        ["Voltage_Rail", "Nominal_V", "Tolerance_Pct"],
        ["VCC_CORE", "1.20", "2.5"],
        ["VDD_IO", "3.30", "5.0"],
    ]
    for r_idx, row in enumerate(grid_data):
        for c_idx, cell_value in enumerate(row):
            p1.insert_text(
                fitz.Point(t_x0 + c_idx * cell_w + 10, t_y0 + (r_idx + 1) * cell_h - 10),
                cell_value,
                fontsize=9,
                fontname="helv",
            )

    # --- Page 2: Rasterized PNG image with clear text (0 native PDF text) ---
    p2 = doc.new_page(width=595, height=842)
    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "SCANNED CALIBRATION TELEMETRY DATA", fill="black")
    draw.text((20, 70), "SENSOR_ID_ALPHA: OFFSET = +0.042 mV", fill="black")
    draw.text((20, 110), "STATUS: VERIFIED NOMINAL", fill="black")

    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    p2.insert_image(fitz.Rect(50, 100, 545, 260), stream=img_bytes)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_bounding_box_methods() -> None:
    """Validates BoundingBox union, intersection area, and IoU calculations."""
    b1 = BoundingBox(x0=10.0, y0=10.0, x1=50.0, y1=50.0)
    b2 = BoundingBox(x0=30.0, y0=30.0, x1=70.0, y1=70.0)

    assert b1.area == 1600.0
    assert b2.area == 1600.0

    union_b = b1.union(b2)
    assert union_b.x0 == 10.0
    assert union_b.y0 == 10.0
    assert union_b.x1 == 70.0
    assert union_b.y1 == 70.0

    inter_area = b1.intersection_area(b2)
    assert inter_area == 400.0

    iou = b1.iou(b2)
    expected_iou = 400.0 / (1600.0 + 1600.0 - 400.0)
    assert abs(iou - expected_iou) < 1e-4


def test_parser_extracts_tables_deduplicates_and_validates_bboxes(
    synthetic_multimodal_pdf: Path,
) -> None:
    """Verifies table extraction in Markdown, deduplication against narrative text, and valid bounding boxes."""
    parser = DocumentParser(max_chunk_tokens=400, min_native_chars=30)
    chunks = parser.parse_document(synthetic_multimodal_pdf, doc_id="synth_doc")

    assert len(chunks) > 0

    # 1. Page 1 extracts at least 1 table chunk in Markdown format
    p1_tables = [c for c in chunks if c.page_num == 1 and c.chunk_type == "table"]
    assert len(p1_tables) >= 1, "Expected at least 1 table chunk on Page 1"
    table_content = p1_tables[0].content
    assert "|" in table_content
    assert "---" in table_content
    assert "Voltage_Rail" in table_content
    assert "VCC_CORE" in table_content

    # 2. Table cell values are NOT duplicated inside text chunks
    p1_texts = [c for c in chunks if c.page_num == 1 and c.chunk_type == "text"]
    assert len(p1_texts) >= 1, "Expected narrative text chunks on Page 1"
    text_corpus = " ".join(c.content for c in p1_texts)
    assert "Voltage_Rail" not in text_corpus
    assert "VCC_CORE" not in text_corpus
    assert "Tolerance_Pct" not in text_corpus
    assert "operational amplifier" in text_corpus

    # 3. All BoundingBox coordinates satisfy x0 < x1 and y0 < y1
    for chunk in chunks:
        assert chunk.bbox.x0 < chunk.bbox.x1, f"Invalid x bounds on {chunk.chunk_id}: {chunk.bbox}"
        assert chunk.bbox.y0 < chunk.bbox.y1, f"Invalid y bounds on {chunk.chunk_id}: {chunk.bbox}"
        assert chunk.token_estimate > 0


def test_parser_ocr_fallback(synthetic_multimodal_pdf: Path) -> None:
    """Verifies that Page 2 with 0 native text triggers OCR fallback."""
    if not is_tesseract_installed():
        pytest.skip("Tesseract OCR binary not found in environment")

    parser = DocumentParser(max_chunk_tokens=400, min_native_chars=30)
    chunks = parser.parse_document(synthetic_multimodal_pdf, doc_id="synth_doc")

    p2_ocr_chunks = [c for c in chunks if c.page_num == 2 and c.chunk_type == "ocr_text"]
    assert len(p2_ocr_chunks) >= 1, "Expected OCR text chunks on Page 2"

    ocr_corpus = " ".join(c.content for c in p2_ocr_chunks).upper()
    assert (
        "CALIBRATION" in ocr_corpus
        or "TELEMETRY" in ocr_corpus
        or "SENSOR" in ocr_corpus
    ), f"Expected OCR text in corpus: {ocr_corpus}"
