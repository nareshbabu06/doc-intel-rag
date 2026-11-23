from pathlib import Path
import pytest
from qdrant_client import QdrantClient

from app.api.schemas import BoundingBox, DocumentChunk
from app.services.generator import RAGGenerator
from app.services.retriever import HybridRetriever


@pytest.fixture
def sample_technical_chunks() -> list[DocumentChunk]:
    doc_id = "test_hardware_spec"
    bbox = BoundingBox(x0=50.0, y0=50.0, x1=500.0, y1=150.0)

    return [
        DocumentChunk(
            chunk_id=f"{doc_id}_p1_c1",
            doc_id=doc_id,
            page_num=1,
            chunk_type="text",
            content="General system architecture overview: The host controller interfaces with multiple peripherals via standard SPI and I2C buses.",
            bbox=bbox,
            token_estimate=25,
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_p1_c2",
            doc_id=doc_id,
            page_num=1,
            chunk_type="text",
            content="Power management IC LM5176 provides wide input voltage regulation from 4.2V to 55V with 98% peak conversion efficiency.",
            bbox=bbox,
            token_estimate=25,
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_p1_c3",
            doc_id=doc_id,
            page_num=1,
            chunk_type="table",
            content="| Rail Name | Min Voltage (V) | Typ Voltage (V) | Max Voltage (V) |\n| --- | --- | --- | --- |\n| VCC_CORE | 1.14 | 1.20 | 1.26 |\n| VDD_FLASH | 3.135 | 3.30 | 3.465 |",
            bbox=bbox,
            token_estimate=45,
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_p2_c1",
            doc_id=doc_id,
            page_num=2,
            chunk_type="text",
            content="Thermal shutdown protection activates when internal silicon junction temperature exceeds TJ_MAX of 150 degrees Celsius.",
            bbox=bbox,
            token_estimate=22,
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_p2_c2",
            doc_id=doc_id,
            page_num=2,
            chunk_type="ocr_text",
            content="FIRMWARE CALIBRATION REGISTRY: ECE-9021-REV_B hardware revision identifier calibrated on automated bench fixture test.",
            bbox=bbox,
            token_estimate=20,
        ),
    ]


def test_hybrid_retriever_indexing_and_ranking(
    tmp_path: Path,
    sample_technical_chunks: list[DocumentChunk],
) -> None:
    """Verifies that HybridRetriever ranks the exact target chunk #1 for specific technical queries and cleans up on delete."""
    client = QdrantClient(path=str(tmp_path / "qdrant_test_db"))
    retriever = HybridRetriever(client=client, collection_name="test_doc_chunks")

    # 1. Index 5 technical chunks
    indexed_count = retriever.index_chunks(sample_technical_chunks)
    assert indexed_count == 5

    # 2. Verify retrieve() ranks the exact target chunk #1 for acronyms and model numbers
    query = "What is the input voltage regulation range for the LM5176 power IC?"
    results = retriever.retrieve(doc_id="test_hardware_spec", query=query, top_k=3)
    assert len(results) > 0

    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id == "test_hardware_spec_p1_c2"
    assert "LM5176" in top_chunk.content
    assert "4.2V to 55V" in top_chunk.content

    # 3. Verify table retrieval
    table_query = "What is the typical voltage for VCC_CORE rail?"
    table_results = retriever.retrieve(doc_id="test_hardware_spec", query=table_query, top_k=3)
    assert len(table_results) > 0
    assert table_results[0][0].chunk_id == "test_hardware_spec_p1_c3"
    assert "VCC_CORE" in table_results[0][0].content

    # 4. Verify delete_document() removes all vectors for that doc_id
    retriever.delete_document("test_hardware_spec")
    post_delete_results = retriever.retrieve(
        doc_id="test_hardware_spec",
        query="LM5176 voltage",
        top_k=3,
    )
    assert len(post_delete_results) == 0


@pytest.mark.anyio
async def test_rag_generator_extractive_fallback(
    sample_technical_chunks: list[DocumentChunk],
) -> None:
    """Verifies that RAGGenerator returns structured extractive citations when LLM_API_KEY is not set."""
    generator = RAGGenerator(api_key=None)
    ranked = [
        (sample_technical_chunks[1], 0.9234),
        (sample_technical_chunks[2], 0.8142),
    ]

    answer, citations = await generator.generate_response(
        doc_id="test_hardware_spec",
        question="What are the LM5176 specs?",
        ranked_chunks=ranked,
    )

    assert len(citations) == 2
    assert citations[0].source_index == 1
    assert citations[0].chunk_id == sample_technical_chunks[1].chunk_id
    assert citations[0].relevance_score == 0.9234
    assert "[Source 1]" in answer
    assert "LM5176" in answer
