from functools import lru_cache
import logging
from typing import Any

from openai import AsyncOpenAI

from app.api.schemas import Citation, DocumentChunk
from app.core.config import get_settings

logger = logging.getLogger("doc_intel.generator")


class RAGGenerator:
    """Citation-grounded synthesis generator with OpenAI LLM client and zero-cost local extractive fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL

        if self.api_key and self.api_key.strip():
            self.client: AsyncOpenAI | None = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            logger.info("RAGGenerator initialized with AsyncOpenAI (model=%s)", self.model)
        else:
            self.client = None
            logger.info("RAGGenerator running in Zero-Cost Local Fallback mode (no LLM_API_KEY)")

    @staticmethod
    def _create_citations(
        ranked_chunks: list[tuple[DocumentChunk, float]],
    ) -> list[Citation]:
        """Builds structured Citation instances referencing spatial coordinates and relevance scores."""
        citations: list[Citation] = []
        for idx, (chunk, score) in enumerate(ranked_chunks, start=1):
            snippet = (
                chunk.content[:220].strip() + ("..." if len(chunk.content) > 220 else "")
            )
            citations.append(
                Citation(
                    source_index=idx,
                    chunk_id=chunk.chunk_id,
                    page_num=chunk.page_num,
                    chunk_type=chunk.chunk_type,
                    bbox=chunk.bbox,
                    relevance_score=round(float(score), 4),
                    snippet=snippet,
                )
            )
        return citations

    def _local_extractive_fallback(
        self,
        question: str,
        ranked_chunks: list[tuple[DocumentChunk, float]],
    ) -> str:
        """Synthesizes structured extractive summary citing sources when external LLM is not configured."""
        if not ranked_chunks:
            return "I cannot find this information in the provided document."

        lines: list[str] = [
            f"Synthesized response for query: '{question}'\n"
        ]

        for idx, (chunk, _) in enumerate(ranked_chunks[:3], start=1):
            if chunk.chunk_type == "table":
                lines.append(
                    f"**Table Data [Source {idx} - Page {chunk.page_num}]:**\n\n{chunk.content}\n"
                )
            else:
                sentences = [s.strip() for s in chunk.content.split(". ") if s.strip()]
                summary = ". ".join(sentences[:2])
                if summary and not summary.endswith("."):
                    summary += "."
                lines.append(f"{summary} [Source {idx}]")

        return "\n\n".join(lines).strip()

    async def generate_response(
        self,
        doc_id: str,
        question: str,
        ranked_chunks: list[tuple[DocumentChunk, float]],
    ) -> tuple[str, list[Citation]]:
        """Generates a citation-grounded response citing inline [Source N] tags."""
        citations = self._create_citations(ranked_chunks)

        if not ranked_chunks:
            return "I cannot find this information in the provided document.", []

        # Local fallback if API key is absent
        if self.client is None:
            answer = self._local_extractive_fallback(question, ranked_chunks)
            return answer, citations

        # Build numbered context blocks
        context_blocks: list[str] = []
        for idx, (chunk, _) in enumerate(ranked_chunks, start=1):
            context_blocks.append(
                f"[Source {idx} | Page {chunk.page_num} | {chunk.chunk_type}]\n{chunk.content}"
            )
        context_corpus = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an expert technical document intelligence system. "
            "Answer the user's question accurately and exclusively using the numbered sources provided below.\n\n"
            "Rules:\n"
            "1. Ground every claim directly in the sources and cite inline immediately using [Source N] tags.\n"
            "2. When quoting tabular data, preserve or render valid Markdown table syntax.\n"
            "3. Do not invent, extrapolate, or speculate beyond the provided sources.\n"
            "4. If the provided context does not contain sufficient facts to answer, state exactly: "
            "'I cannot find this information in the provided document.'"
        )

        user_prompt = (
            f"Document Context:\n{context_corpus}\n\n"
            f"Question: {question}\n\n"
            "Provide a grounded, cited answer:"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content or ""
            return answer.strip(), citations
        except Exception as exc:
            logger.error("LLM generation failed: %s. Falling back to local extractive summary.", exc)
            fallback_answer = self._local_extractive_fallback(question, ranked_chunks)
            return fallback_answer, citations


@lru_cache(maxsize=1)
def get_generator() -> RAGGenerator:
    """Lazy singleton accessor for the RAGGenerator."""
    return RAGGenerator()
