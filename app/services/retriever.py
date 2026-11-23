from functools import lru_cache
import logging
from pathlib import Path
import re
from typing import Any
import uuid

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.api.schemas import DocumentChunk
from app.core.config import get_settings

logger = logging.getLogger("doc_intel.retriever")


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Lazy-loaded singleton embedding model instance."""
    logger.info("Loading SentenceTransformer embedding model: %s", model_name)
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_reranker_model(model_name: str) -> CrossEncoder:
    """Lazy-loaded singleton cross-encoder reranker model instance."""
    logger.info("Loading CrossEncoder reranker model: %s", model_name)
    return CrossEncoder(model_name)


class HybridRetriever:
    """Hybrid vector and lexical retriever with Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking."""

    COLLECTION_NAME = "doc_chunks"
    EMBEDDING_DIM = 384

    # Retrieval Tuning Parameters:
    # 1. RRF_K = 50.0: Reducing the RRF dampening constant from 60 to 50 moderately amplifies the
    #    discriminative advantage of top-3 rankers in either sparse BM25 or dense semantic channels,
    #    preventing highly specific exact-match tokens (e.g. part numbers) from being diluted.
    # 2. DENSE_POOL_MULTIPLIER = 4: Expands initial dense query pool (top_k * 4) to ensure high recall
    #    across multi-page technical documents prior to reciprocal fusion.
    # 3. RRF_CANDIDATE_LIMIT = 15: Passes a richer candidate set to the CrossEncoder reranker.
    RRF_K: float = 50.0
    DENSE_POOL_MULTIPLIER: int = 4
    RRF_CANDIDATE_LIMIT: int = 15

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str = "doc_chunks",
    ) -> None:
        self.collection_name = collection_name
        settings = get_settings()

        if client is not None:
            self.client = client
        else:
            self.client = self._init_qdrant_client(settings)

        self._ensure_collection()

    def _init_qdrant_client(self, settings: Any) -> QdrantClient:
        """Attempts connection to remote Qdrant with 2s timeout, falling back to embedded local storage."""
        if settings.QDRANT_URL:
            try:
                candidate_client = QdrantClient(url=settings.QDRANT_URL, timeout=2.0)
                candidate_client.get_collections()
                logger.info("Connected to remote Qdrant at %s", settings.QDRANT_URL)
                return candidate_client
            except Exception as exc:
                logger.warning(
                    "Remote Qdrant at %s unreachable (%s). Initializing embedded local Qdrant.",
                    settings.QDRANT_URL,
                    exc,
                )

        local_path = Path(settings.QDRANT_LOCAL_PATH)
        local_path.mkdir(parents=True, exist_ok=True)
        embedded_client = QdrantClient(path=str(local_path))
        logger.info("Using embedded local Qdrant at %s", local_path.resolve())
        return embedded_client

    def _ensure_collection(self) -> None:
        """Guarantees the doc_chunks collection and keyword payload index exist."""
        if not self.client.collection_exists(self.collection_name):
            logger.info("Creating Qdrant collection: %s", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            logger.debug("doc_id index check/creation notice: %s", exc)

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embeds and upserts document chunks into Qdrant using deterministic UUIDs."""
        if not chunks:
            return 0

        settings = get_settings()
        encoder = get_embedding_model(settings.EMBEDDING_MODEL)
        contents = [c.content for c in chunks]
        embeddings = encoder.encode(
            contents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points: list[models.PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload=chunk.model_dump(),
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Indexed %d chunks into %s", len(points), self.collection_name)
        return len(points)

    def delete_document(self, doc_id: str) -> None:
        """Deletes all vector points associated with a specific doc_id."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
        logger.info("Deleted vectors for doc_id=%s from %s", doc_id, self.collection_name)

    def retrieve(
        self,
        doc_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Executes dense + BM25 hybrid search, RRF fusion, and Cross-Encoder reranking."""
        settings = get_settings()
        doc_filter = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )

        # 1. Sparse Lexical Search: retrieve all document chunks for BM25
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=doc_filter,
            limit=1000,
            with_payload=True,
        )

        if not records:
            return []

        all_doc_chunks: list[DocumentChunk] = [
            DocumentChunk.model_validate(r.payload)
            for r in records
            if r.payload is not None
        ]

        if not all_doc_chunks:
            return []

        tokenized_corpus = [
            re.findall(r"\w+", c.content.lower()) for c in all_doc_chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = re.findall(r"\w+", query.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        bm25_ranked = sorted(
            zip(all_doc_chunks, bm25_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        # 2. Dense Semantic Search: modern Qdrant v1.16+ query_points API
        encoder = get_embedding_model(settings.EMBEDDING_MODEL)
        query_vec = encoder.encode(query, normalize_embeddings=True).tolist()
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=doc_filter,
            limit=max(15, top_k * self.DENSE_POOL_MULTIPLIER),
            with_payload=True,
        )

        dense_points = dense_results.points

        # 3. Reciprocal Rank Fusion (RRF)
        dense_ranks: dict[str, int] = {}
        for rank, p in enumerate(dense_points, start=1):
            if p.payload and "chunk_id" in p.payload:
                dense_ranks[p.payload["chunk_id"]] = rank

        bm25_ranks: dict[str, int] = {
            chunk.chunk_id: rank
            for rank, (chunk, _) in enumerate(bm25_ranked, start=1)
        }

        chunk_lookup = {c.chunk_id: c for c in all_doc_chunks}
        all_candidate_ids = set(dense_ranks.keys()).union(bm25_ranks.keys())

        rrf_scores: dict[str, float] = {}
        for cid in all_candidate_ids:
            r_dense = dense_ranks.get(cid, 1000)
            r_bm25 = bm25_ranks.get(cid, 1000)
            rrf_scores[cid] = (1.0 / (self.RRF_K + r_dense)) + (1.0 / (self.RRF_K + r_bm25))

        sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        top_candidates = [
            chunk_lookup[cid]
            for cid, _ in sorted_rrf[: min(self.RRF_CANDIDATE_LIMIT, len(sorted_rrf))]
            if cid in chunk_lookup
        ]

        if not top_candidates:
            return []

        # 4. Cross-Encoder Reranking
        reranker = get_reranker_model(settings.RERANKER_MODEL)
        pairs = [(query, c.content) for c in top_candidates]
        cross_scores = reranker.predict(pairs)

        scored_candidates = [
            (chunk, float(score))
            for chunk, score in zip(top_candidates, cross_scores)
        ]
        scored_candidates.sort(key=lambda pair: pair[1], reverse=True)

        return scored_candidates[:top_k]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Lazy singleton accessor for the HybridRetriever."""
    return HybridRetriever()
