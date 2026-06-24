"""
VietPhapLy RAG — Hybrid retriever.

Pipeline:
  1. BM25 sparse (Top-50)
  2. ChromaDB dense (Top-50, với HyDE-expanded query)
  3. RRF fusion k=60
  4. Diversity filter (max 1 chunk per formatted_article)
  5. Cross-encoder reranker
  6. Lexical boost + SME score boost
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vpl.settings import (
    BM25_CORPUS_FILE,
    BM25_ID_MAP_FILE,
    CHUNKS_FILE,
    INDEX,
    SEARCH,
)
from vpl.store.bm25 import tokenize


@dataclass
class ScoredChunk:
    chunk: dict[str, Any]
    score: float

    @property
    def chunk_id(self) -> str:
        return str(self.chunk.get("chunk_id", ""))

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.get("metadata") or {}


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

def _rrf(rankings: Sequence[Sequence[str]], k: int = SEARCH.rrf_k) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# Chunk map loader
# ---------------------------------------------------------------------------

def _load_chunk_map(path: Path = CHUNKS_FILE) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                c = json.loads(line)
                chunks[str(c["chunk_id"])] = c
    return chunks


# ---------------------------------------------------------------------------
# Lexical boost
# ---------------------------------------------------------------------------

def _lexical_boost(query: str, chunk: dict[str, Any]) -> float:
    import re
    meta = chunk.get("metadata") or {}
    haystack = " ".join(
        str(meta.get(k) or "") for k in ("doc_id", "doc_title", "article_number")
    ).lower()
    boost = 0.0
    for token in re.findall(r"\d{1,3}/\d{4}/[\wĐđ-]+|điều\s+\d+[a-z]?", query.lower()):
        if token in haystack:
            boost += 0.02
    boost += min(float(meta.get("sme_score") or 0.0) / 20.0, 1.0) * 0.03
    if not meta.get("submission_eligible"):
        boost -= 0.05
    return boost


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class HybridRetriever:
    """BM25 + ChromaDB dense + RRF + optional cross-encoder."""

    def __init__(
        self,
        chroma_collection=None,
        embedding_model=None,
        reranker=None,
        device: str = "cpu",
    ):
        self.reranker = reranker
        self.device = device
        self._chroma = chroma_collection
        self._embed_model = embedding_model

        # Load BM25
        with BM25_CORPUS_FILE.open("rb") as fh:
            self._bm25 = pickle.load(fh)
        self._bm25_ids: list[str] = json.loads(BM25_ID_MAP_FILE.read_text(encoding="utf-8"))
        if len(self._bm25_ids) != len(self._bm25.doc_len):
            raise ValueError("BM25 id map size mismatch")

        # Load chunk map
        self._chunks = _load_chunk_map()
        print(f"✅ HybridRetriever ready ({len(self._chunks)} chunks)")

    # ------------------------------------------------------------------
    # Individual retrievers
    # ------------------------------------------------------------------

    def _bm25_top_k(self, query: str, k: int = SEARCH.bm25_top_k) -> list[str]:
        tokens = tokenize(query)
        scores = np.asarray(self._bm25.get_scores(tokens))
        top = min(k, len(scores))
        indices = np.argpartition(scores, -top)[-top:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [str(self._bm25_ids[i]) for i in indices if scores[i] > 0]

    def _dense_top_k(self, query: str, k: int = SEARCH.dense_top_k) -> list[str]:
        if self._chroma is None or self._embed_model is None:
            return []
        try:
            vec = self._embed_model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False,
                max_length=INDEX.embedding_max_length,
            ).tolist()[0]
            results = self._chroma.query(
                query_embeddings=[vec],
                n_results=k,
                include=["distances"],
            )
            return [str(cid) for cid in results["ids"][0]]
        except Exception as exc:
            print(f"  ⚠ dense retrieval error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank(
        self,
        query: str,
        chunk_ids: list[str],
        rrf_scores: dict[str, float],
    ) -> list[ScoredChunk]:
        chunks = [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]
        if not chunks:
            return []

        max_rrf = max(rrf_scores.values()) or 1.0

        if self.reranker is None:
            return [
                ScoredChunk(
                    chunk=c,
                    score=min(1.0, rrf_scores.get(c["chunk_id"], 0) / max_rrf + _lexical_boost(query, c)),
                )
                for c in chunks
            ]

        logits = self.reranker.predict(
            [(query, c["text"]) for c in chunks],
            batch_size=SEARCH.reranker_batch_size,
            show_progress_bar=False,
        )
        result = []
        for c, logit in zip(chunks, logits):
            rerank_s = _sigmoid(float(logit))
            rrf_s = rrf_scores.get(c["chunk_id"], 0) / max_rrf
            score = SEARCH.fusion_reranker_weight * rerank_s + SEARCH.fusion_base_weight * rrf_s + _lexical_boost(query, c)
            result.append(ScoredChunk(chunk=c, score=min(1.0, score)))
        return result

    # ------------------------------------------------------------------
    # Main retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, expanded_query: str | None = None) -> list[ScoredChunk]:
        """
        Args:
            query: câu hỏi gốc (dùng để rerank)
            expanded_query: HyDE-expanded query (dùng cho BM25 + dense)
        """
        search_q = expanded_query or query
        bm25_ids = self._bm25_top_k(search_q)
        dense_ids = self._dense_top_k(search_q)

        rankings = [r for r in [bm25_ids, dense_ids] if r]
        fused = _rrf(rankings)

        # Diversity filter: 1 chunk per formatted_article
        ordered: list[str] = []
        seen_articles: dict[str, int] = {}
        for cid in sorted(fused, key=fused.__getitem__, reverse=True):
            c = self._chunks.get(cid) or {}
            article = str((c.get("metadata") or {}).get("formatted_article") or cid)
            if seen_articles.get(article, 0) >= 1:
                continue
            seen_articles[article] = seen_articles.get(article, 0) + 1
            ordered.append(cid)
            if len(ordered) >= SEARCH.fusion_top_k:
                break

        results = self._rerank(query, ordered, fused)

        # ------------------------------------------------------------------
        # Context Window Expansion
        # Gộp các chunk thuộc cùng một Điều luật để cung cấp context rộng hơn
        # cho LLM Generator (giảm thiểu bỏ sót do chunking)
        # ------------------------------------------------------------------
        expanded_results = []
        for rc in results:
            c = rc.chunk
            article = str((c.get("metadata") or {}).get("formatted_article") or c.get("chunk_id", ""))
            
            # Tìm tất cả chunks có cùng article
            same_article_chunks = [
                chk for chk in self._chunks.values()
                if str((chk.get("metadata") or {}).get("formatted_article") or chk.get("chunk_id", "")) == article
            ]
            
            # Sắp xếp theo chunk_id gốc (chunk_id có format {luat}_{id}) để giữ đúng thứ tự đọc
            same_article_chunks.sort(key=lambda x: str(x.get("chunk_id", "")))
            
            # Gộp text
            expanded_text = "\n".join(chk.get("text", "") for chk in same_article_chunks)
            
            new_chunk = dict(c)
            new_chunk["text"] = expanded_text
            expanded_results.append(ScoredChunk(chunk=new_chunk, score=rc.score))

        return sorted(expanded_results, key=lambda x: x.score, reverse=True)
