"""VietPhapLy RAG — cross-encoder reranker loader."""

from __future__ import annotations

from vpl.settings import SEARCH


def load_reranker(device: str = "cpu"):
    """Load BGE reranker. Trả về None nếu không available."""
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(SEARCH.reranker_model, device=device, max_length=512)
        print(f"✅ Reranker loaded: {SEARCH.reranker_model}")
        return reranker
    except Exception as exc:
        print(f"⚠ Reranker unavailable ({exc}), using RRF scores only")
        return None
