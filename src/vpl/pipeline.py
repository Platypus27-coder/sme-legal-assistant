"""
VietPhapLy RAG — end-to-end pipeline orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vpl.settings import DATA_DIR, GENERATION, RESULTS_STREAM_FILE, SEARCH


def run_retrieve(
    questions_file: Path,
    device: str = "cpu",
    no_reranker: bool = False,
    reset: bool = False,
) -> None:
    """Phase: retrieve all questions → populate SQLite cache."""
    from vpl.cache import RetrievalCache
    from vpl.search.expander import expand
    from vpl.search.hybrid import HybridRetriever
    from vpl.search.reranker import load_reranker
    from vpl.store.vectors import get_collection

    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    cache = RetrievalCache()

    if reset:
        import sqlite3
        cache._conn.execute("DELETE FROM retrieval_cache")
        cache._conn.commit()
        print("Cache reset.")

    pending = [q for q in questions if int(q["id"]) not in cache.completed_ids]
    print(f"{len(questions)} questions, {len(pending)} pending retrieval")
    if not pending:
        print("✅ All questions already cached")
        return

    collection, embed_model = get_collection(device=device)
    reranker = None if no_reranker else load_reranker(device=device)
    retriever = HybridRetriever(
        chroma_collection=collection,
        embedding_model=embed_model,
        reranker=reranker,
        device=device,
    )

    for i, q in enumerate(pending, 1):
        expanded = expand(str(q["question"]))
        results = retriever.retrieve(str(q["question"]), expanded_query=expanded)
        cache.append(int(q["id"]), str(q["question"]), results)
        if i % 100 == 0:
            print(f"  retrieved {i}/{len(pending)}...", flush=True)
            
    # Clean up VRAM before returning to prevent OOM in generate phase
    del retriever
    del reranker
    del collection
    del embed_model
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print(f"✅ Retrieval complete → {cache.path}")


def run_generate(
    questions_file: Path,
    device: str = "cuda",
    reset: bool = False,
    min_articles: int | None = None,
    max_articles: int | None = None,
    safe_threshold: float | None = None,
) -> None:
    """Phase: generate answers from cache → results_partial.jsonl."""
    from vpl.answer.generator import LegalGenerator
    from vpl.answer.postprocess import PostConfig, PostProcessor
    from vpl.cache import RetrievalCache
    from vpl.search.hybrid import ScoredChunk
    from vpl.submit import ResultStore

    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    result_store = ResultStore()
    cache = RetrievalCache()

    if reset:
        RESULTS_STREAM_FILE.unlink(missing_ok=True)
        result_store = ResultStore()

    pending = [q for q in questions if int(q["id"]) not in result_store.completed_ids]
    print(f"{len(pending)} questions pending generation")
    if not pending:
        print("✅ All questions already generated")
        return

    generator = LegalGenerator.from_pretrained()
    post_cfg = PostConfig(
        min_articles=min_articles or SEARCH.min_articles,
        max_articles=max_articles or SEARCH.max_articles,
        safe_threshold=safe_threshold or SEARCH.safe_threshold,
    )
    postprocessor = PostProcessor(post_cfg)
    bs = GENERATION.batch_size

    for start in range(0, len(pending), bs):
        batch = pending[start : start + bs]
        contexts: list[list] = []
        for q in batch:
            try:
                raw_chunks = cache.get(int(q["id"]), str(q["question"]))
            except KeyError:
                raw_chunks = []
            # Convert cache dicts back to ScoredChunk-like objects
            scored = [
                ScoredChunk(chunk=c, score=float(c.get("score", 0)))
                for c in raw_chunks
            ]
            contexts.append(scored[:SEARCH.max_context_chunks])

        answers = generator.generate([str(q["question"]) for q in batch], contexts)
        for q, answer, chunks in zip(batch, answers, contexts):
            result = postprocessor.build_result(
                question_id=int(q["id"]),
                question=str(q["question"]),
                answer=answer,
                chunks=chunks,
            )
            result_store.append(result)
        print(f"  generated {start + len(batch)}/{len(pending)}", flush=True)

    print(f"✅ Generation complete → {result_store.path}")
