"""
VietPhapLy RAG — entry point duy nhất.

Thay thế nhiều script rải rác bằng subcommands:

  python run.py ingest            # thu thập + chunking
  python run.py index             # BM25 + ChromaDB
  python run.py retrieve          # hybrid retrieval → SQLite cache
  python run.py generate          # LLM generation
  python run.py submit            # validate + zip

  python run.py pipeline          # toàn bộ end-to-end
  python run.py eval              # local F2 evaluation
  python run.py retune            # tune thresholds không rerun LLM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vpl.settings import DATA_DIR, RESULTS_FILE, RESULTS_STREAM_FILE, SUBMISSION_FILE


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    print("=== INGEST: Thu thập + chunking ===")
    from vpl.corpus.loader import collect
    from vpl.corpus.chunker import chunk
    report = collect()
    print(f"Collected: {report.get('legal_docs', {}).get('count', 0)} legal docs, "
          f"{report.get('precedents', {}).get('count', 0)} precedents")
    stats = chunk()
    print(f"Chunks: {stats['total_chunks']} total, {stats['submission_eligible']} eligible")


def cmd_index(args: argparse.Namespace) -> None:
    print("=== INDEX: BM25 + ChromaDB ===")
    from vpl.store.bm25 import build as build_bm25
    from vpl.store.vectors import build as build_vectors
    bm25_report = build_bm25()
    print(f"BM25: {bm25_report['corpus_size']} docs")
    vec_report = build_vectors(device=args.device, reset=args.reset)
    print(f"ChromaDB: {vec_report['indexed']} new chunks, model={vec_report['model']}")


def cmd_retrieve(args: argparse.Namespace) -> None:
    print("=== RETRIEVE: Hybrid retrieval → SQLite cache ===")
    from vpl.pipeline import run_retrieve
    questions_file = Path(args.questions)
    run_retrieve(
        questions_file=questions_file,
        device=args.device,
        no_reranker=args.no_reranker,
        reset=args.reset,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    print("=== GENERATE: LLM answers ===")
    from vpl.pipeline import run_generate
    questions_file = Path(args.questions)
    run_generate(
        questions_file=questions_file,
        device=args.device,
        reset=args.reset,
        min_articles=args.min_articles,
        max_articles=args.max_articles,
        safe_threshold=args.safe_threshold,
    )


def cmd_submit(args: argparse.Namespace) -> None:
    print("=== SUBMIT: Validate + package ===")
    from vpl.submit import ResultStore, package

    # Convert JSONL → JSON
    store = ResultStore()
    results_path = store.export()
    print(f"Exported {len(store.completed_ids)} results → {results_path}")

    questions_file = Path(args.questions) if args.questions else None
    package(results_path=results_path, questions_path=questions_file)


def cmd_pipeline(args: argparse.Namespace) -> None:
    print("=== FULL PIPELINE ===")
    if not args.skip_ingest:
        cmd_ingest(args)
    if not args.skip_index:
        cmd_index(args)
    cmd_retrieve(args)
    cmd_generate(args)
    cmd_submit(args)


def cmd_eval(args: argparse.Namespace) -> None:
    print("=== EVAL: Local F2 ===")
    from vpl.evaluate import macro_f2
    preds = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    refs = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    metrics = macro_f2(preds, refs)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def cmd_retune(args: argparse.Namespace) -> None:
    """Tune thresholds từ SQLite cache mà không rerun LLM."""
    print("=== RETUNE: Threshold tuning ===")
    from vpl.answer.postprocess import PostConfig, PostProcessor
    from vpl.cache import RetrievalCache
    from vpl.search.hybrid import ScoredChunk
    from vpl.submit import ResultStore

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    cache = RetrievalCache()
    post_cfg = PostConfig(
        min_articles=args.min_articles,
        max_articles=args.max_articles,
        safe_threshold=args.safe_threshold,
        high_conf_threshold=args.high_conf_threshold,
    )
    postprocessor = PostProcessor(post_cfg)

    # Load existing answers
    existing_store = ResultStore()
    existing = {r["id"]: r for r in existing_store.read_all()}

    # New store at different path
    from vpl.settings import OUTPUT_DIR
    new_path = OUTPUT_DIR / f"results_retune_{args.safe_threshold}_{args.max_articles}.jsonl"
    from vpl.submit import ResultStore as RS2
    new_store = RS2(new_path)

    for q in questions:
        qid = int(q["id"])
        if qid not in existing:
            continue
        try:
            raw = cache.get(qid, str(q["question"]))
        except KeyError:
            continue
        chunks = [ScoredChunk(chunk=c, score=float(c.get("score", 0))) for c in raw]
        answer = existing[qid]["answer"]
        result = postprocessor.build_result(qid, str(q["question"]), answer, chunks)
        new_store.append(result)

    out = new_store.export(OUTPUT_DIR / f"results_retune_{args.safe_threshold}_{args.max_articles}.json")
    print(f"✅ Retuned results → {out}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cuda", help="cpu / cuda / mps")
    p.add_argument(
        "--questions",
        default=str(DATA_DIR / "R2AIStage1DATA.json"),
        help="Path to competition questions JSON",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="vpl", description="VietPhapLy RAG pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Collect + chunk legal data")
    _add_common(p_ingest)
    p_ingest.set_defaults(func=cmd_ingest)

    # index
    p_index = sub.add_parser("index", help="Build BM25 + ChromaDB index")
    _add_common(p_index)
    p_index.add_argument("--reset", action="store_true", help="Delete existing index")
    p_index.set_defaults(func=cmd_index)

    # retrieve
    p_ret = sub.add_parser("retrieve", help="Hybrid retrieval → SQLite cache")
    _add_common(p_ret)
    p_ret.add_argument("--no-reranker", action="store_true")
    p_ret.add_argument("--reset", action="store_true")
    p_ret.set_defaults(func=cmd_retrieve)

    # generate
    p_gen = sub.add_parser("generate", help="LLM generation from cache")
    _add_common(p_gen)
    p_gen.add_argument("--reset", action="store_true")
    p_gen.add_argument("--min-articles", type=int, default=None)
    p_gen.add_argument("--max-articles", type=int, default=None)
    p_gen.add_argument("--safe-threshold", type=float, default=None)
    p_gen.set_defaults(func=cmd_generate)

    # submit
    p_sub = sub.add_parser("submit", help="Validate + package submission.zip")
    _add_common(p_sub)
    p_sub.set_defaults(func=cmd_submit)

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Full end-to-end pipeline")
    _add_common(p_pipe)
    p_pipe.add_argument("--skip-ingest", action="store_true")
    p_pipe.add_argument("--skip-index", action="store_true")
    p_pipe.add_argument("--reset", action="store_true")
    p_pipe.add_argument("--no-reranker", action="store_true")
    p_pipe.add_argument("--min-articles", type=int, default=None)
    p_pipe.add_argument("--max-articles", type=int, default=None)
    p_pipe.add_argument("--safe-threshold", type=float, default=None)
    p_pipe.set_defaults(func=cmd_pipeline)

    # eval
    p_eval = sub.add_parser("eval", help="Local F2 evaluation")
    p_eval.add_argument("--pred", required=True, help="Path to predictions JSON")
    p_eval.add_argument("--ref", required=True, help="Path to reference JSON")
    p_eval.set_defaults(func=cmd_eval)

    # retune
    p_tune = sub.add_parser("retune", help="Tune thresholds without rerunning LLM")
    _add_common(p_tune)
    p_tune.add_argument("--min-articles", type=int, default=3)
    p_tune.add_argument("--max-articles", type=int, default=8)
    p_tune.add_argument("--safe-threshold", type=float, default=0.3)
    p_tune.add_argument("--high-conf-threshold", type=float, default=0.5)
    p_tune.set_defaults(func=cmd_retune)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
