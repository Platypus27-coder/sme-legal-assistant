"""
rerank_retune.py — Phương án A+B Standalone

Chạy SAU KHI đã có retrieval.db mới từ Hybrid Search.
KHÔNG cần GPU mạnh, KHÔNG cần chạy lại LLM Gemma.

Pipeline:
  1. Đọc retrieval.db (kết quả Hybrid BM25+Dense)
  2. Với mỗi câu hỏi: BGE-Reranker chấm điểm lại các điều luật ứng viên
  3. Áp threshold → chọn relevant_articles
  4. Ghép với câu trả lời gốc từ results_partial.jsonl
  5. Xuất submission.zip

Usage:
  python rerank_retune.py
  python rerank_retune.py --high-conf 0.62 --safe 0.52 --max-art 3
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

# ─── Mặc định tốt nhất tìm được (0.3976) ────────────────────────────────────
DEFAULT_HIGH_CONF = 0.62
DEFAULT_SAFE      = 0.52
DEFAULT_MIN_ART   = 0
DEFAULT_MAX_ART   = 3

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
ANSWERS_IN  = BASE / "artifacts" / "output" / "results_partial.jsonl"
CACHE_DB    = BASE / "artifacts" / "cache" / "retrieval.db"
OUT_JSON    = BASE / "artifacts" / "output" / "results_reranked.json"
OUT_ZIP     = BASE / "artifacts" / "output" / "submission_reranked.zip"

# Google Drive path (tự động dùng nếu chạy trên Colab)
DRIVE_DIR   = Path("/content/drive/MyDrive/R2AI_Artifacts")


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def load_answers(path: Path) -> dict[int, dict]:
    """Đọc results_partial.jsonl → dict[id → row]."""
    print(f"📖 Loading answers from {path.name}...")
    answers = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                answers[int(row["id"])] = row
    print(f"   ✅ {len(answers)} answers loaded")
    return answers


def load_cache(path: Path) -> dict[int, list[dict]]:
    """Đọc retrieval.db → dict[question_id → list of chunks]."""
    print(f"📦 Loading retrieval cache from {path.name}...")
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT question_id, chunks_json FROM retrieval_cache")
    cache = {int(r[0]): json.loads(r[1]) for r in cursor.fetchall()}
    conn.close()
    print(f"   ✅ {len(cache)} cached questions")
    return cache


def load_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
    """Tải BGE-Reranker model."""
    print(f"\n🤖 Loading BGE-Reranker: {model_name} on {device}...")
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(
            model_name,
            max_length=512,
            device=device,
        )
        print("   ✅ Reranker loaded!")
        return reranker
    except Exception as e:
        print(f"   ⚠️ Không tải được Reranker: {e}")
        print("   → Tiếp tục với BM25/Hybrid score thuần túy (không rerank)")
        return None


def rerank_chunks(
    reranker,
    question: str,
    chunks: list[dict],
    batch_size: int = 64,
) -> list[tuple[float, dict]]:
    """
    Dùng BGE-Reranker chấm điểm lại từng cặp (câu hỏi, điều luật).
    Trả về list[(rerank_score, metadata)] đã sort descending.
    """
    # Dedup theo formatted_article, giữ chunk có score cao nhất
    seen: dict[str, dict] = {}
    for c in chunks:
        meta = c.get("metadata") or c.get("meta") or {}
        art = str(meta.get("formatted_article") or "")
        doc_id = str(meta.get("doc_id") or "")
        if not art or not doc_id:
            continue
        if not meta.get("submission_eligible", True):
            continue
        key = art
        old_score = seen.get(key, {}).get("score", -1)
        if float(c.get("score", 0)) > old_score:
            seen[key] = c

    unique_chunks = list(seen.values())
    if not unique_chunks:
        return []

    texts = [str(c.get("text", ""))[:400] for c in unique_chunks]
    pairs = [(question, t) for t in texts]

    if reranker is not None:
        try:
            logits = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            scores = [_sigmoid(float(l)) for l in logits]
        except Exception as e:
            print(f"   ⚠️ Rerank error: {e}, fallback to BM25 score")
            scores = [float(c.get("score", 0)) for c in unique_chunks]
    else:
        scores = [float(c.get("score", 0)) for c in unique_chunks]

    results = []
    for score, c in zip(scores, unique_chunks):
        meta = c.get("metadata") or c.get("meta") or {}
        results.append((score, meta))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def select_articles(
    scored: list[tuple[float, dict]],
    high_conf: float,
    safe: float,
    min_art: int,
    max_art: int,
) -> list[tuple[float, dict]]:
    """Chọn relevant_articles dựa trên threshold."""
    high = [(s, m) for s, m in scored if s >= high_conf]
    safe_tier = [(s, m) for s, m in scored if s >= safe]

    if len(high) > 0:
        selected = high
    elif len(safe_tier) > 0:
        selected = safe_tier
    else:
        selected = scored[:min_art]  # fallback cuối cùng

    return selected[:max_art]


def dedupe(lst: list) -> list:
    return list(dict.fromkeys(v for v in lst if v))


def save_to_drive(src: Path, drive_dir: Path, label: str = "") -> None:
    """Sao lưu file lên Google Drive nếu đang chạy trên Colab."""
    if not drive_dir.exists():
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    dst = drive_dir / src.name
    import shutil
    shutil.copy2(src, dst)
    print(f"   ☁️  Saved to Drive: {dst} {label}")


def main():
    parser = argparse.ArgumentParser(description="Rerank + Retune (Phương án A+B)")
    parser.add_argument("--high-conf", type=float, default=DEFAULT_HIGH_CONF)
    parser.add_argument("--safe",      type=float, default=DEFAULT_SAFE)
    parser.add_argument("--min-art",   type=int,   default=DEFAULT_MIN_ART)
    parser.add_argument("--max-art",   type=int,   default=DEFAULT_MAX_ART)
    parser.add_argument("--device",    type=str,   default="cuda")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Bỏ qua reranker, chỉ dùng hybrid score")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 RERANK + RETUNE (Phương án A+B)")
    print(f"   HIGH_CONF = {args.high_conf}")
    print(f"   SAFE      = {args.safe}")
    print(f"   MIN_ART   = {args.min_art}")
    print(f"   MAX_ART   = {args.max_art}")
    print(f"   Device    = {args.device}")
    print("=" * 60)

    # Load data
    answers = load_answers(ANSWERS_IN)
    cache   = load_cache(CACHE_DB)

    # Load reranker
    reranker = None if args.no_reranker else load_reranker(device=args.device)

    # Process
    print(f"\n🔧 Processing {len(answers)} questions...")
    results = []
    zero_articles = 0
    skipped = 0
    checkpoint_every = 100

    for i, (qid, row) in enumerate(sorted(answers.items()), 1):
        chunks = cache.get(qid)
        if chunks is None:
            results.append(row)
            skipped += 1
            continue

        question = str(row.get("question", ""))

        # Rerank
        scored = rerank_chunks(reranker, question, chunks, batch_size=args.batch_size)

        # Select
        selected = select_articles(
            scored,
            high_conf=args.high_conf,
            safe=args.safe,
            min_art=args.min_art,
            max_art=args.max_art,
        )

        if len(selected) == 0:
            zero_articles += 1

        rel_docs     = dedupe([str(m.get("formatted_doc") or "")     for _, m in selected])
        rel_articles = dedupe([str(m.get("formatted_article") or "") for _, m in selected])

        results.append({
            "id":               row["id"],
            "question":         row["question"],
            "answer":           row["answer"],
            "relevant_docs":    rel_docs,
            "relevant_articles": rel_articles,
        })

        # Progress + checkpoint mỗi 100 câu
        if i % checkpoint_every == 0 or i == len(answers):
            pct = i / len(answers) * 100
            print(f"   [{i:4d}/{len(answers)}] {pct:5.1f}% | "
                  f"zero_arts={zero_articles} | skipped={skipped}")

            # Checkpoint: lưu tạm lên Drive
            if len(results) > 0:
                OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path = OUT_JSON.parent / f"results_reranked_ckpt_{i}.json"
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                save_to_drive(ckpt_path, DRIVE_DIR, f"(checkpoint {i})")

    # Write final output
    print(f"\n💾 Writing final output...")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Package zip
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_JSON, "results.json")

    # Save to Drive
    save_to_drive(OUT_JSON, DRIVE_DIR, "(FINAL)")
    save_to_drive(OUT_ZIP,  DRIVE_DIR, "(SUBMIT THIS!)")

    # Stats
    art_counts = [len(r["relevant_articles"]) for r in results]
    import statistics
    print(f"\n✅ DONE!")
    print(f"   Total:          {len(results)}")
    print(f"   Zero articles:  {zero_articles}")
    print(f"   Skipped:        {skipped}")
    print(f"\n📊 Articles per question:")
    print(f"   Min:    {min(art_counts)}")
    print(f"   Max:    {max(art_counts)}")
    print(f"   Mean:   {statistics.mean(art_counts):.2f}")
    print(f"   Median: {statistics.median(art_counts):.1f}")
    dist = {k: art_counts.count(k) for k in sorted(set(art_counts))}
    print(f"   Dist:   {dist}")
    print(f"\n🏆 Submit file: {OUT_ZIP}")
    if DRIVE_DIR.exists():
        print(f"   ☁️  Also saved to Drive: {DRIVE_DIR / OUT_ZIP.name}")


if __name__ == "__main__":
    main()
