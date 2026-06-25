"""
rerank_retune.py — Phương án A+B Standalone

Features:
  - Auto-checkpoint mỗi 50 câu → lưu lên Google Drive
  - Auto-RESUME: nếu bị ngắt, chạy lại sẽ tiếp tục từ câu còn dở
  - Không cần chạy lại LLM Gemma

Usage:
  python rerank_retune.py
  python rerank_retune.py --high-conf 0.62 --safe 0.52 --max-art 3
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Iterable

# ─── Mặc định tốt nhất (0.3976) ─────────────────────────────────────────────
DEFAULT_HIGH_CONF   = 0.62
DEFAULT_SAFE        = 0.52
DEFAULT_MIN_ART     = 0
DEFAULT_MAX_ART     = 3
DEFAULT_CHECKPOINT  = 50   # lưu Drive mỗi 50 câu

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
OUT_DIR     = BASE / "artifacts" / "output"

# 1. Tự động phát hiện chế độ TEST cô lập
_DB_TEST = BASE / "artifacts" / "cache" / "retrieval_test.db"
_DB_PROD = BASE / "artifacts" / "cache" / "retrieval.db"
IS_TEST = _DB_TEST.exists() or Path("/content/drive/MyDrive/R2AI_Artifacts_Test").exists()

# 2. Thiết lập đường dẫn động thông minh
if IS_TEST:
    print("\n🧪 [MODE] Khởi chạy chế độ kiểm thử (TEST isolated mode)")
    CACHE_DB = _DB_TEST
    DRIVE_DIR = Path("/content/drive/MyDrive/R2AI_Artifacts_Test")
    
    # Thứ tự ưu tiên tìm kiếm câu trả lời đầu vào trong chế độ TEST
    candidates = [
        OUT_DIR / "results_retune.json",
        OUT_DIR / "results_test.json",
        OUT_DIR / "results_partial_test.jsonl",
        OUT_DIR / "results_partial.jsonl"
    ]
    ANSWERS_IN = next((c for c in candidates if c.exists()), candidates[0])
    
    OUT_JSON    = OUT_DIR / "results_reranked_test.json"
    OUT_ZIP     = OUT_DIR / "submission_reranked_test.zip"
    CKPT_JSON   = OUT_DIR / "results_reranked_checkpoint_test.json"
    
    DRIVE_CKPT  = DRIVE_DIR / "results_reranked_checkpoint_test.json"
    DRIVE_ZIP   = DRIVE_DIR / "submission_reranked_test.zip"
else:
    print("\n🚀 [MODE] Khởi chạy chế độ chính thức (PROD mode)")
    CACHE_DB = _DB_PROD
    DRIVE_DIR = Path("/content/drive/MyDrive/R2AI_Artifacts")
    
    _ANSWERS_JSON = OUT_DIR / "results_retune.json"
    _ANSWERS_JSONL = OUT_DIR / "results_partial.jsonl"
    ANSWERS_IN = _ANSWERS_JSON if _ANSWERS_JSON.exists() else _ANSWERS_JSONL
    
    OUT_JSON    = OUT_DIR / "results_reranked.json"
    OUT_ZIP     = OUT_DIR / "submission_reranked.zip"
    CKPT_JSON   = OUT_DIR / "results_reranked_checkpoint.json"
    
    DRIVE_CKPT  = DRIVE_DIR / "results_reranked_checkpoint.json"
    DRIVE_ZIP   = DRIVE_DIR / "submission_reranked.zip"

print(f"   📂 Database Cache: {CACHE_DB.name}")
print(f"   📖 Dữ liệu đầu vào: {ANSWERS_IN.name}")
print(f"   📦 File xuất bản: {OUT_ZIP.name}")
print(f"   ☁️ Thư mục Drive: {DRIVE_DIR}\n")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


LEGAL_BASIS_RE = re.compile(
    r"\n+C(?:ơ|Æ¡)\s+s(?:ở|á»Ÿ)\s+ph(?:á|Ă¡)p\s+l(?:ý|Ă½)\s+tham\s+chi(?:ếu|áº¿u):.*?"
    r"(?=\n+C(?:ả|áº£)nh\s+b(?:á|Ă¡)o\s+gi(?:ới|á»›i)\s+h(?:ạn|áº¡n):|\Z)",
    re.IGNORECASE | re.DOTALL,
)

WARNING_RE = re.compile(
    r"\n+C(?:ả|áº£)nh\s+b(?:á|Ă¡)o\s+gi(?:ới|á»›i)\s+h(?:ạn|áº¡n):.*?\Z",
    re.IGNORECASE | re.DOTALL,
)

def _article_label(formatted: str) -> str:
    parts = formatted.split("|")
    return parts[-1] if parts else ""

def _doc_label(formatted: str) -> str:
    parts = formatted.split("|")
    return parts[-1] if len(parts) >= 2 else ""

def _dedupe_list(items: Iterable[str]) -> list[str]:
    seen = set()
    res = []
    for x in items:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res

def _basis_line(selected: list[tuple[float, dict[str, Any]]]) -> str:
    refs = []
    for _, meta in selected:
        article = str(meta.get("article_number") or _article_label(str(meta.get("formatted_article") or "")))
        doc = _doc_label(str(meta.get("formatted_doc") or ""))
        if article and doc:
            refs.append(f"{article} của {doc}")
    refs = _dedupe_list(refs)
    if not refs:
        return ""
    return "\n\nCơ sở pháp lý tham chiếu: " + "; ".join(refs) + "."

def _rewrite_answer(answer: str, selected: list[tuple[float, dict[str, Any]]]) -> str:
    warning_match = WARNING_RE.search(answer)
    warning = warning_match.group(0) if warning_match else ""
    body = WARNING_RE.sub("", answer)
    body = LEGAL_BASIS_RE.sub("", body).rstrip()
    basis = _basis_line(selected)
    return (body + basis + warning).strip()


def save_to_drive(src: Path, dst: Path) -> bool:
    """Copy file lên Drive. Trả về True nếu thành công."""
    if not DRIVE_DIR.parent.exists():
        return False
    try:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"   ⚠️  Drive save failed: {e}")
        return False


def load_answers(path: Path) -> dict[int, dict]:
    print(f"📖 Loading answers from {path.name}...")
    answers = {}
    if not path.exists():
        print(f"   ❌ File not found: {path}")
        return answers
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return answers
        if content.startswith("["):
            # Định dạng JSON Array (results_retune.json)
            try:
                data = json.loads(content)
                for row in data:
                    answers[int(row["id"])] = row
            except Exception as e:
                print(f"   ❌ Lỗi parse JSON Array: {e}")
        else:
            # Định dạng JSON Lines (results_partial.jsonl)
            for i, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        answers[int(row["id"])] = row
                    except Exception as e:
                        print(f"   ⚠️ Lỗi parse dòng {i}: {e}")
    print(f"   ✅ Loaded {len(answers)} answers")
    return answers


def load_cache(path: Path) -> dict[int, list[dict]]:
    print(f"📦 Loading retrieval cache from {path.name}...")
    if not path.exists():
        print(f"\n❌ LỖI NGHIÊM TRỌNG: Không tìm thấy database cache tại: {path}")
        print("   Vui lòng chắc chắn rằng bạn đã chạy Cell 6 (Retrieve) để tạo cache trước khi chạy Reranker!")
        raise FileNotFoundError(f"Database cache not found at {path}")
    conn = sqlite3.connect(path)
    cache = {int(r[0]): json.loads(r[1])
             for r in conn.execute("SELECT question_id, chunks_json FROM retrieval_cache")}
    conn.close()
    print(f"   ✅ {len(cache)} cached questions")
    return cache


def load_checkpoint() -> tuple[dict[int, dict], int]:
    """
    Load kết quả đã làm từ checkpoint.
    Trả về (done_results_dict, count_done).
    Ưu tiên: Drive checkpoint > Local checkpoint
    """
    # Thử restore từ Drive trước
    if DRIVE_CKPT.exists() and not CKPT_JSON.exists():
        print(f"☁️  Restore checkpoint từ Drive ({DRIVE_CKPT.stat().st_size/1024:.0f}KB)...")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DRIVE_CKPT, CKPT_JSON)

    if not CKPT_JSON.exists():
        return {}, 0

    try:
        data = json.loads(CKPT_JSON.read_text(encoding="utf-8"))
        done = {int(r["id"]): r for r in data}
        print(f"🔄 RESUME: Tìm thấy checkpoint với {len(done)} câu đã xong")
        return done, len(done)
    except Exception as e:
        print(f"⚠️  Checkpoint lỗi ({e}), bắt đầu lại từ đầu")
        return {}, 0


def save_checkpoint(results: list[dict], i: int, total: int) -> None:
    """Lưu checkpoint local + Drive."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    ok = save_to_drive(CKPT_JSON, DRIVE_CKPT)
    drive_str = "☁️ → Drive" if ok else "💾 local only"
    pct = i / total * 100
    print(f"   [{i:4d}/{total}] {pct:4.1f}% — checkpoint saved {drive_str}")


def load_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
    print(f"\n🤖 Loading BGE-Reranker ({model_name}) on {device}...")
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(model_name, max_length=512, device=device)
        print("   ✅ Reranker ready!")
        return reranker
    except Exception as e:
        print(f"   ⚠️ Không tải được ({e}) → dùng hybrid score thuần")
        return None


def rerank_chunks(reranker, question: str, chunks: list[dict],
                  batch_size: int = 64) -> list[tuple[float, dict]]:
    """Dedup + rerank → list[(score, metadata)] sorted desc."""
    seen: dict[str, dict] = {}
    for c in chunks:
        meta = c.get("metadata") or c.get("meta") or {}
        art  = str(meta.get("formatted_article") or "")
        if not art or not meta.get("doc_id"):
            continue
        if not meta.get("submission_eligible", True):
            continue
        if float(c.get("score", 0)) > seen.get(art, {}).get("score", -1):
            seen[art] = c

    unique = list(seen.values())
    if not unique:
        return []

    if reranker is not None:
        try:
            pairs  = [(question, str(c.get("text", ""))) for c in unique]
            logits = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            scores = [_sigmoid(float(l)) for l in logits]
        except Exception as e:
            print(f"   ⚠️ rerank error: {e}")
            scores = [float(c.get("score", 0)) for c in unique]
    else:
        scores = [float(c.get("score", 0)) for c in unique]

    out = [(s, (c.get("metadata") or c.get("meta") or {}))
           for s, c in zip(scores, unique)]
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def select_articles(scored, high_conf, safe, min_art, max_art):
    high      = [(s, m) for s, m in scored if s >= high_conf]
    safe_tier = [(s, m) for s, m in scored if s >= safe]
    if high:
        selected = high
    elif safe_tier:
        selected = safe_tier
    else:
        selected = scored[:min_art]
    return selected[:max_art]


def dedupe(lst: list) -> list:
    return list(dict.fromkeys(v for v in lst if v))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-conf",    type=float, default=DEFAULT_HIGH_CONF)
    parser.add_argument("--safe",         type=float, default=DEFAULT_SAFE)
    parser.add_argument("--min-art",      type=int,   default=DEFAULT_MIN_ART)
    parser.add_argument("--max-art",      type=int,   default=DEFAULT_MAX_ART)
    parser.add_argument("--device",       type=str,   default="cuda")
    parser.add_argument("--no-reranker",  action="store_true")
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reset",        action="store_true",
                        help="Xoá checkpoint, chạy lại từ đầu")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 RERANK + RETUNE (Phương án A+B)")
    print(f"   HIGH_CONF = {args.high_conf} | SAFE = {args.safe}")
    print(f"   MAX_ART = {args.max_art}     | MIN_ART = {args.min_art}")
    print(f"   Device = {args.device}       | Checkpoint / {args.checkpoint_every} câu")
    print("=" * 60)

    # Xoá checkpoint nếu --reset
    if args.reset:
        for p in [CKPT_JSON, DRIVE_CKPT]:
            if p.exists():
                p.unlink()
        print("🗑️  Checkpoint đã xoá, bắt đầu lại từ đầu\n")

    # Load dữ liệu
    answers  = load_answers(ANSWERS_IN)
    if not answers:
        print("\n❌ LỖI NGHIÊM TRỌNG: Không tìm thấy bất kỳ câu trả lời đầu vào nào để chạy Reranker!")
        print(f"   Hành lang kiểm tra tại: {ANSWERS_IN}")
        print("   👉 Giải pháp: Vui lòng chắc chắn rằng bạn đã có tệp câu trả lời từ lần chạy trước")
        print("   (ví dụ: results_retune.json hoặc results_partial_test.jsonl) trong thư mục artifacts/output/ hoặc Google Drive.")
        raise SystemExit(1)
    cache    = load_cache(CACHE_DB)
    reranker = None if args.no_reranker else load_reranker(device=args.device)

    # Load checkpoint (resume)
    done_map, n_done = load_checkpoint()

    # Build danh sách câu cần làm (bỏ qua câu đã có trong checkpoint)
    todo = [(qid, row) for qid, row in sorted(answers.items())
            if qid not in done_map]
    results = list(done_map.values())  # bắt đầu từ kết quả đã có
    total   = len(answers)

    print(f"\n🔧 Còn {len(todo)}/{total} câu chưa làm...")
    if n_done > 0:
        print(f"   (Đã có {n_done} câu từ checkpoint trước)")

    zero_articles = 0
    skipped = 0

    for step, (qid, row) in enumerate(todo, 1):
        chunks = cache.get(qid)
        if chunks is None:
            results.append(row)
            skipped += 1
        else:
            question = str(row.get("question", ""))
            scored   = rerank_chunks(reranker, question, chunks, args.batch_size)
            selected = select_articles(scored, args.high_conf, args.safe,
                                       args.min_art, args.max_art)

            if not selected:
                zero_articles += 1

            results.append({
                "id":                row["id"],
                "question":          row["question"],
                "answer":            _rewrite_answer(str(row.get("answer") or ""), selected),
                "relevant_docs":     dedupe([str(m.get("formatted_doc")     or "") for _, m in selected]),
                "relevant_articles": dedupe([str(m.get("formatted_article") or "") for _, m in selected]),
            })

        # Checkpoint
        global_i = n_done + step
        if step % args.checkpoint_every == 0 or step == len(todo):
            save_checkpoint(results, global_i, total)

    # Xuất file cuối
    print(f"\n💾 Xuất kết quả cuối...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Chạy validation
    try:
        from vpl.submit import validate
        q_path = BASE / "data" / "R2AIStage1DATA.json"
        if q_path.exists():
            questions = json.loads(q_path.read_text(encoding="utf-8"))
            errors = validate(results, questions)
            if errors:
                print(f"⚠️ Validation warnings ({len(errors)}):")
                for e in errors[:20]:
                    print(f"  - {e}")
            else:
                print("✅ Submission validation passed with 0 errors!")
        else:
            print(f"⚠️ Không tìm thấy file đề thi {q_path.name} để validate.")
    except Exception as e:
        print(f"⚠️ Không thể validate: {e}")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_JSON, "results.json")

    save_to_drive(OUT_JSON, DRIVE_DIR / OUT_JSON.name)
    ok = save_to_drive(OUT_ZIP, DRIVE_ZIP)

    # Xoá checkpoint sau khi xong
    if CKPT_JSON.exists():
        CKPT_JSON.unlink()
    if DRIVE_CKPT.exists():
        DRIVE_CKPT.unlink()

    # Stats
    import statistics
    art_counts = [len(r.get("relevant_articles", [])) for r in results]
    print(f"\n✅ DONE! ({len(results)} câu)")
    print(f"   Zero articles: {zero_articles} | Skipped (no cache): {skipped}")
    print(f"   Articles/câu — Min:{min(art_counts)} Max:{max(art_counts)} "
          f"Mean:{statistics.mean(art_counts):.2f}")
    dist = {k: art_counts.count(k) for k in sorted(set(art_counts))}
    print(f"   Phân bố: {dist}")
    print(f"\n🏆 File nộp: {OUT_ZIP}")
    if ok:
        print(f"   ☁️  Drive: {DRIVE_ZIP}")


if __name__ == "__main__":
    main()
