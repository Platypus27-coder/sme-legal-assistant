"""
Retune standalone — đọc results_partial.jsonl + retrieval.db
Áp threshold mới, xuất results_retune.json
Không cần GPU, không cần chromadb, không cần embedding model.
"""
import json
import sqlite3
import re
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
HIGH_CONF  = 0.62   # score >= này → vào relevant_articles
SAFE       = 0.52   # score >= này → vào LLM context (fallback)
MIN_ART    = 0      # cho phép nộp mảng rỗng nếu không tìm thấy luật
MAX_ART    = 3      # tối đa 3 articles mỗi câu (ép Precision cực độ)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
# Ưu tiên results_retune.json (đầy đủ 2000 câu từ repo) hơn results_partial.jsonl
_ANSWERS_JSON = BASE / "artifacts" / "output" / "results_retune.json"
_ANSWERS_JSONL = BASE / "artifacts" / "output" / "results_partial.jsonl"
ANSWERS_IN = _ANSWERS_JSON if _ANSWERS_JSON.exists() else _ANSWERS_JSONL

CACHE_DB    = BASE / "artifacts" / "cache" / "retrieval.db"
OUT_JSON    = BASE / "artifacts" / "output" / "results_retune.json"

# ─── Load existing answers ────────────────────────────────────────────────────
print(f"📖 Loading answers from {ANSWERS_IN.name}...")
answers = {}
if not ANSWERS_IN.exists():
    print(f"   ❌ File not found: {ANSWERS_IN}")
else:
    with open(ANSWERS_IN, encoding="utf-8") as f:
        content = f.read().strip()
        if content:
            if content.startswith("["):
                try:
                    data = json.loads(content)
                    for row in data:
                        answers[int(row["id"])] = row
                except Exception as e:
                    print(f"   ❌ Lỗi parse JSON Array: {e}")
            else:
                for i, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if line:
                        try:
                            row = json.loads(line)
                            answers[int(row["id"])] = row
                        except Exception as e:
                            pass
print(f"   {len(answers)} answers loaded")

# ─── Load retrieval cache ─────────────────────────────────────────────────────
print("📦 Loading retrieval cache...")
conn = sqlite3.connect(CACHE_DB)
cursor = conn.cursor()
cursor.execute("SELECT question_id, chunks_json FROM retrieval_cache")
cache = {int(r[0]): json.loads(r[1]) for r in cursor.fetchall()}
conn.close()
print(f"   {len(cache)} cached questions")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def select_articles(chunks):
    """Chọn articles với threshold mới."""
    # Sort by score descending
    ranked = sorted(chunks, key=lambda c: float(c.get("score", 0)), reverse=True)

    # Dedup by formatted_article
    seen = set()
    unique = []
    for c in ranked:
        meta = c.get("metadata") or c.get("meta") or {}
        art  = str(meta.get("formatted_article") or "")
        doc_id = str(meta.get("doc_id") or "")
        eligible = meta.get("submission_eligible", True)
        if art and doc_id and eligible and art not in seen:
            seen.add(art)
            unique.append((float(c.get("score", 0)), meta))

    # Apply threshold
    high_conf = [(s, m) for s, m in unique if s >= HIGH_CONF]
    safe_tier = [(s, m) for s, m in unique if s >= SAFE]

    if len(high_conf) > 0:
        selected = high_conf
    elif len(safe_tier) > 0:
        selected = safe_tier
    else:
        selected = unique[:MIN_ART]  # fallback cuối cùng

    return selected[:MAX_ART]

def dedupe(lst):
    return list(dict.fromkeys(v for v in lst if v))

# ─── Retune ───────────────────────────────────────────────────────────────────
print("🔧 Retuning...")
results = []
skipped = 0
zero_articles = 0

for qid, row in sorted(answers.items()):
    chunks = cache.get(qid)
    if chunks is None:
        # Không có cache → giữ nguyên bản cũ
        results.append(row)
        skipped += 1
        continue

    selected = select_articles(chunks)

    if len(selected) == 0:
        zero_articles += 1

    rel_docs     = dedupe([str(m.get("formatted_doc") or "") for _, m in selected])
    rel_articles = dedupe([str(m.get("formatted_article") or "") for _, m in selected])

    new_row = {
        "id":               row["id"],
        "question":         row["question"],
        "answer":           row["answer"],
        "relevant_docs":    rel_docs,
        "relevant_articles": rel_articles,
    }
    results.append(new_row)

# ─── Write output ─────────────────────────────────────────────────────────────
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done!")
print(f"   Total:          {len(results)}")
print(f"   Skipped (no cache): {skipped}")
print(f"   Zero articles:  {zero_articles}")
print(f"   Output → {OUT_JSON}")

# Quick stats
art_counts = [len(r["relevant_articles"]) for r in results]
import statistics
print(f"\n📊 Articles per question:")
print(f"   Min:    {min(art_counts)}")
print(f"   Max:    {max(art_counts)}")
print(f"   Mean:   {statistics.mean(art_counts):.2f}")
print(f"   Median: {statistics.median(art_counts):.1f}")
print(f"   Dist:   { {k: art_counts.count(k) for k in sorted(set(art_counts))} }")
