"""
Error Analysis Script — Phân tích lỗi Retrieval trên file temp.json
Mục tiêu: Xác định BM25 yếu ở đâu, BGE-M3 có giúp được không.
"""

import json
import re
import sys
import io
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
PROJECT = Path(__file__).resolve().parents[1]
TEMP_JSON = PROJECT / "artifacts" / "output" / "temp.json"
QUESTIONS_FILE = PROJECT / "data" / "R2AIStage1DATA.json"

# Legal keyword patterns
LEGAL_REF_PATTERNS = [
    re.compile(r"(?:Điều|điều)\s+\d+[a-zA-Z]?", re.IGNORECASE),
    re.compile(r"\d{1,3}/\d{4}/[\wĐđ-]+", re.IGNORECASE),  # e.g. 80/2021/NĐ-CP
    re.compile(r"(?:Luật|Nghị định|Thông tư|Bộ luật|Quyết định|Pháp lệnh)", re.IGNORECASE),
]

SITUATION_KEYWORDS = [
    "phải làm gì", "cần làm gì", "thủ tục", "hướng dẫn",
    "có được phép", "có quyền", "có phải", "có cần",
    "bị phạt", "xử phạt", "mức phạt",
    "muốn mở", "muốn thành lập", "muốn đăng ký",
    "tôi muốn", "doanh nghiệp muốn", "công ty muốn",
    "trường hợp nào", "điều kiện", "quy định",
    "như thế nào", "ra sao", "bao lâu", "bao nhiêu",
]


def classify_question(question: str) -> str:
    """Phân loại câu hỏi: 'specific' (có tham chiếu luật cụ thể) vs 'situational' (mô tả tình huống)."""
    has_legal_ref = any(p.search(question) for p in LEGAL_REF_PATTERNS)
    has_situation = any(kw in question.lower() for kw in SITUATION_KEYWORDS)
    
    if has_legal_ref and not has_situation:
        return "specific"       # Hỏi thẳng về điều luật cụ thể
    elif has_legal_ref and has_situation:
        return "mixed"          # Có cả tham chiếu luật + tình huống
    elif has_situation:
        return "situational"    # Mô tả tình huống, không nhắc luật
    else:
        return "general"        # Câu hỏi chung chung


def analyze():
    # Load data
    results = json.loads(TEMP_JSON.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    q_map = {int(q["id"]): q["question"] for q in questions}
    
    print(f"Total results: {len(results)}")
    print(f"Total questions: {len(questions)}")
    print()
    
    # --- 1. Phân loại câu hỏi ---
    q_types = Counter()
    type_article_counts = {}  # type -> list of article counts
    
    for r in results:
        qid = int(r["id"])
        question = q_map.get(qid, r["question"])
        qtype = classify_question(question)
        q_types[qtype] += 1
        type_article_counts.setdefault(qtype, []).append(len(r["relevant_articles"]))
    
    print("=" * 70)
    print("1. PHAN LOAI CAU HOI")
    print("=" * 70)
    for qtype, count in q_types.most_common():
        avg_arts = sum(type_article_counts[qtype]) / len(type_article_counts[qtype])
        print(f"  {qtype:15s}: {count:5d} cau ({count/len(results)*100:.1f}%) | avg articles: {avg_arts:.2f}")
    
    # --- 2. Phân bố số lượng relevant_articles ---
    art_counts = Counter(len(r["relevant_articles"]) for r in results)
    print()
    print("=" * 70)
    print("2. PHAN BO SO LUONG RELEVANT_ARTICLES")
    print("=" * 70)
    for n in sorted(art_counts.keys()):
        bar = "#" * (art_counts[n] // 10)
        print(f"  {n} articles: {art_counts[n]:5d} cau | {bar}")
    
    # --- 3. Câu hỏi có ít articles nhất (BM25 yếu nhất) ---
    few_articles = [r for r in results if len(r["relevant_articles"]) <= 1]
    
    print()
    print("=" * 70)
    print(f"3. CAU HOI CO <= 1 RELEVANT_ARTICLES ({len(few_articles)} cau)")
    print("   (Day la nhung cau BM25 gap kho khan nhat)")
    print("=" * 70)
    
    few_types = Counter()
    for r in few_articles:
        qtype = classify_question(q_map.get(int(r["id"]), r["question"]))
        few_types[qtype] += 1
    
    for qtype, count in few_types.most_common():
        pct_of_type = count / q_types[qtype] * 100
        print(f"  {qtype:15s}: {count:5d} cau ({pct_of_type:.1f}% cua loai nay bi BM25 yeu)")
    
    # --- 4. Phân tích chi tiết 20 câu có ít articles nhất ---
    print()
    print("=" * 70)
    print("4. MAU 20 CAU HOI CO IT ARTICLES NHAT (BM25 failures)")
    print("=" * 70)
    
    one_art = [r for r in results if len(r["relevant_articles"]) == 1][:10]
    zero_art = [r for r in results if len(r["relevant_articles"]) == 0][:10]
    
    samples = zero_art + one_art
    for i, r in enumerate(samples[:20], 1):
        qid = int(r["id"])
        question = q_map.get(qid, r["question"])
        qtype = classify_question(question)
        arts = r["relevant_articles"]
        
        # Truncate question for display
        q_short = question[:120] + "..." if len(question) > 120 else question
        print(f"\n  [{i}] ID={qid} | Type={qtype} | Articles={len(arts)}")
        print(f"      Q: {q_short}")
        if arts:
            for a in arts:
                print(f"      -> {a}")
    
    # --- 5. Phân tích tần suất văn bản được trích dẫn ---
    print()
    print("=" * 70)
    print("5. TOP 20 VAN BAN DUOC TRICH DAN NHIEU NHAT")
    print("=" * 70)
    
    doc_counter = Counter()
    for r in results:
        for doc in r.get("relevant_docs", []):
            doc_counter[doc] += 1
    
    for doc, count in doc_counter.most_common(20):
        # Extract just the title part
        parts = doc.split("|")
        title = parts[-1].strip() if len(parts) > 1 else doc
        print(f"  {count:5d}x | {title[:80]}")
    
    # --- 6. Phân tích câu hỏi situational vs specific ---
    print()
    print("=" * 70)
    print("6. KET LUAN: BGE-M3 CO GIUP DUOC KHONG?")
    print("=" * 70)
    
    situational_few = sum(1 for r in few_articles 
                          if classify_question(q_map.get(int(r["id"]), r["question"])) == "situational")
    specific_few = sum(1 for r in few_articles 
                       if classify_question(q_map.get(int(r["id"]), r["question"])) == "specific")
    
    total_few = len(few_articles)
    if total_few > 0:
        print(f"\n  Trong {total_few} cau BM25 yeu (<=1 article):")
        print(f"    - Situational (BGE-M3 giup duoc):  {situational_few} ({situational_few/total_few*100:.1f}%)")
        print(f"    - Specific    (BGE-M3 khong giup): {specific_few} ({specific_few/total_few*100:.1f}%)")
        print(f"    - Mixed/General:                   {total_few - situational_few - specific_few} ({(total_few - situational_few - specific_few)/total_few*100:.1f}%)")
        
        if situational_few / total_few > 0.5:
            print(f"\n  => DA SO la situational => BGE-M3 CO KHA NANG GIUP TANG RECALL")
        elif specific_few / total_few > 0.5:
            print(f"\n  => DA SO la specific => BGE-M3 IT GIUP, can sua CHUNKING/CORPUS")
        else:
            print(f"\n  => HON HOP => BGE-M3 giup 1 phan, can ket hop voi cai thien khac")
    
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    analyze()
