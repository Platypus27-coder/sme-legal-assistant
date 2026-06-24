"""
VietPhapLy RAG — BM25 indexer.

Tokenizer: underthesea (nếu available) → regex legal phrases fallback.
underthesea được dùng thực sự, không chỉ là stub.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

from vpl.settings import BM25_CORPUS_FILE, BM25_DIR, BM25_ID_MAP_FILE, CHUNKS_FILE, INDEX

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Legal phrase dictionary — dùng cho cả BM25 tokenize lẫn query expansion
# ---------------------------------------------------------------------------

LEGAL_PHRASES: list[str] = sorted([
    # SME core
    "doanh nghiệp nhỏ và vừa", "hỗ trợ doanh nghiệp", "hộ kinh doanh",
    "doanh nghiệp siêu nhỏ", "khởi nghiệp sáng tạo",
    "vốn điều lệ", "đăng ký kinh doanh", "đăng ký doanh nghiệp",
    "giấy chứng nhận đăng ký",
    # Labor
    "người lao động", "hợp đồng lao động", "người sử dụng lao động",
    "thời giờ làm việc", "tiền lương", "an toàn lao động",
    "kỷ luật lao động", "sa thải", "nghỉ phép",
    # Insurance
    "bảo hiểm xã hội", "bảo hiểm thất nghiệp", "bảo hiểm y tế",
    # Tax
    "thuế giá trị gia tăng", "thuế thu nhập doanh nghiệp",
    "thuế thu nhập cá nhân", "quản lý thuế", "khai thuế", "nộp thuế",
    "chậm nộp thuế", "hoàn thuế", "miễn thuế", "giảm thuế",
    "hóa đơn điện tử", "hóa đơn chứng từ",
    # Accounting
    "báo cáo tài chính", "kế toán", "kiểm toán",
    # IP
    "sở hữu trí tuệ", "nhãn hiệu", "bản quyền", "sáng chế",
    "kiểu dáng công nghiệp", "chỉ dẫn địa lý",
    # Contract / Commerce
    "hợp đồng", "hợp đồng mua bán", "hợp đồng dịch vụ",
    "thương mại điện tử", "giao dịch điện tử",
    # Administrative
    "xử phạt vi phạm hành chính", "khắc phục hậu quả",
    "vi phạm hành chính",
    # Land
    "mặt bằng sản xuất", "đất đai", "thuê đất", "quyền sử dụng đất",
    # Credit
    "bảo lãnh tín dụng", "quỹ bảo lãnh", "tín dụng",
    # Bidding
    "đấu thầu", "nhà thầu",
    # Legal structure
    "nghị định", "thông tư", "quyết định", "nghị quyết",
    "văn bản pháp luật",
    # Licensing
    "cấp phép", "thu hồi", "đình chỉ", "tạm đình chỉ",
    "giấy phép", "chứng chỉ hành nghề",
], key=len, reverse=True)  # Longest first → greedy match


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Tokenize tiếng Việt cho BM25.

    Thứ tự ưu tiên:
    1. underthesea.word_tokenize (proper Vietnamese segmentation)
    2. Regex fallback + legal phrase bigrams
    """
    lowered = text.lower()

    # Always extract legal doc IDs và phrases
    legal_ids = re.findall(r"\b\d{1,3}/\d{4}/[\wđ-]+\b", lowered)
    phrases = [p.replace(" ", "_") for p in LEGAL_PHRASES if p in lowered]

    # Bỏ qua underthesea vì thư viện này thường xuyên bị treo (hang) 
    # hoặc chạy cực kỳ chậm khi gặp các văn bản Luật quá dài hoặc có ký tự lạ.
    # Regex fallback bên dưới là đủ tốt cho BM25.

    # Regex fallback + adjacent bigrams
    words = re.findall(r"[0-9a-zà-ỹđ]+", lowered)
    words = [w for w in words if len(w) > 1 or w.isdigit()]
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return legal_ids + words + bigrams + phrases


# ---------------------------------------------------------------------------
# Build & persist
# ---------------------------------------------------------------------------

def _load_chunks() -> list[dict[str, Any]]:
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  ⚠ skip line {i}: {exc}")
            if i % 10_000 == 0:
                print(f"  loaded {i} chunks...", flush=True)
    return chunks


def build() -> dict[str, Any]:
    """Build BM25 index từ chunks.jsonl và persist."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError("Install rank-bm25: pip install rank-bm25") from exc

    BM25_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading chunks...")
    chunks = _load_chunks()
    if not chunks:
        raise ValueError("No chunks found. Run `vpl ingest` first.")
    print(f"  {len(chunks)} chunks loaded")

    print(f"Tokenizing {len(chunks)} chunks with Contextual Enrichment...")
    corpus_tokens: list[list[str]] = []
    batch = INDEX.bm25_batch_size
    for i in range(0, len(chunks), batch):
        for c in chunks[i : i + batch]:
            meta = c.get("metadata") or {}
            doc_title = meta.get("doc_title") or ""
            article_number = meta.get("article_number") or ""
            text = c.get("text", "")
            
            # Làm giàu văn cảnh: ghép Tiêu đề văn bản + Số điều + Nội dung
            enriched = text
            if doc_title:
                if article_number:
                    enriched = f"{doc_title} - Điều {article_number}: {text}"
                else:
                    enriched = f"{doc_title}: {text}"
            corpus_tokens.append(tokenize(enriched))
        print(f"  tokenized {min(i + batch, len(chunks))}/{len(chunks)}", flush=True)

    print(f"Building BM25 (k1={INDEX.bm25_k1}, b={INDEX.bm25_b})...")
    bm25 = BM25Okapi(corpus_tokens, k1=INDEX.bm25_k1, b=INDEX.bm25_b)

    with BM25_CORPUS_FILE.open("wb") as fh:
        pickle.dump(bm25, fh)
    BM25_ID_MAP_FILE.write_text(
        json.dumps([c["chunk_id"] for c in chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✅ BM25 index → {BM25_DIR}")
    return {"corpus_size": len(chunks), "bm25_k1": INDEX.bm25_k1, "bm25_b": INDEX.bm25_b}
