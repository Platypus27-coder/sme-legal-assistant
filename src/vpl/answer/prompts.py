"""
VietPhapLy RAG — prompt templates cho Gemma-2-9B-it.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Sequence


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý AI chuyên về pháp luật Việt Nam dành cho doanh nghiệp nhỏ và vừa (SME). Nhiệm vụ của bạn là trả lời câu hỏi pháp lý một cách chính xác, trực tiếp dựa trên tài liệu được cung cấp.

QUY TẮC BẮT BUỘC:
1. Trả lời trực tiếp vào câu hỏi bằng tiếng Việt. Không tóm tắt chung chung.
2. Chỉ sử dụng thông tin trong [NGỮ CẢNH] để trả lời. Không bịa đặt điều luật, số hiệu văn bản.
3. Khi trích dẫn, phải nêu rõ "Điều X" và tên văn bản tương ứng từ danh sách [TRÍCH DẪN HỢP LỆ].
4. Nếu ngữ cảnh không đủ căn cứ, nêu rõ: "Hệ thống dữ liệu chưa ghi nhận quy định cụ thể cho tình huống này."
5. Kết thúc bằng cảnh báo giới hạn chuẩn.
6. Viết toàn bộ bằng tiếng Việt."""

ANSWER_FORMAT = """Trả lời theo cấu trúc:
1. **Căn cứ pháp lý**: Liệt kê Điều X của văn bản nào áp dụng.
2. **Phân tích**: Giải thích cách áp dụng vào tình huống được hỏi.
3. **Tư vấn thực tế**: Hướng xử lý cụ thể cho doanh nghiệp.
4. **Cảnh báo giới hạn**: Đây là tư vấn sơ bộ từ AI, doanh nghiệp cần đối chiếu văn bản gốc hoặc tham khảo chuyên gia pháp lý trước khi áp dụng."""


def _article_sort_key(article: str) -> tuple[int, str]:
    m = re.search(r"(\d+)\s*([A-Za-z]?)", article or "")
    return (int(m.group(1)), m.group(2).lower()) if m else (10**9, article)


def format_context(chunks: Sequence[Any], max_chars: int | None = None) -> str:
    """
    Group chunks by doc_id để tránh nhầm số Điều giữa các văn bản.
    Tránh trường hợp nhầm lẫn Điều 4 Luật A với Điều 4 Luật B.
    """
    groups: OrderedDict[str, list] = OrderedDict()
    for c in chunks:
        meta = c.metadata if hasattr(c, "metadata") else (c.get("metadata") or {})
        doc_id = str(meta.get("doc_id") or "UNKNOWN")
        groups.setdefault(doc_id, []).append((c, meta))

    blocks = ["=== NGỮ CẢNH PHÁP LÝ ==="]
    for idx, (doc_id, items) in enumerate(groups.items(), 1):
        first_meta = items[0][1]
        doc_type = str(first_meta.get("doc_type") or "Văn bản")
        doc_title = str(first_meta.get("doc_title") or doc_id)
        blocks.append(f"\n[TÀI LIỆU {idx}]: {doc_type} {doc_id} — {doc_title}")
        for c, meta in sorted(
            items,
            key=lambda x: _article_sort_key(str(x[1].get("article_number") or "")),
        ):
            article = str(meta.get("article_number") or "Nội dung liên quan")
            citation = str(meta.get("formatted_article") or "")
            text = c.chunk.get("text", "") if hasattr(c, "chunk") else str(c.get("text", ""))
            citation_line = (
                f"\n  [TRÍCH DẪN HỢP LỆ]: {citation}"
                if citation and meta.get("submission_eligible", True)
                else ""
            )
            blocks.append(f"- {article}: {text.strip()}{citation_line}")

    context = "\n".join(blocks)
    return context[:max_chars].rstrip() if max_chars else context


def build_messages(question: str, chunks: Sequence[Any], max_context_chars: int | None = None) -> list[dict]:
    context = format_context(chunks, max_chars=max_context_chars)
    user_content = f"""[NGỮ CẢNH]
{context}
[/NGỮ CẢNH]

Dựa trên ngữ cảnh trên, trả lời câu hỏi sau bằng tiếng Việt:
Câu hỏi: {question.strip()}

{ANSWER_FORMAT}"""
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"},
    ]
