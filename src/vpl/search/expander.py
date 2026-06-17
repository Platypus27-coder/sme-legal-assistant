"""
VietPhapLy RAG — HyDE query expansion.

HyDE (Hypothetical Document Embeddings):
  Input:  câu hỏi người dùng
  Output: hypothetical answer ngắn (~100 tokens) do LLM sinh ra
  Dùng:   embed (query + hypothetical) → dense retrieval tốt hơn


Repo tham khảo dùng hard-coded dict mapping.
"""

from __future__ import annotations

from vpl.settings import SEARCH


# ---------------------------------------------------------------------------
# Domain expansion dict
# Dùng làm fallback khi HyDE không available
# ---------------------------------------------------------------------------

_DOMAIN_EXPANSIONS: dict[tuple[str, ...], str] = {
    ("doanh nghiệp nhỏ và vừa", "dnnvv", "khởi nghiệp"): "Luật hỗ trợ doanh nghiệp nhỏ và vừa nghị định hướng dẫn",
    ("người lao động", "hợp đồng lao động", "tiền lương"): "Bộ luật Lao động xử phạt vi phạm lao động",
    ("bảo hiểm xã hội", "bhxh", "bảo hiểm thất nghiệp"): "Luật Bảo hiểm xã hội xử phạt chậm đóng",
    ("thuế", "khai thuế", "nộp thuế", "chậm nộp"): "Luật Quản lý thuế xử phạt vi phạm hành chính",
    ("hóa đơn", "chứng từ"): "hóa đơn chứng từ xử phạt vi phạm hành chính",
    ("kế toán", "báo cáo tài chính"): "Luật Kế toán chuẩn mực báo cáo tài chính",
    ("nhãn hiệu", "sở hữu trí tuệ", "bản quyền"): "Luật Sở hữu trí tuệ",
    ("hợp đồng", "thương mại", "mua bán"): "Luật Thương mại Bộ luật Dân sự hợp đồng",
    ("xử phạt", "vi phạm", "khắc phục hậu quả"): "nghị định xử phạt vi phạm hành chính",
    ("đất đai", "mặt bằng", "thuê đất"): "Luật Đất đai hỗ trợ mặt bằng sản xuất",
    ("đấu thầu", "nhà thầu"): "Luật Đấu thầu ưu đãi doanh nghiệp nhỏ và vừa",
    ("tín dụng", "bảo lãnh", "vay vốn"): "quỹ bảo lãnh tín dụng hỗ trợ doanh nghiệp nhỏ và vừa",
}


def _dict_expand(query: str) -> str:
    """Fallback domain expansion dùng dict."""
    lowered = query.lower()
    additions = [
        expansion
        for keywords, expansion in _DOMAIN_EXPANSIONS.items()
        if any(kw in lowered for kw in keywords)
    ]
    return f"{query} {' '.join(additions)}".strip()


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------

_HYDE_SYSTEM = """Bạn là chuyên gia pháp lý Việt Nam. Hãy viết 1 đoạn văn ngắn (~100 từ) 
như thể đây là nội dung của một điều luật hoặc quy định trả lời cho câu hỏi sau. 
Dùng thuật ngữ pháp lý chính xác. Viết bằng tiếng Việt."""


def generate_hypothetical_doc(query: str, generator: object | None) -> str:
    """
    Sinh hypothetical document cho HyDE.

    Args:
        query: câu hỏi gốc
        generator: UnslothGenerator instance (nếu đã load)

    Returns:
        hypothetical doc text để concat với query cho dense retrieval
    """
    if generator is None or not SEARCH.hyde_enabled:
        return _dict_expand(query)

    try:
        # Dùng generator đã có để sinh hypothetical doc
        from vpl.answer.generator import build_messages
        messages = [
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": f"Câu hỏi: {query}"},
        ]
        # Generate 1 câu trả lời ngắn — batch_size=1, max_new_tokens nhỏ
        results = generator.generate_raw(messages, max_new_tokens=SEARCH.hyde_max_tokens)  # type: ignore[attr-defined]
        hypothetical = results[0] if results else ""
        if hypothetical:
            return f"{query} {hypothetical}"
    except Exception:
        pass

    return _dict_expand(query)


def expand(query: str, generator: object | None = None) -> str:
    """
    Expand query cho retrieval.

    Nếu generator có → HyDE.
    Fallback → domain dict expansion.
    """
    return generate_hypothetical_doc(query, generator)
