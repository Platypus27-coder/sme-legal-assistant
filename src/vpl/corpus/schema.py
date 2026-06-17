"""
VietPhapLy RAG — chunk schema và metadata helpers.

Mỗi LegalChunk phải mang đủ thông tin để tạo submission trực tiếp,
không cần lookup ngược lại raw data.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_DOC_ID_RE = re.compile(
    r"\b(?:\d{1,4}/\d{4}/QH\d+|\d{1,4}/(?:\d{4}/)?(?:NĐ-CP|ND-CP|[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)+))\b",
    flags=re.IGNORECASE,
)

_ARTICLE_RE = re.compile(r"(?:Điều|điều)\s*(\d+[a-zA-Z]?)\b")


def normalize_text(value: Any) -> str:
    """Chuẩn hóa unicode, collapse whitespace."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def extract_doc_id(text: str) -> str:
    """Trích mã văn bản đầu tiên từ chuỗi (e.g. '04/2017/QH14')."""
    text = normalize_text(text)
    m = _DOC_ID_RE.search(text)
    return m.group(0).upper() if m else ""


def extract_all_doc_ids(text: str) -> list[str]:
    text = normalize_text(text)
    return list(dict.fromkeys(m.group(0).upper() for m in _DOC_ID_RE.finditer(text)))


def is_valid_doc_id(value: Any) -> bool:
    return bool(extract_doc_id(str(value or "")))


def normalize_article_number(value: Any) -> str:
    """Chuẩn hóa về dạng 'Điều X'."""
    text = normalize_text(value)
    if not text:
        return ""
    m = _ARTICLE_RE.search(text)
    if m:
        return f"Điều {m.group(1)}"
    m2 = re.match(r"^(\d+[a-zA-Z]?)$", text)
    if m2:
        return f"Điều {m2.group(1)}"
    return text


def extract_articles_from_text(text: str) -> list[str]:
    text = normalize_text(text)
    return list(dict.fromkeys(
        f"Điều {m.group(1)}"
        for m in re.finditer(r"\bĐiều\s+(\d+[a-zA-Z]?)\b", text, re.IGNORECASE)
    ))


def infer_doc_type(text: str, patterns: tuple) -> str:
    lowered = normalize_text(text).lower()
    for doc_type, keys in patterns:
        if any(k in lowered for k in keys):
            return doc_type
    return "Văn bản"


def _safe_slug(value: str) -> str:
    value = normalize_text(value)
    value = value.replace("/", "_").replace("|", "_")
    value = re.sub(r"[^0-9A-Za-zÀ-ỹ_\-.]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "unknown"


def build_formatted_doc(doc_id: str, doc_type: str, doc_title: str) -> str:
    """
    Tạo chuỗi formatted_doc theo chuẩn BTC:
    '<mã VB>|<Loại VB> <Mã VB> <Trích yếu>'
    """
    doc_id = normalize_text(doc_id)
    doc_type = normalize_text(doc_type)
    title = normalize_text(doc_title)

    # Loại bỏ prefix thừa khỏi title
    if doc_type:
        title = re.sub(rf"^{re.escape(doc_type)}\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^số\s+", "", title, flags=re.IGNORECASE)
    if doc_id:
        title = re.sub(re.escape(doc_id), "", title, count=1, flags=re.IGNORECASE)
    title = title.strip(" ,;:-")

    label = " ".join(p for p in (doc_type, doc_id, title) if p)
    return f"{doc_id}|{label}"


def build_formatted_article(doc_id: str, doc_type: str, doc_title: str, article_number: str) -> str:
    """
    Tạo chuỗi formatted_article theo chuẩn BTC:
    '<mã VB>|<Loại VB> <Mã VB> <Trích yếu>|<Điều X>'
    """
    article = normalize_article_number(article_number)
    return f"{build_formatted_doc(doc_id, doc_type, doc_title)}|{article}"


def build_chunk_id(doc_id: str, article_number: str, paragraph: str = "", suffix: str = "") -> str:
    parts = [_safe_slug(doc_id)]
    article = normalize_article_number(article_number)
    if article:
        parts.append(_safe_slug(article.replace("Điều", "Dieu")))
    if paragraph:
        parts.append(_safe_slug(f"Khoan_{paragraph}"))
    if suffix:
        parts.append(_safe_slug(suffix))
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkMeta:
    doc_id: str
    doc_type: str
    doc_title: str
    article_number: str
    formatted_doc: str          # "04/2017/QH14|Luật 04/2017/QH14 ..."
    formatted_article: str      # "04/2017/QH14|...|Điều 4"
    source: str = ""
    paragraph_number: str = ""
    sme_score: float = 0.0
    source_note: str = ""
    submission_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    text: str
    meta: ChunkMeta

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.meta.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LegalChunk":
        meta_data = data.get("metadata") or data.get("meta") or {}
        meta = ChunkMeta(**{
            k: meta_data.get(k, v)
            for k, v in ChunkMeta.__dataclass_fields__.items()  # type: ignore[attr-defined]
        })
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            meta=meta,
        )
