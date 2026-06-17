"""
VietPhapLy RAG — structural chunker.

Quy tắc chunking:
  - Mỗi Điều = 1 chunk
  - Nếu Điều > MAX_ARTICLE_CHARS → tách theo Khoản
  - Án lệ → 1 chunk per vụ án
  - Dedup chunk_id trùng → suffix _dup_N
  - Canonicalize doc_title per doc_id (lấy title phổ biến nhất)

Output: artifacts/raw/chunks.jsonl + chunk_stats.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from vpl.corpus.schema import (
    ChunkMeta,
    LegalChunk,
    build_chunk_id,
    build_formatted_article,
    build_formatted_doc,
    is_valid_doc_id,
    normalize_article_number,
    normalize_text,
)
from vpl.settings import (
    CHUNKS_FILE,
    CHUNK_STATS_FILE,
    CORPUS,
    LEGAL_DOCS_FILE,
    PRECEDENTS_FILE,
    RAW_DIR,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ARTICLE_RE = re.compile(
    r"(?P<header>(?:^|\n)\s*Điều\s+(?P<num>\d+[a-zA-Z]?)\.?[^\n]*)",
    flags=re.IGNORECASE,
)
_PARAGRAPH_RE = re.compile(
    r"(?m)^\s*(?P<num>\d+)\.\s+(?P<body>.*?)(?=^\s*\d+\.\s+|\Z)",
    flags=re.DOTALL,
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"_error": str(exc), "_line": i})
    return rows


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def _split_by_article(text: str) -> list[tuple[str, str]]:
    """Tách text thành danh sách (article_number, article_text)."""
    text = normalize_text(text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"\s+(Điều\s+\d+[a-zA-Z]?\.)", r"\n\1", text, flags=re.IGNORECASE)
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return []
    result = []
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        article_num = normalize_article_number(m.group("num"))
        article_text = normalize_text(text[m.start():end])
        if len(article_text) >= CORPUS.min_chunk_chars:
            result.append((article_num, article_text))
    return result


def _split_long_article(text: str) -> list[tuple[str, str]]:
    """Tách Điều dài thành các Khoản."""
    if len(text) <= CORPUS.max_article_chars:
        return [("", text)]
    paragraphs = []
    for m in _PARAGRAPH_RE.finditer(text):
        body = normalize_text(m.group(0))
        if len(body) >= CORPUS.min_chunk_chars:
            paragraphs.append((m.group("num"), body))
    return paragraphs or [("", text)]


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def _make_chunk(
    doc: dict[str, Any],
    article_number: str,
    text: str,
    paragraph: str = "",
    suffix: str = "",
) -> LegalChunk:
    doc_id = normalize_text(doc.get("doc_id")) or "UNKNOWN"
    doc_type = normalize_text(doc.get("doc_type")) or "Văn bản"
    doc_title = normalize_text(doc.get("doc_title")) or doc_id
    article_number = normalize_article_number(article_number)
    chunk_id = build_chunk_id(doc_id, article_number, paragraph, suffix)
    fmt_doc = build_formatted_doc(doc_id, doc_type, doc_title)
    fmt_article = build_formatted_article(doc_id, doc_type, doc_title, article_number) if article_number else ""
    meta = ChunkMeta(
        doc_id=doc_id,
        doc_type=doc_type,
        doc_title=doc_title,
        article_number=article_number,
        formatted_doc=fmt_doc,
        formatted_article=fmt_article,
        source=normalize_text(doc.get("source")),
        paragraph_number=paragraph,
        sme_score=float(doc.get("sme_score") or 0.0),
        source_note=normalize_text(doc.get("source_note")),
        submission_eligible=bool(
            article_number
            and is_valid_doc_id(doc_id)
            and doc.get("submission_article", True)
        ),
    )
    return LegalChunk(chunk_id=chunk_id, text=normalize_text(text), meta=meta)


def _chunk_legal_doc(doc: dict[str, Any]) -> list[LegalChunk]:
    raw = normalize_text(doc.get("raw_text"))
    if not raw:
        return []

    # Có article_numbers từ source_note → dùng luôn
    source_articles = [
        normalize_article_number(a)
        for a in (doc.get("article_numbers") or [doc.get("article_number")])
        if normalize_article_number(a)
    ]
    if source_articles:
        return [
            _make_chunk(doc, art, chunk_text, para)
            for art in source_articles
            for para, chunk_text in _split_long_article(raw)
        ]

    # Parse Điều từ text
    articles = _split_by_article(raw)
    if not articles:
        return [_make_chunk(doc, "", raw, suffix="full_doc")]

    chunks = []
    for art_num, art_text in articles:
        for para, chunk_text in _split_long_article(art_text):
            chunks.append(_make_chunk(doc, art_num, chunk_text, para))
    return chunks


def _chunk_precedent(doc: dict[str, Any]) -> list[LegalChunk]:
    raw = normalize_text(doc.get("raw_text"))
    if not raw:
        return []
    return [_make_chunk(doc, "", raw, suffix="anle")]


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _canonicalize_titles(chunks: list[LegalChunk]) -> list[LegalChunk]:
    """Dùng 1 title ổn định nhất cho mỗi doc_id."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for c in chunks:
        if c.meta.submission_eligible:
            counts[c.meta.doc_id][c.meta.doc_title] += 1

    canonical: dict[str, str] = {
        doc_id: sorted(ctr, key=lambda t: (-ctr[t], len(t), t))[0]
        for doc_id, ctr in counts.items()
    }

    result = []
    for c in chunks:
        title = canonical.get(c.meta.doc_id)
        if not title or title == c.meta.doc_title:
            result.append(c)
            continue
        fmt_doc = build_formatted_doc(c.meta.doc_id, c.meta.doc_type, title)
        new_meta = replace(
            c.meta,
            doc_title=title,
            formatted_doc=fmt_doc,
            formatted_article=f"{fmt_doc}|{c.meta.article_number}" if c.meta.article_number else "",
        )
        result.append(LegalChunk(c.chunk_id, c.text, new_meta))
    return result


def _dedup_chunk_ids(chunks: list[LegalChunk]) -> list[LegalChunk]:
    counts = Counter(c.chunk_id for c in chunks)
    seen: dict[str, int] = defaultdict(int)
    result = []
    for c in chunks:
        if counts[c.chunk_id] > 1:
            seen[c.chunk_id] += 1
            c = LegalChunk(f"{c.chunk_id}_dup_{seen[c.chunk_id]}", c.text, c.meta)
        result.append(c)
    return result


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def chunk() -> dict[str, Any]:
    """Đọc raw JSONL, chunking, ghi chunks.jsonl."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    legal_docs = _read_jsonl(LEGAL_DOCS_FILE)
    precedents = _read_jsonl(PRECEDENTS_FILE)
    print(f"Loaded {len(legal_docs)} legal docs, {len(precedents)} precedents")

    all_chunks: list[LegalChunk] = []
    errors: list[dict] = []

    for doc in legal_docs:
        if "_error" in doc:
            errors.append(doc)
            continue
        all_chunks.extend(_chunk_legal_doc(doc))

    for doc in precedents:
        if "_error" in doc:
            errors.append(doc)
            continue
        all_chunks.extend(_chunk_precedent(doc))

    all_chunks = _canonicalize_titles(all_chunks)
    all_chunks = _dedup_chunk_ids(all_chunks)

    # Write chunks
    with CHUNKS_FILE.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    eligible = [c for c in all_chunks if c.meta.submission_eligible]
    stats = {
        "total_legal_docs": len(legal_docs),
        "total_precedents": len(precedents),
        "total_chunks": len(all_chunks),
        "submission_eligible": len(eligible),
        "unique_docs": len({c.meta.formatted_doc for c in eligible}),
        "unique_articles": len({c.meta.formatted_article for c in eligible}),
        "error_count": len(errors),
        "output_file": str(CHUNKS_FILE),
    }
    CHUNK_STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {stats['total_chunks']} chunks → {CHUNKS_FILE}")
    return stats
