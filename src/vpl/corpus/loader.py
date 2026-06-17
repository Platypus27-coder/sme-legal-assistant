"""
VietPhapLy RAG — data loader.

Thu thập dữ liệu từ HuggingFace:
  - phapdien-moj-gov-vn (config: articles)
  - anle-toaan-gov-vn

Output:
  - artifacts/raw/legal_docs.jsonl
  - artifacts/raw/precedents.jsonl
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from vpl.corpus.schema import (
    build_chunk_id,
    extract_all_doc_ids,
    extract_articles_from_text,
    extract_doc_id,
    infer_doc_type,
    normalize_article_number,
    normalize_text,
)
from vpl.settings import CORPUS, LEGAL_DOCS_FILE, PRECEDENTS_FILE, RAW_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# HuggingFace loader
# ---------------------------------------------------------------------------

def _load_hf_dataset(
    candidates: tuple[str, ...],
    config_name: str | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        return None, {"status": "datasets_unavailable", "error": str(exc)}

    for name in candidates:
        try:
            ds = load_dataset(name, config_name) if config_name else load_dataset(name)
            return ds, {"status": "ok", "dataset": name}
        except Exception as exc:
            print(f"  ⚠ {name}: {str(exc)[:200]}")
    return None, {"status": "all_failed", "tried": list(candidates)}


def _iter_rows(dataset: Any) -> Iterable[dict[str, Any]]:
    if dataset is None:
        return
    if hasattr(dataset, "keys"):
        for split in dataset.keys():
            for row in dataset[split]:
                yield dict(row)
    else:
        yield from (dict(row) for row in dataset)


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _first(row: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = normalize_text(row.get(k))
        if v:
            return v
    return ""


def _source_links_text(row: dict[str, Any]) -> str:
    parts = []
    for link in row.get("source_links") or []:
        if isinstance(link, dict):
            parts.append(str(link.get("text") or link.get("href") or ""))
        else:
            parts.append(str(link))
    return normalize_text(" ".join(parts))


# ---------------------------------------------------------------------------
# SME scoring
# ---------------------------------------------------------------------------

def _sme_score(text: str) -> float:
    lowered = normalize_text(text).lower()
    score = sum(2.0 for kw in CORPUS.keywords_high if kw.lower() in lowered)
    score += sum(1.0 for kw in CORPUS.keywords_medium if kw.lower() in lowered)
    return score


# ---------------------------------------------------------------------------
# Normalize a single Pháp điển row
# ---------------------------------------------------------------------------

def _normalize_legal_doc(row: dict[str, Any]) -> dict[str, Any]:
    article_title = _first(row, ["article_title", "title", "doc_title", "subject_title"])
    source_note = _first(row, ["source_note_text", "source_note", "citation"])
    link_text = _source_links_text(row)
    if not source_note:
        source_note = link_text

    doc_title = _first(row, ["document_title", "document_name", "law_title", "subject_title"])
    raw_text = _first(row, ["content_text", "markdown", "text", "content", "noi_dung"])
    combined = f"{source_note} {link_text} {doc_title} {article_title} {raw_text}"

    doc_id = extract_doc_id(_first(row, ["doc_id", "law_id", "so_hieu", "document_id"]))
    if not doc_id:
        doc_id = extract_doc_id(combined)

    doc_type = _first(row, ["doc_type", "loai_van_ban", "type"])
    if not doc_type:
        doc_type = infer_doc_type(source_note or combined, CORPUS.doc_type_patterns)

    # Article numbers — prefer explicit field, fallback to source_note parsing
    article_number = normalize_article_number(_first(row, ["article_number", "article_num"]))
    article_numbers = [article_number] if article_number else []
    if not article_numbers:
        source_articles = extract_articles_from_text(source_note)
        doc_ids_in_note = extract_all_doc_ids(source_note)
        article_numbers = source_articles if len(doc_ids_in_note) <= 1 else source_articles[:1]
        article_number = article_numbers[0] if article_numbers else ""
    if not article_number:
        article_number = extract_articles_from_text(article_title)[0] if article_title else ""

    enriched = ". ".join(v for v in (
        article_title,
        raw_text,
        f"Căn cứ nguồn: {source_note}" if source_note else "",
    ) if v)

    return {
        "doc_id": normalize_text(doc_id),
        "doc_type": normalize_text(doc_type),
        "doc_title": normalize_text(doc_title or article_title),
        "article_number": article_number,
        "article_numbers": article_numbers,
        "submission_article": bool(article_numbers),
        "article_title": article_title,
        "source_note": source_note,
        "source": "phapdien",
        "raw_text": enriched,
        "sme_score": _sme_score(combined),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Normalize a single Án lệ row
# ---------------------------------------------------------------------------

def _normalize_precedent(row: dict[str, Any]) -> dict[str, Any]:
    title = _first(row, ["title", "doc_name", "name", "case_name", "doc_code"])
    raw_text = _first(row, ["markdown", "text", "content"])

    # Fallback: extract from structure_json sentences
    if not raw_text and "structure_json" in row:
        try:
            structure = row["structure_json"]
            if isinstance(structure, str):
                structure = json.loads(structure)
            sentences = [s.get("text", "") for s in structure.get("sentences", []) if s.get("text")]
            raw_text = " ".join(sentences)
        except Exception:
            pass

    doc_id = _first(row, ["doc_code", "doc_name", "doc_id", "case_id", "precedent_number"])

    return {
        "doc_id": normalize_text(doc_id),
        "doc_type": "Án lệ",
        "doc_title": title,
        "source": "anle",
        "raw_text": raw_text,
        "applied_article_number": row.get("applied_article_number"),
        "applied_article_code": row.get("applied_article_code"),
        "sme_score": _sme_score(f"{title} {raw_text}"),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main collect function
# ---------------------------------------------------------------------------

def collect() -> dict[str, Any]:
    """Thu thập dữ liệu từ HuggingFace và lưu ra JSONL."""
    _ensure_dirs()
    report: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat()}

    # Pháp điển
    print("\n[1/2] Loading Pháp điển dataset...")
    phapdien_ds, phapdien_status = _load_hf_dataset(CORPUS.phapdien_candidates, "articles")
    legal_docs = [
        doc for doc in (_normalize_legal_doc(row) for row in _iter_rows(phapdien_ds))
        if doc["raw_text"] or doc["doc_title"]
    ]
    legal_count = _write_jsonl(LEGAL_DOCS_FILE, legal_docs)
    print(f"  ✅ {legal_count} legal docs → {LEGAL_DOCS_FILE}")
    report["legal_docs"] = {
        "status": phapdien_status,
        "count": legal_count,
        "missing_doc_id": sum(1 for d in legal_docs if not d["doc_id"]),
    }

    # Án lệ
    print("\n[2/2] Loading Án lệ dataset...")
    anle_ds, anle_status = _load_hf_dataset(CORPUS.anle_candidates)
    precedents = [
        doc for doc in (_normalize_precedent(row) for row in _iter_rows(anle_ds))
        if doc["raw_text"] or doc["doc_title"]
    ]
    precedent_count = _write_jsonl(PRECEDENTS_FILE, precedents)
    print(f"  ✅ {precedent_count} precedents → {PRECEDENTS_FILE}")
    report["precedents"] = {
        "status": anle_status,
        "count": precedent_count,
    }

    return report
