"""Fast submission retune without regenerating answers.

Reads an existing results.json plus a retrieval SQLite cache, then writes three
candidate submissions with cleaner relevant_articles selections.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


LEGAL_BASIS_RE = re.compile(
    r"\n\nC(?:ơ|Æ¡)\s+s(?:ở|á»Ÿ)\s+ph(?:á|Ă¡)p\s+l(?:ý|Ă½)\s+tham\s+chi(?:ếu|áº¿u):.*?"
    r"(?=\n\nC(?:ả|áº£)nh\s+b(?:á|Ă¡)o\s+gi(?:ới|á»›i)\s+h(?:ạn|áº¡n):|\Z)",
    re.IGNORECASE | re.DOTALL,
)

WARNING_RE = re.compile(
    r"\n\nC(?:ả|áº£)nh\s+b(?:á|Ă¡)o\s+gi(?:ới|á»›i)\s+h(?:ạn|áº¡n):.*?\Z",
    re.IGNORECASE | re.DOTALL,
)

ARTICLE_RE = re.compile(r"(?:Điều|Äiá»u|Ä‘iá»u)\s+(\d+[A-Za-z]?)", re.IGNORECASE)
DOC_ID_RE = re.compile(r"\b\d{1,4}/(?:\d{4}/)?[\wĐđÄÄ‘-]+\b", re.IGNORECASE)


@dataclass(frozen=True)
class Variant:
    name: str
    high: float
    safe: float
    min_articles: int
    max_articles: int


VARIANTS = (
    Variant("balanced_v1", high=0.56, safe=0.46, min_articles=1, max_articles=4),
    Variant("precision_v1", high=0.62, safe=0.52, min_articles=0, max_articles=3),
    Variant("recall_v1", high=0.50, safe=0.40, min_articles=1, max_articles=5),
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array or a JSONL stream."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON array")
        return rows

    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc
    return rows


def _load_cache(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing retrieval cache: {path}. Copy retrieval.db from Drive or pass --cache."
        )
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute("SELECT question_id, chunks_json FROM retrieval_cache").fetchall()
    finally:
        conn.close()
    return {int(qid): json.loads(chunks_json) for qid, chunks_json in rows}


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = chunk.get("metadata") or chunk.get("meta") or {}
    return meta if isinstance(meta, dict) else {}


def _score(chunk: dict[str, Any]) -> float:
    try:
        return float(chunk.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def _article_label(formatted_article: str) -> str:
    return formatted_article.rsplit("|", 1)[-1] if "|" in formatted_article else formatted_article


def _doc_label(formatted_doc: str) -> str:
    return formatted_doc.split("|", 1)[-1] if "|" in formatted_doc else formatted_doc


def _extract_main_answer(answer: str) -> str:
    answer = LEGAL_BASIS_RE.sub("", answer)
    answer = WARNING_RE.sub("", answer)
    return answer


def _lexical_boost(question: str, main_answer: str, meta: dict[str, Any]) -> float:
    text = f"{question}\n{main_answer}".lower()
    boost = 0.0

    article = str(meta.get("article_number") or "")
    if article and article.lower() in text:
        boost += 0.08

    doc_id = str(meta.get("doc_id") or "")
    if doc_id and doc_id.lower() in text:
        boost += 0.05

    formatted_doc = str(meta.get("formatted_doc") or "")
    doc_name = _doc_label(formatted_doc).lower()
    if doc_name and len(doc_name) > 8 and doc_name in text:
        boost += 0.04

    for token in DOC_ID_RE.findall(question):
        if doc_id and token.upper() == doc_id.upper():
            boost += 0.04

    mentioned_articles = {f"Điều {m.upper()}" for m in ARTICLE_RE.findall(question)}
    if article in mentioned_articles:
        boost += 0.05

    return boost


def _candidate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best raw cache chunk per formatted_article."""
    best: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        meta = _metadata(chunk)
        art = str(meta.get("formatted_article") or "")
        doc_id = str(meta.get("doc_id") or "")
        if not art or not doc_id or meta.get("submission_eligible", True) is False:
            continue
        if _score(chunk) > _score(best.get(art, {})):
            best[art] = chunk
    return list(best.values())


def _load_reranker(enabled: bool, model_name: str, device: str):
    if not enabled:
        return None
    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(model_name, max_length=512, device=device)
        print(f"Loaded reranker: {model_name} on {device}")
        return reranker
    except Exception as exc:
        print(f"Reranker unavailable, using cached scores only: {exc}")
        return None


def _rerank_question(
    reranker: Any,
    question: str,
    answer: str,
    candidates: list[dict[str, Any]],
    batch_size: int,
) -> list[tuple[float, dict[str, Any]]]:
    if not candidates:
        return []

    if reranker is not None:
        try:
            pairs = [(question, str(c.get("text", ""))) for c in candidates]
            logits = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            base_scores = [_sigmoid(float(logit)) for logit in logits]
        except Exception as exc:
            print(f"  rerank failed for one question, fallback to cache score: {exc}")
            base_scores = [_score(c) for c in candidates]
    else:
        base_scores = [_score(c) for c in candidates]

    main_answer = _extract_main_answer(answer)
    scored: list[tuple[float, dict[str, Any]]] = []
    for base, chunk in zip(base_scores, candidates):
        meta = _metadata(chunk)
        score = min(1.0, float(base) + _lexical_boost(question, main_answer, meta))
        scored.append((score, meta))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _select(scored: list[tuple[float, dict[str, Any]]], variant: Variant) -> list[tuple[float, dict[str, Any]]]:
    high = [(s, m) for s, m in scored if s >= variant.high]
    safe = [(s, m) for s, m in scored if s >= variant.safe]
    if high:
        selected = high
    elif safe:
        selected = safe
    else:
        selected = scored[: variant.min_articles]
    return selected[: variant.max_articles]


def _basis_line(selected: list[tuple[float, dict[str, Any]]]) -> str:
    refs = []
    for _, meta in selected:
        article = str(meta.get("article_number") or _article_label(str(meta.get("formatted_article") or "")))
        doc = _doc_label(str(meta.get("formatted_doc") or ""))
        if article and doc:
            refs.append(f"{article} của {doc}")
    refs = _dedupe(refs)
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


def _make_result(
    row: dict[str, Any],
    selected: list[tuple[float, dict[str, Any]]],
) -> dict[str, Any]:
    rel_docs = _dedupe(str(meta.get("formatted_doc") or "") for _, meta in selected)
    rel_articles = _dedupe(str(meta.get("formatted_article") or "") for _, meta in selected)
    return {
        "id": int(row["id"]),
        "question": str(row["question"]),
        "answer": _rewrite_answer(str(row.get("answer") or ""), selected),
        "relevant_docs": rel_docs,
        "relevant_articles": rel_articles,
    }


def _validate(results: list[dict[str, Any]], questions: list[dict[str, Any]]) -> list[str]:
    from vpl.submit import validate

    return validate(results, questions)


def _basic_validate(results: list[dict[str, Any]], questions: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    q_map = {int(q["id"]): str(q["question"]) for q in questions}
    seen: set[int] = set()
    for i, row in enumerate(results):
        prefix = f"[{i}]"
        missing = {"id", "question", "answer", "relevant_docs", "relevant_articles"} - set(row)
        if missing:
            errors.append(f"{prefix} missing fields: {sorted(missing)}")
            continue
        qid = int(row["id"])
        if qid in seen:
            errors.append(f"{prefix} duplicate id {qid}")
        seen.add(qid)
        if q_map.get(qid) != row["question"]:
            errors.append(f"{prefix} question mismatch for id {qid}")
        if not str(row.get("answer") or "").strip():
            errors.append(f"{prefix} empty answer")
    return errors


def _write_variant(
    out_dir: Path,
    variant: Variant,
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    strict_validation: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"results_{variant.name}.json"
    zip_path = out_dir / f"submission_{variant.name}.zip"

    errors = _validate(results, questions) if strict_validation else _basic_validate(results, questions)
    if errors:
        preview = "\n".join(f"  - {e}" for e in errors[:20])
        raise ValueError(f"{variant.name} validation failed ({len(errors)} errors):\n{preview}")

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="results.json")

    # Copy to Google Drive if available
    drive_dir = Path("/content/drive/MyDrive/R2AI_Artifacts")
    if drive_dir.exists():
        try:
            drive_zip = drive_dir / f"submission_{variant.name}.zip"
            import shutil
            shutil.copy2(zip_path, drive_zip)
            print(f"  ☁️ Copped to Drive: {drive_zip}")
        except Exception as e:
            print(f"  ⚠️ Drive copy failed: {e}")

    counts = [len(r["relevant_articles"]) for r in results]
    doc_counter = Counter(d for r in results for d in r["relevant_docs"])
    article_counter = Counter(a for r in results for a in r["relevant_articles"])
    zip_members = zipfile.ZipFile(zip_path).namelist()

    print(f"\n{variant.name}")
    print(f"  json: {results_path}")
    print(f"  zip:  {zip_path}")
    print(f"  zip_members: {zip_members}")
    print(
        "  articles/question: "
        f"min={min(counts)} max={max(counts)} mean={statistics.mean(counts):.2f} "
        f"dist={dict(sorted(Counter(counts).items()))}"
    )
    print(f"  zero_articles: {sum(1 for c in counts if c == 0)}")
    print(f"  top_docs: {doc_counter.most_common(5)}")
    print(f"  top_articles: {article_counter.most_common(5)}")


def build_variants(
    answers: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    cache: dict[int, list[dict[str, Any]]],
    reranker: Any,
    batch_size: int,
    out_dir: Path,
    limit: int | None = None,
) -> None:
    rows = answers[:limit] if limit else answers
    per_question: dict[int, list[tuple[float, dict[str, Any]]]] = {}

    for idx, row in enumerate(rows, 1):
        qid = int(row["id"])
        chunks = cache.get(qid, [])
        candidates = _candidate_chunks(chunks)
        per_question[qid] = _rerank_question(
            reranker=reranker,
            question=str(row["question"]),
            answer=str(row.get("answer") or ""),
            candidates=candidates,
            batch_size=batch_size,
        )
        if idx % 100 == 0 or idx == len(rows):
            print(f"scored {idx}/{len(rows)} questions", flush=True)

    question_rows = questions[:limit] if limit else questions
    for variant in VARIANTS:
        results = [
            _make_result(row, _select(per_question[int(row["id"])], variant))
            for row in rows
        ]
        _write_variant(out_dir, variant, results, question_rows, strict_validation=limit is None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=ROOT / "artifacts" / "output" / "results.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "cache" / "retrieval.db")
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "R2AIStage1DATA.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "output" / "dev_huy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only: process first N rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answers = _load_rows(args.answers)
    questions = _load_json(args.questions)
    cache = _load_cache(args.cache)
    if args.limit is None and len(answers) != len(questions):
        raise ValueError(f"answers/questions length mismatch: {len(answers)} != {len(questions)}")
    if args.limit is not None and len(answers) < args.limit:
        raise ValueError(f"--limit {args.limit} requested but answers only has {len(answers)} rows")

    print(f"answers:   {args.answers} ({len(answers)} rows)")
    print(f"questions: {args.questions} ({len(questions)} rows)")
    print(f"cache:     {args.cache} ({len(cache)} rows)")
    print(f"out_dir:   {args.out_dir}")
    reranker = _load_reranker(args.use_reranker, args.reranker_model, args.device)
    build_variants(
        answers=answers,
        questions=questions,
        cache=cache,
        reranker=reranker,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
