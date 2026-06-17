"""
VietPhapLy RAG — submission validation + ZIP packaging.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Sequence

from vpl.corpus.schema import is_valid_doc_id
from vpl.answer.postprocess import extract_cited_articles
from vpl.settings import RESULTS_FILE, RESULTS_STREAM_FILE, SUBMISSION, SUBMISSION_FILE

_ARTICLE_FORMAT = re.compile(r"^[^|]+\|[^|]+\|Điều \d+[A-Za-z]?$")


def load_jsonl_stream(path: Path) -> list[dict[str, Any]]:
    """Đọc results_partial.jsonl → list sorted by id."""
    rows: dict[int, dict] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows[int(row["id"])] = row
            except Exception:
                pass
    return sorted(rows.values(), key=lambda r: int(r["id"]))


class ResultStore:
    """Crash-safe JSONL streaming output."""

    def __init__(self, path: Path = RESULTS_STREAM_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[int] = self._scan()

    def _scan(self) -> set[int]:
        done: set[int] = set()
        if not self.path.exists():
            return done
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(int(json.loads(line)["id"]))
                except Exception:
                    pass
        return done

    @property
    def completed_ids(self) -> set[int]:
        return set(self._done)

    def append(self, result: dict[str, Any]) -> bool:
        rid = int(result["id"])
        if rid in self._done:
            return False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb+") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.seek(0, os.SEEK_END)
                    fh.write(b"\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._done.add(rid)
        return True

    def export(self, path: Path = RESULTS_FILE) -> Path:
        rows = sorted(self.read_all(), key=lambda r: int(r["id"]))
        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        return path

    def read_all(self) -> list[dict[str, Any]]:
        return load_jsonl_stream(self.path)


def validate(
    results: Sequence[dict[str, Any]],
    questions: Sequence[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(results, list):
        return ["results must be a JSON array"]
    q_map = {int(q["id"]): str(q["question"]) for q in questions}
    seen: set[int] = set()

    if len(results) != SUBMISSION.expected_count:
        errors.append(f"expected {SUBMISSION.expected_count} rows, got {len(results)}")

    for i, row in enumerate(results):
        pfx = f"[{i}]"
        missing = SUBMISSION.required_fields - set(row)
        if missing:
            errors.append(f"{pfx} missing fields: {sorted(missing)}")
            continue
        try:
            rid = int(row["id"])
        except (TypeError, ValueError):
            errors.append(f"{pfx} id must be integer")
            continue
        if rid in seen:
            errors.append(f"{pfx} duplicate id {rid}")
        seen.add(rid)
        if rid not in q_map:
            errors.append(f"{pfx} unknown id {rid}")
        elif row["question"] != q_map[rid]:
            errors.append(f"{pfx} question mismatch for id {rid}")
        if not isinstance(row["answer"], str) or not row["answer"].strip():
            errors.append(f"{pfx} empty answer")
        for doc in row.get("relevant_docs", []):
            parts = str(doc).split("|")
            if len(parts) != 2 or not is_valid_doc_id(parts[0]) or not parts[1].strip():
                errors.append(f"{pfx} invalid relevant_doc: {doc!r}")
        for art in row.get("relevant_articles", []):
            if not _ARTICLE_FORMAT.match(str(art)) or not is_valid_doc_id(str(art).split("|")[0]):
                errors.append(f"{pfx} invalid relevant_article: {art!r}")
        # Cross-check: articles cited in answer must include relevant_articles
        cited = {a.upper() for a in extract_cited_articles(row["answer"])}
        missing_in_answer = [
            str(a).split("|")[-1]
            for a in row.get("relevant_articles", [])
            if str(a).split("|")[-1].upper() not in cited
        ]
        if missing_in_answer:
            errors.append(f"{pfx} relevant_articles not cited in answer: {missing_in_answer}")

    missing_ids = sorted(set(q_map) - seen)
    if missing_ids:
        errors.append(f"missing ids: {missing_ids[:20]}")
    return errors


def package(
    results_path: Path = RESULTS_FILE,
    questions_path: Path | None = None,
    zip_path: Path = SUBMISSION_FILE,
) -> Path:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8")) if questions_path else []
    if questions:
        errors = validate(results, questions)
        if errors:
            preview = "\n".join(f"  - {e}" for e in errors[:50])
            raise ValueError(f"Validation failed ({len(errors)} errors):\n{preview}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="results.json")
    print(f"✅ Submission → {zip_path}")
    return zip_path
