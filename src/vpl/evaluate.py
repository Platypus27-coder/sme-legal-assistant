"""
VietPhapLy RAG — local evaluation metrics.

- macro_f2: tính F2 score trên dev-set có nhãn
- silver_recall: tính recall trên câu hỏi có mention "Điều X"
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from vpl.corpus.schema import extract_doc_id


def macro_f2(
    predictions: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    Tính macro F2 score.
    F2 = 5PR / (4P + R) — Recall quan trọng gấp 4 lần Precision.
    """
    ref_map = {int(r["id"]): set(str(a) for a in r.get("relevant_articles", [])) for r in references}
    p_vals, r_vals, f2_vals = [], [], []
    for pred in predictions:
        expected = ref_map.get(int(pred["id"]), set())
        actual = set(str(a) for a in pred.get("relevant_articles", []))
        correct = len(actual & expected)
        p = correct / len(actual) if actual else 0.0
        r = correct / len(expected) if expected else 0.0
        f2 = (5 * p * r / (4 * p + r)) if (p or r) else 0.0
        p_vals.append(p)
        r_vals.append(r)
        f2_vals.append(f2)
    n = len(p_vals) or 1
    return {
        "macro_precision": sum(p_vals) / n,
        "macro_recall": sum(r_vals) / n,
        "macro_f2": sum(f2_vals) / n,
        "n_evaluated": len(p_vals),
    }


def silver_recall(
    questions: Sequence[dict[str, Any]],
    cached_rows: Sequence[dict[str, Any]],
    cutoffs: Sequence[int] = (3, 5, 10),
) -> dict[str, float]:
    """
    Silver recall: dùng câu hỏi có mention "Điều X" làm ground truth đơn giản.
    """
    cache_map = {int(r["id"]): r for r in cached_rows}
    totals = {k: 0 for k in cutoffs}
    n = 0
    for q in questions:
        m = re.search(r"\bĐiều\s+(\d+[A-Za-z]?)\b", q["question"], re.IGNORECASE)
        qid = int(q["id"])
        if not m or qid not in cache_map:
            continue
        expected_article = f"Điều {m.group(1)}".lower()
        expected_doc = extract_doc_id(q["question"])
        chunks = cache_map[qid].get("chunks", [])
        n += 1
        for cutoff in cutoffs:
            for c in chunks[:cutoff]:
                meta = c.get("metadata") or {}
                art_match = str(meta.get("article_number") or "").lower() == expected_article
                doc_match = not expected_doc or str(meta.get("doc_id") or "") == expected_doc
                if art_match and doc_match:
                    totals[cutoff] += 1
                    break
    denom = n or 1
    return {f"silver_recall@{k}": totals[k] / denom for k in cutoffs} | {"n_evaluated": n}
