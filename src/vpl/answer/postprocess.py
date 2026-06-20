"""
VietPhapLy RAG — 3-tier post-processing.

Tầng 1: Regex trích xuất Điều X từ answer
Tầng 2: Đối chiếu với retrieved chunks → loại hallucination
Tầng 3: Append citation fallback nếu LLM không cite

Logic được refactored sạch sẽ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from vpl.corpus.schema import is_valid_doc_id
from vpl.settings import SEARCH, SUBMISSION

_ARTICLE_RE = re.compile(r"(?:Điều|Điểu|điều|điểu)\s+(\d+[A-Za-z]?)", re.IGNORECASE)


def extract_cited_articles(text: str) -> list[str]:
    seen: set[str] = set()
    result = []
    for num in _ARTICLE_RE.findall(text or ""):
        article = f"Điều {num.upper()}"
        if article not in seen:
            seen.add(article)
            result.append(article)
    return result


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


@dataclass(frozen=True)
class PostConfig:
    safe_threshold: float = SEARCH.safe_threshold
    high_conf_threshold: float = SEARCH.high_conf_threshold
    min_articles: int = SEARCH.min_articles
    max_articles: int = SEARCH.max_articles
    max_context_chunks: int = SEARCH.max_context_chunks
    max_fallback_citations: int = 15
    standard_warning: str = SUBMISSION.standard_warning


class PostProcessor:

    def __init__(self, config: PostConfig | None = None):
        self.cfg = config or PostConfig()

    def select_relevant_chunks(self, chunks: Sequence[Any]) -> list[Any]:
        """Chọn chunks eligible để đưa vào relevant_articles."""
        ranked = sorted(chunks, key=lambda c: c.score if hasattr(c, "score") else c.get("score", 0), reverse=True)

        def _meta(c):
            return c.metadata if hasattr(c, "metadata") else (c.get("metadata") or {})

        eligible = [
            c for c in ranked
            if _meta(c).get("formatted_article")
            and is_valid_doc_id(_meta(c).get("doc_id"))
            and _meta(c).get("submission_eligible", True)
        ]

        # Dedup by formatted_article
        seen: set[str] = set()
        unique: list[Any] = []
        for c in eligible:
            article = str(_meta(c).get("formatted_article"))
            if article not in seen:
                seen.add(article)
                unique.append(c)
        eligible = unique

        # Dynamic threshold
        selected = [c for c in eligible if (c.score if hasattr(c, "score") else c.get("score", 0)) >= self.cfg.high_conf_threshold]
        if len(selected) < self.cfg.min_articles:
            selected = [c for c in eligible if (c.score if hasattr(c, "score") else c.get("score", 0)) >= self.cfg.safe_threshold]
        if len(selected) < self.cfg.min_articles:
            selected = eligible[:self.cfg.min_articles]
        return selected[:self.cfg.max_articles]

    def process_answer(self, answer: str, chunks: Sequence[Any]) -> tuple[str, list[str]]:
        """
        Làm sạch answer:
        - Loại Điều hallucinated
        - Append citation fallback
        - Đảm bảo có cảnh báo cuối
        """
        def _meta(c):
            return c.metadata if hasattr(c, "metadata") else (c.get("metadata") or {})

        cited = extract_cited_articles(answer)
        available = {
            str(_meta(c).get("article_number")): str(_meta(c).get("formatted_article"))
            for c in chunks
            if _meta(c).get("article_number") and _meta(c).get("formatted_article")
        }
        hallucinated = [a for a in cited if a not in available]

        # Tầng 2: loại bỏ citations không có trong retrieved chunks
        for article in hallucinated:
            num = re.escape(article.split()[-1])
            answer = re.sub(
                rf"\b(?:Điều|Điểu|điều|điểu)\s+{num}\b",
                "quy định liên quan",
                answer,
                flags=re.IGNORECASE,
            )

        # Tầng 3: citation fallback
        top_score = max(
            (c.score if hasattr(c, "score") else c.get("score", 0) for c in chunks),
            default=0.0,
        )
        if top_score >= 0.0:
            refs = []
            for c in chunks:
                meta = _meta(c)
                article = str(meta.get("article_number") or "")
                fmt_doc = str(meta.get("formatted_doc") or "")
                if article and fmt_doc:
                    refs.append(f"{article} của {fmt_doc.split('|', 1)[-1]}")
            refs = _dedupe(refs)[:self.cfg.max_fallback_citations]
            if refs:
                answer = answer.rstrip() + "\n\nCơ sở pháp lý tham chiếu: " + "; ".join(refs) + "."

        # Đảm bảo có cảnh báo cuối
        if self.cfg.standard_warning.lower() not in answer.lower():
            answer = answer.rstrip() + "\n\n" + self.cfg.standard_warning

        return answer, hallucinated

    def build_result(
        self,
        question_id: int,
        question: str,
        answer: str,
        chunks: Sequence[Any],
    ) -> dict[str, Any]:
        selected = self.select_relevant_chunks(chunks)
        if not answer.strip():
            answer = "Hệ thống dữ liệu chưa ghi nhận quy định pháp lý cụ thể cho tình huống này."
        answer, _ = self.process_answer(answer, selected)

        def _meta(c):
            return c.metadata if hasattr(c, "metadata") else (c.get("metadata") or {})

        return {
            "id": int(question_id),
            "question": question,
            "answer": answer,
            "relevant_docs": _dedupe([str(_meta(c).get("formatted_doc") or "") for c in selected]),
            "relevant_articles": _dedupe([str(_meta(c).get("formatted_article") or "") for c in selected]),
        }
