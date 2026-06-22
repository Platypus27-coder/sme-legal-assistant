"""
VietPhapLy RAG — tất cả config tập trung tại đây.

Không có config.py rải rác theo từng module.
Mọi hằng số đều import từ file này.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DIR = ARTIFACTS_DIR / "raw"
INDEX_DIR = ARTIFACTS_DIR / "index"
CACHE_DIR = ARTIFACTS_DIR / "cache"
OUTPUT_DIR = ARTIFACTS_DIR / "output"

# Raw data files
LEGAL_DOCS_FILE = RAW_DIR / "legal_docs.jsonl"
PRECEDENTS_FILE = RAW_DIR / "precedents.jsonl"
CHUNKS_FILE = RAW_DIR / "chunks.jsonl"
CHUNK_STATS_FILE = RAW_DIR / "chunk_stats.json"

# Index files
BM25_DIR = INDEX_DIR / "bm25"
BM25_CORPUS_FILE = BM25_DIR / "corpus.pkl"
BM25_ID_MAP_FILE = BM25_DIR / "id_map.json"
CHROMA_DIR = INDEX_DIR / "chroma"

# Cache
RETRIEVAL_DB_FILE = CACHE_DIR / "retrieval.db"

# Output
RESULTS_STREAM_FILE = OUTPUT_DIR / "results_partial.jsonl"
RESULTS_FILE = OUTPUT_DIR / "results.json"
SUBMISSION_FILE = OUTPUT_DIR / "submission.zip"


# ---------------------------------------------------------------------------
# Corpus config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorpusConfig:
    # HuggingFace dataset candidates (thử theo thứ tự)
    phapdien_candidates: tuple[str, ...] = (
        "tmquan/phapdien-moj-gov-vn",
        "Vietnamese-Legal/phapdien-moj-gov-vn",
    )
    anle_candidates: tuple[str, ...] = (
        "tmquan/anle-toaan-gov-vn",
        "Vietnamese-Legal/anle-toaan-gov-vn",
    )

    # Chunking
    max_article_chars: int = 8000       # Tách theo Khoản nếu Điều dài hơn
    min_chunk_chars: int = 80           # Bỏ chunk quá ngắn

    # SME scoring keywords
    keywords_high: tuple[str, ...] = (
        "doanh nghiệp nhỏ và vừa", "dnnvv", "sme",
        "hỗ trợ doanh nghiệp", "luật doanh nghiệp",
        "thuế giá trị gia tăng", "thuế thu nhập doanh nghiệp",
        "hóa đơn điện tử", "bảo hiểm xã hội",
        "lao động", "kế toán", "sở hữu trí tuệ",
        "nhãn hiệu", "hợp đồng", "thương mại",
    )
    keywords_medium: tuple[str, ...] = (
        "doanh nghiệp", "công ty", "hộ kinh doanh",
        "thuế", "hóa đơn", "người lao động",
        "báo cáo tài chính", "vốn điều lệ",
        "đấu thầu", "tín dụng", "mặt bằng sản xuất",
    )

    # Loại văn bản (để infer từ text)
    doc_type_patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Thông tư liên tịch", ("thông tư liên tịch", "ttlt-")),
        ("Nghị quyết liên tịch", ("nghị quyết liên tịch",)),
        ("Nghị định", ("nghị định", "nđ-cp", "nd-cp")),
        ("Thông tư", ("thông tư", "tt-btc", "tt-blđtbxh")),
        ("Quyết định", ("quyết định", "qđ-ttg")),
        ("Nghị quyết", ("nghị quyết", "nq-cp")),
        ("Pháp lệnh", ("pháp lệnh",)),
        ("Bộ luật", ("bộ luật",)),
        ("Luật", ("luật",)),
        ("Án lệ", ("án lệ",)),
    )


# ---------------------------------------------------------------------------
# Index config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexConfig:
    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_batch_size: int = 5000

    # Embedding model — ưu tiên domain-specific Vietnamese legal
    embedding_model_candidates: tuple[str, ...] = (
        "mainguyen9/vietlegal-e5",                          # Fine-tuned legal VN, NDCG@10=0.7229
        "cyhapun/vn-legal-embedding-v1",                    # BGE-M3 fine-tuned legal VN
        "BAAI/bge-m3",                                      # Multilingual, 1024-dim, strong semantic
    )
    embedding_batch_size: int = 32
    embedding_max_length: int = 512

    # ChromaDB collection name
    chroma_collection: str = "vpl_legal"


# ---------------------------------------------------------------------------
# Search / Retrieval config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchConfig:
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rrf_k: int = 60             # RRF constant: RRF(d) = Σ 1/(k + rank_i(d))
    fusion_top_k: int = 30      # Candidates đưa vào reranker sau RRF
    fusion_reranker_weight: float = 0.8         # Tỷ trọng điểm Reranker (so với điểm nền BM25+BGE)
    fusion_base_weight: float = 0.2             # Tỷ trọng điểm nền RRF (so với điểm Reranker)

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 16

    # Dynamic thresholds (F2-optimized: recall >> precision)
    # v2 tune: tăng threshold để cắt articles rác, giảm spam
    high_conf_threshold: float = 0.62   # → relevant_articles (tăng từ 0.60)
    safe_threshold: float = 0.52        # → LLM context (tăng từ 0.42)
    min_articles: int = 0               # Cho phép nộp mảng rỗng [] nếu không tìm thấy luật
    max_articles: int = 3               # Cap relevant_articles (giảm từ 5, ép Precision)
    max_context_chunks: int = 25        # Cap chunks gửi cho LLM

    # HyDE expansion
    hyde_enabled: bool = True
    hyde_max_tokens: int = 150          # Độ dài hypothetical doc


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationConfig:
    # Gemma-2-9B-it (June 2024, < 14B, Apache 2.0, open-source)
    model_name: str = os.getenv(
        "VPL_MODEL_NAME",
        "unsloth/gemma-2-9b-it-bnb-4bit",
    )
    # Fallback models nếu OOM
    model_fallbacks: tuple[str, ...] = (
        "unsloth/gemma-2-2b-it-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    )
    max_seq_length: int = int(os.getenv("VPL_MAX_SEQ_LEN", "8192"))
    max_new_tokens: int = int(os.getenv("VPL_MAX_NEW_TOKENS", "1024"))
    max_context_chars: int = int(os.getenv("VPL_MAX_CTX_CHARS", "7000"))
    batch_size: int = int(os.getenv("VPL_BATCH_SIZE", "4"))
    temperature: float = float(os.getenv("VPL_TEMPERATURE", "0.0"))
    top_p: float = float(os.getenv("VPL_TOP_P", "0.9"))
    repetition_penalty: float = float(os.getenv("VPL_REP_PENALTY", "1.1"))
    load_in_4bit: bool = os.getenv("VPL_4BIT", "1") != "0"


# ---------------------------------------------------------------------------
# Submission config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubmissionConfig:
    required_fields: frozenset[str] = frozenset(
        {"id", "question", "answer", "relevant_docs", "relevant_articles"}
    )
    expected_count: int = 2000
    standard_warning: str = (
        "Cảnh báo giới hạn: Đây là tư vấn sơ bộ từ AI, doanh nghiệp "
        "cần đối chiếu văn bản gốc hoặc tham khảo chuyên gia pháp lý "
        "trước khi áp dụng."
    )


# ---------------------------------------------------------------------------
# Singleton instances — import trực tiếp
# ---------------------------------------------------------------------------

CORPUS = CorpusConfig()
INDEX = IndexConfig()
SEARCH = SearchConfig()
GENERATION = GenerationConfig()
SUBMISSION = SubmissionConfig()
