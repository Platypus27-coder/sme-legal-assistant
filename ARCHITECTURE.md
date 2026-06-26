# VietPhapLy RAG — Kiến Trúc Hệ Thống

> **Cuộc thi**: R2AI2026 — Legal Information Retrieval & QA  
> **Mục tiêu**: Trả lời 2000 câu hỏi pháp lý SME, tối ưu macro **F2 score** (Recall × 4 so với Precision)  
> **Ràng buộc LLM**: < 14B params, open-source, công bố trước 01/03/2026  
> **Môi trường**: Google Colab T4 (15GB RAM, 16GB VRAM)

---

## 1. Tổng Quan Pipeline

```
Raw Data (HuggingFace)
        │
        ▼
┌─────────────────┐
│   corpus/       │  Thu thập + Chunking + Schema
│   loader.py     │  → phapdien, anle
│   chunker.py    │  → tách theo Điều/Khoản
│   schema.py     │  → LegalChunk dataclass
└────────┬────────┘
         │ chunks.jsonl
         ▼
┌─────────────────┐
│   store/        │  Indexing
│   bm25.py       │  → BM25Okapi + underthesea + legal phrases
│   vectors.py    │  → ChromaDB persistent (vietlegal-e5 embeddings)
└────────┬────────┘
         │ BM25 pickle + ChromaDB
         ▼
┌─────────────────┐
│   search/       │  Hybrid Retrieval
│   expander.py   │  → HyDE query expansion
│   hybrid.py     │  → BM25 (Top50) + Dense (Top50) → RRF fusion
│   reranker.py   │  → BGE-reranker-v2-m3 cross-encoder
└────────┬────────┘
         │ Top-K scored chunks (cache → SQLite)
         ▼
┌─────────────────┐
│   answer/       │  Generation + Post-processing
│   generator.py  │  → Gemma-2-9B-it (4-bit Unsloth)
│   prompts.py    │  → System + Context + Format prompts
│   postprocess.py│  → Hallucination removal + citation fallback
└────────┬────────┘
         │ results_partial.jsonl (crash-safe streaming)
         ▼
┌─────────────────┐
│   submit.py     │  Validation + ZIP packaging
│   evaluate.py   │  → Macro F2, Silver Recall
└─────────────────┘
         │
         ▼
    submission.zip
```

---

## 2. Cấu Trúc Thư Mục

```
vietphaply-rag/
├── src/
│   └── vpl/                        # Package chính
│       ├── __init__.py
│       ├── settings.py             # Tất cả config tập trung 1 file
│       ├── corpus/                 # Thu thập & xử lý dữ liệu
│       │   ├── __init__.py
│       │   ├── loader.py           # Tải HuggingFace datasets (phapdien + anle)
│       │   ├── chunker.py          # Structural chunking theo Điều/Khoản
│       │   └── schema.py           # LegalChunk, ChunkMeta dataclasses
│       ├── store/                  # Xây dựng index
│       │   ├── __init__.py
│       │   ├── bm25.py             # BM25 indexer + legal tokenizer
│       │   └── vectors.py          # ChromaDB manager + embedding
│       ├── search/                 # Retrieval pipeline
│       │   ├── __init__.py
│       │   ├── expander.py         # HyDE query expansion
│       │   ├── hybrid.py           # BM25 + Dense + RRF fusion
│       │   └── reranker.py         # Cross-encoder reranker
│       ├── answer/                 # Generation pipeline
│       │   ├── __init__.py
│       │   ├── generator.py        # Gemma-2-9B-it batched inference
│       │   ├── prompts.py          # Prompt templates
│       │   └── postprocess.py      # 3-tier post-processing
│       ├── cache.py                # SQLite retrieval cache (crash-safe)
│       ├── pipeline.py             # End-to-end orchestrator
│       ├── evaluate.py             # Macro F2 + Silver Recall
│       └── submit.py               # Validation + ZIP packaging
│
├── run.py                          # Entry point duy nhất (subcommands)
├── notebooks/                      # Experiment & EDA
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_eval.ipynb
│   └── 03_threshold_tuning.ipynb
├── data/                           # Competition data
│   └── R2AIStage1DATA.json
├── artifacts/                      # Generated artifacts (gitignored)
│   ├── raw/                        # Raw collected data
│   ├── index/                      # BM25 + ChromaDB
│   ├── cache/                      # SQLite retrieval cache
│   └── output/                     # Results + submission
├── tests/
├── pyproject.toml
├── requirements.txt
├── requirements-gpu.txt
└── README.md
```

---

## 3. Chi Tiết Từng Module

### 3.1 `settings.py` — Config Tập Trung

Toàn bộ hằng số và config nằm ở **1 file duy nhất**, không phân tán theo phase.

```python
# Nhóm theo domain, không phải theo phase
class CorpusConfig: ...       # dataset names, chunking params
class IndexConfig: ...        # BM25 params, embedding model
class SearchConfig: ...       # top-k, RRF k, thresholds
class GenerationConfig: ...   # model name, batch size, temperature
class SubmissionConfig: ...   # output paths, format rules
```

---

### 3.2 `corpus/` — Thu Thập & Chunking

**loader.py**
- Tải `phapdien-moj-gov-vn` (config `articles`) và `anle-toaan-gov-vn` từ HuggingFace
- Normalize: extract `doc_id`, `doc_type`, `doc_title`, `article_number`, `source_note`
- SME scoring: keyword matching để ưu tiên văn bản liên quan SME
- Output: `artifacts/raw/legal_docs.jsonl`, `artifacts/raw/precedents.jsonl`

**chunker.py**
- Chunking rule: mỗi **Điều** = 1 chunk
- Nếu Điều > 8000 ký tự → tách theo **Khoản** (paragraph split)
- Án lệ: 1 chunk per vụ án (gộp toàn bộ nội dung)
- Canonicalize: 1 title chuẩn per `doc_id` (lấy title phổ biến nhất)
- Dedup: `chunk_id` collision → suffix `_dup_N`
- Output: `artifacts/raw/chunks.jsonl`

**schema.py**
```python
@dataclass(frozen=True)
class ChunkMeta:
    doc_id: str           # "04/2017/QH14"
    doc_type: str         # "Luật"
    doc_title: str        # "Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
    article_number: str   # "Điều 4"
    formatted_doc: str    # "04/2017/QH14|Luật 04/2017/QH14 ..."
    formatted_article: str# "04/2017/QH14|...|Điều 4"
    submission_eligible: bool
    sme_score: float

@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    text: str
    meta: ChunkMeta
```

> **Lưu ý**: Format `formatted_doc` và `formatted_article` tuân thủ **chính xác** yêu cầu ban tổ chức:
> - `relevant_docs`: `<mã VB>|<Loại VB> <Mã VB> <Trích yếu>`
> - `relevant_articles`: `<mã VB>|<Loại VB> <Mã VB> <Trích yếu>|<Điều X>`

---

### 3.3 `store/` — Indexing

**bm25.py**
- Tokenizer: `underthesea.word_tokenize` (thực sự, không stub) + legal phrase extraction
- Legal phrases: 80+ thuật ngữ pháp lý cố định (bigram compound terms)
- Fallback: regex tokenizer nếu underthesea không available
- Parameters: `k1=1.5`, `b=0.75`
- Persist: `artifacts/index/bm25/corpus.pkl` + `chunk_id_map.json`

**vectors.py**
- Embedding model: **`mainguyen9/vietlegal-e5`** (fine-tuned trên legal VN, NDCG@10=0.7229)
  - Fallback: `cyhapun/vn-legal-embedding-v1` (bge-m3 fine-tuned cho pháp luật VN)
  - Fallback 2: `BAAI/bge-m3` (generic multilingual)
- Vector DB: **ChromaDB** (persistent local, không cần server)
- Batch encoding với checkpoint mỗi 5000 chunks
- Persist: `artifacts/index/chroma/`

> **Tại sao ChromaDB thay Qdrant?**  
> - Zero server setup — chỉ cần `chromadb` package  
> - API đơn giản hơn, không cần LlamaIndex wrapper  
> - Đủ cho scale ~100K chunks trên Colab

---

### 3.4 `search/` — Hybrid Retrieval

**expander.py — HyDE (Hypothetical Document Expansion)**
```
Input:  "Doanh nghiệp nhỏ có được hỗ trợ vay vốn không?"
↓ LLM sinh ra hypothetical answer ngắn (~100 tokens)
Output: "Theo Điều 21 Luật Hỗ trợ DNNVV, doanh nghiệp nhỏ được ưu đãi lãi suất..."
↓ Encode cả query gốc + hypothetical doc
→ Dense retrieval tốt hơn vì search vector space gần với docs thực
```

> Đây là cải tiến đáng kể giúp tăng cường độ chính xác khi tìm kiếm bằng semantic search.

**hybrid.py — RRF Fusion**
```
Query → [BM25 Top-50] + [Dense Top-50 (với HyDE)]
                    ↓
            RRF(d) = Σ 1/(60 + rank_i(d))
                    ↓
              Top-30 candidates
                    ↓
           Cross-encoder Reranker
                    ↓
              Final Top-K (scored)
```

- Diversity filter: tối đa 1 chunk per `formatted_article`
- Lexical boost: +0.02 nếu doc_id/article_number khớp query

**reranker.py**
- Model: `BAAI/bge-reranker-v2-m3`
- Score blend: `0.8 × rerank_score + 0.2 × rrf_score`
- Dynamic threshold:
  - `HIGH_CONF = 0.5` → `relevant_articles`
  - `SAFE = 0.3` → LLM context
  - Fallback: lấy top-3 nếu không đủ high-conf

---

### 3.5 `cache.py` — SQLite Retrieval Cache

Thay JSONL offset-based bằng **SQLite** — truy vấn nhanh hơn, dễ debug:

```sql
CREATE TABLE retrieval_cache (
    question_id  INTEGER PRIMARY KEY,
    question     TEXT NOT NULL,
    chunks_json  TEXT NOT NULL,   -- JSON serialized chunks
    cached_at    TEXT NOT NULL
);
```

- Crash-safe: mỗi row là 1 transaction riêng
- Resume: `SELECT question_id FROM retrieval_cache` → skip đã có
- Debug: `SELECT * WHERE question_id = 42` — đơn giản hơn JSONL offset

---

### 3.6 `answer/` — Generation

**generator.py — Gemma-2-9B-it**

| Spec | Giá trị |
|---|---|
| Model | `unsloth/gemma-2-9b-it-bnb-4bit` |
| Release | June 2024 ✅ (trước 01/03/2026) |
| Params | 9B (< 14B limit ✅) |
| VRAM (4-bit) | ~6GB |
| Context | 8192 tokens |
| License | Apache 2.0 ✅ |

Fallback chain: Gemma-2-9B → Gemma-2-2B-it → Qwen2.5-3B (nếu VRAM thiếu)

**prompts.py**
- System: ràng buộc pháp lý, ngôn ngữ, format
- Context: grouped by `doc_id` để tránh nhầm số Điều giữa các văn bản
- Format: 4 phần bắt buộc (Căn cứ / Phân tích / Tư vấn / Cảnh báo)

**postprocess.py — 3 tầng**
1. **Regex Extraction**: tìm `Điều X` trong answer
2. **Context Validation**: đối chiếu với retrieved chunks → loại hallucination
3. **Citation Fallback**: nếu LLM không cite → append từ top-scored chunks

---

### 3.7 `run.py` — Entry Point Duy Nhất

```bash
# Chạy toàn bộ pipeline
python run.py pipeline --device cuda

# Chạy từng bước
python run.py ingest          # corpus/loader + chunker
python run.py index           # store/bm25 + vectors
python run.py retrieve        # search/ → populate SQLite cache
python run.py generate        # answer/ → results_partial.jsonl
python run.py submit          # validate + zip
```

---

## 4. Các Ý Tưởng Thiết Kế Cốt Lõi

Những tính năng dưới đây đã được thiết kế tối ưu và giữ vai trò quan trọng trong pipeline:

| Tính năng | Lợi ích mang lại |
|---|---|
| Crash-safe streaming JSONL cho generation output | ✅ Giữ pattern `append + fsync` để chống mất dữ liệu khi đứt kết nối |
| `formatted_doc` / `formatted_article` schema | ✅ Tuân thủ định dạng khắt khe của hệ thống nộp bài Kaggle |
| Canonicalize document titles | ✅ Lấy title phổ biến nhất per doc_id |
| RRF fusion k=60 | ✅ Công thức chuẩn kết hợp BM25 và Vector Search |
| Sigmoid Reranking Transformation | ✅ Chuyển đổi logits thô của BGE-M3 thành dải xác suất [0,1] chuẩn mực |
| 3-tier post-processing | ✅ Regex → Validation → Fallback |
| Dynamic threshold (F2 optimization) | ✅ Tham số hóa `HIGH_CONF`, `SAFE_CONF`, `min_articles` để dễ dàng tinh chỉnh |
| Document-grouped context format | ✅ Tránh nhầm Điều giữa các văn bản khi đưa vào prompt |
| Silver retrieval recall check | ✅ Tích hợp script đánh giá nội bộ |
| Retune thresholds offline | ✅ Tái sử dụng SQLite Cache để tinh chỉnh ngưỡng mà không cần chạy lại LLM |

---

## 5. Chiến Lược Tối Ưu F2

```text
F2 = (1 + 4) × P × R / (4P + R) = 5PR / (4P + R)
→ Recall quan trọng gấp 4 lần Precision
→ Thà bắt thêm (precision thấp) hơn bỏ sót (recall thấp)
```

**Retrieval**: BM25 (exact match keyword, số hiệu luật) + Dense Vector (semantic search) → bổ sung ưu khuyết điểm cho nhau.  
**HyDE**: Sinh câu trả lời giả định (Hypothetical Document) để cải thiện dense recall cho các câu hỏi paraphrase đa dạng.  
**Reranker**: Dùng `BGE-M3 Cross-Encoder` để chấm điểm lại chính xác độ liên quan giữa Câu hỏi và Điều luật.  
**Threshold**: Hệ thống áp dụng 2 ngưỡng linh hoạt: 
- `HIGH_CONF` để đưa thẳng vào danh sách `relevant_articles`.
- `SAFE` (thấp hơn) để đưa thêm vào ngữ cảnh cho LLM tự phân tích.  

---

## 6. Rủi Ro & Giải Pháp Tương Ứng

| Rủi ro | Mitigation |
|---|---|
| Colab crash giữa quá trình LLM Generation (9 tiếng) | Đã thiết kế JSONL streaming + SQLite cache để có thể Resume ngay vị trí đứt gãy. |
| OOM (Out Of Memory) khi chạy BGE-Reranker | Bắt lỗi trực tiếp bằng `raise SystemExit(1)` thay vì âm thầm dùng lại điểm cũ, kết hợp giảm `batch_size` xuống mức an toàn. |
| LLM tự bịa số hiệu luật (Hallucination) | Tầng Post-Processing dùng Regex quét lại toàn bộ, nếu số hiệu không nằm trong tập Retrieved sẽ bị xóa ngay. |
| HyDE tốn thêm VRAM lúc retrieve | Có thể tắt hoặc dùng mô hình nhỏ gọn chuyên biệt chỉ cho HyDE expansion. |

---

## 7. Timeline Lộ Trình

```text
Giai đoạn 1: Ingest + Index (Xử lý dữ liệu và tạo từ điển BM25, ChromaDB)
Giai đoạn 2: Retrieval Pipeline + Cache (Thiết lập thuật toán Hybrid, Reranker và bộ nhớ đệm SQLite)
Giai đoạn 3: Generation + Post-processing (Tích hợp LLM sinh văn bản và hậu xử lý)
Giai đoạn 4: Iterate threshold/prompt, Final submission (Tối ưu hóa siêu tham số offline để tìm ra cấu hình đạt F2 cao nhất)
```

---

*Package: `vpl` | Model: Gemma-2-9B-it | Reranker: BGE-reranker-v2-m3 | VectorDB: ChromaDB*
