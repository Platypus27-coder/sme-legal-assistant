# VietPhapLy RAG

Vietnamese Legal RAG pipeline for **R2AI2026** competition.

- **Model**: Gemma-2-9B-it (June 2024, 9B < 14B, Apache 2.0)
- **Embedding**: `mainguyen9/vietlegal-e5` (domain-specific, NDCG@10=0.7229)
- **Vector DB**: ChromaDB (persistent local, zero server)
- **Retrieval**: BM25 + Dense + RRF + Cross-encoder reranker + HyDE expansion
- **Metric target**: Macro F2 (recall-heavy)

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design documentation.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Full pipeline
python run.py pipeline --device cuda

# Step by step
python run.py ingest
python run.py index --device cuda
python run.py retrieve --device cuda
python run.py generate --device cuda
python run.py submit

# Skip completed phases
python run.py pipeline --skip-ingest --skip-index --device cuda

# Local evaluation
python run.py eval --pred artifacts/output/results.json --ref data/dev_set.json

# Retune thresholds without rerunning LLM
python run.py retune --min-articles 3 --max-articles 8 --safe-threshold 0.25
```

## Project Structure

```text
vietphaply-rag/
├── src/vpl/                # Mã nguồn chính của pipeline
│   ├── settings.py         # File cấu hình duy nhất tập trung mọi hằng số
│   ├── corpus/             # Tải dữ liệu từ HuggingFace & Chunking theo Điều/Khoản
│   ├── store/              # Xây dựng index (BM25 + ChromaDB vector store)
│   ├── search/             # Hybrid retrieval (BM25 + Dense), RRF fusion, Reranker, HyDE
│   ├── answer/             # Sinh câu trả lời với Gemma-2-9B & Post-processing 3 tầng
│   ├── cache.py            # SQLite database cho việc cache retrieval results (crash-safe)
│   ├── pipeline.py         # Điều phối toàn bộ quy trình end-to-end
│   ├── evaluate.py         # Đánh giá tự động Macro F2 & Silver Recall
│   └── submit.py           # Đóng gói và validate định dạng file zip nộp bài nộp
├── run.py                  # Entry point duy nhất chứa các subcommands (ingest, index, retrieve...)
├── notebooks/              # Jupyter notebooks dùng cho EDA, thử nghiệm và debug
├── docs/                   # Tài liệu thiết kế kiến trúc
├── data/                   # Thư mục chứa dữ liệu đầu vào (VD: R2AIStage1DATA.json)
├── artifacts/              # Thư mục chứa kết quả sinh ra (raw data, index, cache, output)
└── tests/                  # Unit tests cho các module
```
