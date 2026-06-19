# VietPhapLy RAG - Road2AI Legal Assistant

**Team: impact** | **Current Score:** `Macro F2 0.3211` *(Beating Official Baseline)*

Giải pháp Legal RAG (Retrieval-Augmented Generation) dành riêng cho hệ thống pháp luật Việt Nam, được phát triển cho cuộc thi **Road2AI 2026**. Hệ thống được thiết kế tối ưu để tra cứu văn bản pháp luật, nghị định, thông tư một cách chính xác và chống ảo giác (hallucination) nghiêm ngặt.

## Điểm nổi bật (Key Features)

- **Mô hình ngôn ngữ (LLM):** `Gemma-2-9B-it` - Tối ưu hóa prompt để trả lời chính xác, đóng vai chuyên gia pháp lý và chủ động từ chối trả lời nếu hệ thống không tìm thấy luật định liên quan.
- **Hệ thống truy xuất (Retrieval):** Khởi điểm cực kỳ ổn định với **BM25** (bắt keyword, số hiệu luật chuẩn xác) kết hợp caching qua SQLite. Nền tảng được thiết kế sẵn sàng để mở rộng sang Hybrid Search (BM25 + BGE-M3 + Reranker).
- **Tối ưu phần cứng:** Quản lý VRAM tự động (`gc.collect()`, `torch.cuda.empty_cache()`), cơ chế chia lô (batching) và retry thông minh giúp chạy mượt mà 2000 câu trên Google Colab (T4/L4) mà không bị Out-Of-Memory.
- **Post-Processing 3 tầng:** Tự động lọc các điều luật ảo do LLM tự bịa, bổ sung citations (trích dẫn) bị thiếu một cách tự động để vượt qua khâu kiểm duyệt gắt gao (Validation) của hệ thống chấm điểm.

## Hướng dẫn cài đặt (Setup)

1. **Clone repository:**
   ```bash
   git clone https://github.com/Platypus27-coder/sme-legal-assistant.git
   cd sme-legal-assistant
   ```

2. **Cài đặt môi trường (Khuyên dùng Conda):**
   ```bash
   conda create -n exact-env python=3.10 -y
   conda activate exact-env
   pip install -r requirements.txt
   pip install -e .
   ```

## Hướng dẫn sử dụng (Usage)

Dự án được gom gọn vào duy nhất một entry point là `run.py`. Bạn có thể chạy toàn bộ hệ thống bằng 1 lệnh, hoặc chạy từng bước để dễ dàng theo dõi và debug.

### 1. Chạy từng bước (Khuyên dùng trên Colab/Drive để tránh mất dữ liệu)

- **Bước 1 - Ingest:** Thu thập và cắt nhỏ văn bản luật (chunking) theo Điều/Khoản.
  ```bash
  python run.py ingest
  ```
- **Bước 2 - Index:** Xây dựng cơ sở dữ liệu tìm kiếm (BM25 và Vector).
  ```bash
  python run.py index --device cuda
  ```
- **Bước 3 - Retrieve:** Tìm kiếm các đoạn luật liên quan cho từng câu hỏi và lưu vào Cache SQLite. Quá trình này giúp tách biệt việc tìm kiếm và sinh văn bản.
  ```bash
  python run.py retrieve --device cuda
  ```
- **Bước 4 - Generate:** Chạy mô hình LLM (Gemma 2) để đọc luật từ Cache và viết câu trả lời.
  ```bash
  python run.py generate --device cuda
  ```
- **Bước 5 - Submit:** Kiểm tra lỗi (Validate), tự động vá lỗi trích dẫn và nén thành file `submission.zip` sẵn sàng nộp bài.
  ```bash
  python run.py submit
  ```

### 2. Chạy toàn bộ luồng tự động (End-to-End Pipeline)

```bash
# Tự động chạy tuần tự từ Ingest -> Submit
python run.py pipeline --device cuda

# Bỏ qua Ingest/Index nếu đã tạo sẵn Database trên Google Drive
python run.py pipeline --skip-ingest --skip-index --device cuda
```

### 3. Tiện ích tinh chỉnh nhanh (Retune)
Nếu muốn đổi ngưỡng giới hạn số lượng điều luật tối đa/tối thiểu được hiển thị mà **không cần mất hàng chục tiếng để GPU chạy lại LLM**:
```bash
python run.py retune --min-articles 3 --max-articles 8 --safe-threshold 0.3
```

## Cấu trúc thư mục (Project Structure)

```text
vietphaply-rag/
├── src/vpl/                # Mã nguồn chính của pipeline RAG
│   ├── settings.py         # File cấu hình trung tâm (Đường dẫn, siêu tham số, LLM limits)
│   ├── corpus/             # Tải dữ liệu & Chunking logic
│   ├── store/              # Xây dựng Index (BM25, ChromaDB)
│   ├── search/             # Logic tìm kiếm (Hybrid, Reranker)
│   ├── answer/             # Prompt engineering & Post-processing chống ảo giác
│   ├── cache.py            # SQLite database cho Retrieval results (crash-safe)
│   ├── pipeline.py         # Điều phối toàn bộ quy trình
│   └── submit.py           # Đóng gói ZIP nộp bài & Validate
├── run.py                  # Entry point CLI chính
├── notebooks/              # Jupyter notebooks dùng để EDA & test
├── docs/                   # Tài liệu thiết kế kiến trúc hệ thống
├── data/                   # Chứa dữ liệu đầu vào (VD: R2AIStage1DATA.json)
├── artifacts/              # Tự động sinh ra khi chạy (Cache, db, file Output)
└── tests/                  # Unit tests
```

## Lộ trình phát triển (Roadmap)
- [x] Xây dựng kiến trúc end-to-end trên Google Colab.
- [x] Thiết lập BM25 Baseline an toàn, không OOM.
- [x] Hoàn thiện Post-processing tự động vá lỗi trích dẫn (Validation Passed).
- [x] Cán mốc điểm số `0.3211` vượt Baseline chính thức.
- [ ] Tích hợp **Hybrid Search (BM25 + BGE-M3 + Reranker)** để bắt ngữ nghĩa ẩn và bứt phá thứ hạng Top 10.
