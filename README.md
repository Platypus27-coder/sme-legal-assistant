# VietPhapLy RAG - Road2AI Legal Assistant

Giải pháp Legal RAG (Retrieval-Augmented Generation) dành riêng cho hệ thống pháp luật Việt Nam, được phát triển cho cuộc thi **Road2AI 2026**. Hệ thống được thiết kế tối ưu để tra cứu văn bản pháp luật, nghị định, thông tư một cách chính xác và chống ảo giác (hallucination) nghiêm ngặt.

## Điểm nổi bật (Key Features)

- **Mô hình ngôn ngữ (LLM):** `Gemma-2-9B-it` - Tối ưu hóa prompt để trả lời chính xác, đóng vai chuyên gia pháp lý và chủ động từ chối trả lời nếu hệ thống không tìm thấy luật định liên quan.
- **Hệ thống truy xuất (Retrieval):** Khởi điểm cực kỳ ổn định với **BM25** (bắt keyword, số hiệu luật chuẩn xác) kết hợp caching qua SQLite. Nền tảng được thiết kế sẵn sàng để mở rộng sang Hybrid Search (BM25 + BGE-M3 + Reranker).
- **Tối ưu phần cứng:** Quản lý VRAM tự động (`gc.collect()`, `torch.cuda.empty_cache()`), cơ chế chia lô (batching) và retry thông minh giúp chạy mượt mà 2000 câu trên Google Colab (T4/L4) mà không bị Out-Of-Memory.
- **Post-Processing 3 tầng:** Tự động lọc các điều luật ảo do LLM tự bịa, bổ sung citations (trích dẫn) bị thiếu một cách tự động để vượt qua khâu kiểm duyệt gắt gao (Validation) của hệ thống chấm điểm.


## Link tải Dữ liệu và Mô hình (Dành cho Ban Giám khảo)

Để hệ thống hoạt động, Ban Giám khảo vui lòng tải dữ liệu và mô hình theo hướng dẫn sau:

1. **Dữ liệu hệ thống (Data):**
   - **Đường dẫn tải dữ liệu:** [ĐIỀN LINK GOOGLE DRIVE CỦA BẠN VÀO ĐÂY]
   - **Cách sử dụng:** Tải file .zip từ đường link trên, giải nén và đặt toàn bộ các file (R2AIStage1DATA.json, phapdien.jsonl, nle.jsonl) vào thư mục data/ ở thư mục gốc của mã nguồn.

2. **Trọng số Mô hình (Checkpoints):**
   - Giải pháp của chúng tôi sử dụng mô hình mã nguồn mở nguyên bản kết hợp RAG thay vì Fine-tuning, do đó **không có Checkpoint cục bộ nào cần phải tải thủ công**.
   - Mã nguồn sẽ tự động kết nối và tải các trọng số này thông qua HuggingFace khi chạy lệnh:
     - LLM: [unsloth/gemma-2-9b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-2-9b-it-bnb-4bit)
     - Reranker: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
     - Embeddings: [mainguyen9/vietlegal-e5](https://huggingface.co/mainguyen9/vietlegal-e5)

## Quy trình hoạt động (Baseline Pipeline)

Phiên bản hiện tại đang hoạt động dựa trên luồng quy trình 5 bước độc lập, được thiết kế để chống đứt gãy (crash-safe) trên môi trường Google Colab:

```text
[Dữ liệu Luật Thô] 
        │
        ▼ (Chunking)
   [1. Ingest] ─────────────► (Từ điển BM25)
                                     │
[2000 Câu Hỏi] ────► [2. Retrieve] ◄─┘ (Tìm kiếm Keyword)
                           │
                           ▼
                    (SQLite Cache)
                           │
                           ▼
 [Gemma-2-9B-it] ──► [3. Generate]
                           │
                           ▼ (Câu trả lời thô)
                    [4. Post-Process] (Lọc ảo giác & Vá lỗi)
                           │
                           ▼
                  [submission.zip]
```

1. **Ingest (Xử lý dữ liệu thô):** Tải các bộ luật, nghị định, thông tư và án lệ. Băm nhỏ văn bản (chunking) theo từng Điều/Khoản riêng biệt để LLM dễ đọc, loại bỏ các đoạn quá ngắn.
2. **Index (Lập chỉ mục tìm kiếm):** Quét toàn bộ các đoạn luật vừa cắt và đưa vào từ điển tìm kiếm từ khóa (BM25 Sparse Index). Ở phiên bản này, hệ thống tập trung hoàn toàn vào việc bắt keyword và số hiệu luật chính xác tuyệt đối.
3. **Retrieve (Truy xuất tài liệu):** Đọc 2000 câu hỏi, dùng thuật toán để tìm ra các Điều luật liên quan nhất cho từng câu. Lưu toàn bộ kết quả tìm được vào một Database trung gian (SQLite Cache) để tách biệt hoàn toàn khâu tìm kiếm và khâu sinh text.
4. **Generate (Sinh câu trả lời - LLM):** Bật mô hình `Gemma-2-9B-it` lên GPU. Mô hình đọc câu hỏi và các Điều luật tương ứng lấy từ SQLite Cache. Nó đóng vai trò một chuyên gia pháp lý để phân tích, tổng hợp và viết ra câu trả lời cuối cùng. Quá trình này được chia lô nhỏ (batching) và dọn rác VRAM liên tục để chống tràn bộ nhớ (Out-of-Memory).
5. **Post-Process & Submit (Hậu xử lý):** Quét lại câu trả lời của LLM. Tự động xóa bỏ các điều luật "ảo" (hallucinations) do AI tự bịa ra. Bổ sung các trích dẫn pháp lý (citations) bị thiếu vào cuối câu trả lời để đảm bảo vượt qua vòng kiểm duyệt (Validation) gắt gao của hệ thống chấm điểm, sau đó nén thành `submission.zip`.

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

### 1. Chạy toàn bộ luồng tự động (End-to-End Pipeline)

```bash
# Tự động chạy tuần tự từ Ingest -> Submit
python run.py pipeline --device cuda

# Bỏ qua Ingest/Index nếu đã tạo sẵn Database trên Google Drive
python run.py pipeline --skip-ingest --skip-index --device cuda
```

### 2. Chạy từng bước (Khuyên dùng trên Colab/Drive để tránh mất dữ liệu)

- **Bước 1 - Ingest:** Thu thập và cắt nhỏ văn bản luật (chunking) theo Điều/Khoản.
  ```bash
  python run.py ingest
  ```
- **Bước 2 - Index:** Xây dựng cơ sở dữ liệu tìm kiếm (BM25 và Vector).
  ```bash
  python run.py index --device cuda
  ```
- **Bước 3 - Retrieve:** Tìm kiếm các đoạn luật liên quan cho từng câu hỏi và lưu vào Cache SQLite.
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

### 3. Tiện ích tinh chỉnh nhanh (Retune)
Nếu muốn đổi ngưỡng giới hạn số lượng điều luật tối đa/tối thiểu được hiển thị mà **không cần mất hàng chục tiếng để GPU chạy lại LLM**:
```bash
python run.py retune --min-articles 0 --max-articles 2 --safe-threshold 0.58
```

## Cấu trúc thư mục (Project Structure)

```text
vietphaply-rag/
├── src/
│   └── vpl/                        # Package chính
│       ├── settings.py             # Tất cả config tập trung 1 file
│       ├── corpus/                 # Thu thập & xử lý dữ liệu
│       ├── store/                  # Xây dựng index (BM25, ChromaDB)
│       ├── search/                 # Retrieval pipeline (Hybrid, Reranker)
│       ├── answer/                 # Generation pipeline (LLM, Prompts, Postprocess)
│       ├── cache.py                # SQLite retrieval cache (crash-safe)
│       ├── pipeline.py             # End-to-end orchestrator
│       ├── evaluate.py             # Macro F2 + Silver Recall
│       └── submit.py               # Validation + ZIP packaging
│
├── tools/
│   └── optimize_submission.py      # Script tinh chỉnh offline
├── notebooks/                      # Khám phá dữ liệu (EDA)
├── run.py                          # Entry point duy nhất (subcommands)
├── data/                           # Competition data
├── artifacts/                      # Generated artifacts (gitignored)
├── tests/
├── pyproject.toml
└── requirements.txt
```
