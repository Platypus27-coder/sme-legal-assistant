# Hướng Dẫn Quy Trình Chạy Pipeline R2AI2026

Tài liệu này hướng dẫn chi tiết cách thực thi toàn bộ luồng xử lý (pipeline) của hệ thống VietPhapLy RAG. Hệ thống được thiết kế dưới dạng CLI module hóa thông qua file `run.py`.

## 1. Chuẩn Bị Môi Trường
Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
# Nếu chạy trên máy có GPU (hoặc Google Colab):
pip install -r requirements-gpu.txt
```

> [!WARNING]
> Bước sinh văn bản (`generate`) cần GPU khá mạnh (tối thiểu 6-8GB VRAM cho bản ép nén 4-bit, khuyên dùng 16GB VRAM như Tesla T4 trên Colab). Chạy trên CPU cục bộ sẽ cực kỳ chậm.

---

## 2. Quy Trình Chạy Từng Bước (Khuyên Dùng Để Nắm Rõ Hệ Thống)

Chạy từng lệnh dưới đây theo đúng thứ tự. Kết quả mỗi bước sẽ tự động lưu vào thư mục `artifacts/`.

### Bước 1: Thu thập và Xử lý dữ liệu (Ingest)
Tải dữ liệu Pháp điển & Án lệ từ HuggingFace, phân loại và cắt nhỏ (chunking).
```bash
python run.py ingest
```
*Kết quả:* Các file `jsonl` nằm trong `artifacts/raw/`

### Bước 2: Xây dựng Cơ sở dữ liệu (Index)
Tạo BM25 Index (Sparse) và ChromaDB Vector Store (Dense). Nhúng (Embed) hàng chục nghìn đoạn luật.
```bash
python run.py index --device cuda
```
*(Bỏ `--device cuda` nếu máy bạn không có GPU NVIDIA)*
*Kết quả:* Các thư mục database nằm trong `artifacts/index/`

### Bước 3: Tìm kiếm pháp lý (Retrieve)
Thực hiện Hybrid Search (kết hợp BM25 + Semantic) và Reranker để lấy ra top các Điều Luật liên quan nhất cho 2000 câu hỏi.
```bash
python run.py retrieve --device cuda
```
*Kết quả:* File `retrieval.db` (SQLite) nằm trong `artifacts/cache/`

### Bước 4: Sinh câu trả lời (Generate)
Sử dụng mô hình Gemma-2-9B-it để đọc các Điều Luật đã tìm thấy và sinh ra câu trả lời cuối cùng.
```bash
python run.py generate --device cuda
```
> [!TIP]
> **Crash-safe:** Nếu bạn đang chạy trên Colab và bị ngắt kết nối giữa chừng, bạn chỉ cần mở lại và chạy đúng lệnh này. Hệ thống sẽ tự động đọc cache và chạy tiếp từ câu hỏi bị đứt quãng mà không phải bắt đầu lại từ đầu!
*Kết quả:* File `results_partial.jsonl` (cập nhật liên tục) và file `results.json` hoàn chỉnh nằm trong `artifacts/output/`

### Bước 5: Kiểm tra và Đóng gói (Submit)
Quét qua bộ kết quả, đảm bảo 100% tuân thủ định dạng nghiêm ngặt của BTC (`Mã VB|Loại VB Tên VB|Điều X`) và nén thành file chuẩn.
```bash
python run.py submit
```
*Kết quả:* File `submission.zip` đã sẵn sàng để upload lên Leaderboard.

---

## 3. Chạy Tự Động Toàn Bộ (End-to-End Pipeline)

Nếu bạn đã quen với pipeline và không muốn gọi lắt nhắt từng lệnh, chỉ cần chạy một dòng duy nhất:
```bash
python run.py pipeline --device cuda
```
Lệnh này sẽ nối toàn bộ quy trình: `ingest` → `index` → `retrieve` → `generate` → `submit`.

Bạn cũng có thể **bỏ qua các bước đã hoàn thành** để tiết kiệm thời gian, ví dụ như khi đã có index:
```bash
python run.py pipeline --skip-ingest --skip-index --device cuda
```

---

## 4. Công Cụ Đánh Giá và Tối Ưu (Evaluation & Tuning)

Sau khi có kết quả, thay vì nộp bài ngay làm tốn 1 trong 5 lượt nộp Private (theo luật BTC), bạn hãy dùng các tool sau:

**Đánh giá điểm số cục bộ (Local Eval):** Đo lường trực tiếp chỉ số Macro F2 (trọng tâm vào Recall) nếu có file dev.
```bash
python run.py eval --pred artifacts/output/results.json --ref data/dev_set.json
```

**Tinh chỉnh Ngưỡng (Retune):** Thay đổi thuật toán chọn luật bằng cách nâng/hạ ngưỡng tự tin (Threshold) mà **không cần mất thời gian chạy lại mô hình LLM**.
```bash
python run.py retune --min-articles 3 --max-articles 8 --safe-threshold 0.25
```
Lệnh này tái sử dụng dữ liệu trong `retrieval.db` (SQLite) rất nhẹ và nhanh.

---

## 5. Hướng Dẫn Chạy Trên Google Colab (Khuyên Dùng)

Google Colab cung cấp GPU Tesla T4 (16GB VRAM) miễn phí, rất phù hợp để chạy toàn bộ pipeline này. Dưới đây là các bước setup trên Colab:

**Bước 1:** Mở 1 sổ tay (Notebook) mới trên [Google Colab](https://colab.research.google.com/). Vào `Runtime` -> `Change runtime type` -> Chọn **T4 GPU**.

**Bước 2:** Clone source code từ Github về Colab:
```python
!git clone https://github.com/Platypus27-coder/sme-legal-assistant.git
%cd sme-legal-assistant
```

**Bước 3:** Cài đặt các thư viện (đã được tối ưu cho GPU/Colab):
```python
!pip install -r requirements-gpu.txt
```

**Bước 4 (Quan trọng):** Mount Google Drive để lưu kết quả. Colab sẽ xóa toàn bộ file nếu bị ngắt kết nối, do đó bạn cần lưu thư mục `artifacts/` thẳng vào Drive của bạn:
```python
from google.colab import drive
drive.mount('/content/drive')

# Tạo thư mục trên Drive và link nó vào repo
!mkdir -p /content/drive/MyDrive/R2AI_Artifacts
!rm -rf artifacts
!ln -s /content/drive/MyDrive/R2AI_Artifacts artifacts
```

**Bước 5:** Chạy toàn bộ pipeline (hoặc chạy từng lệnh `ingest`, `index`, `retrieve`, `generate` tương tự như trên local):
```python
!python run.py pipeline --device cuda
```
Khi chạy xong, toàn bộ kết quả (file zip, database) sẽ tự động nằm an toàn trong thư mục `R2AI_Artifacts` trên Google Drive của bạn!
