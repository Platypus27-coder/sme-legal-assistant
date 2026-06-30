# BẢN THUYẾT MINH SẢN PHẨM DỰ THI - ROAD2AI 2026
**Đội thi:** impact  
**Dự án:** Hệ thống RAG Trợ lý Pháp lý Doanh nghiệp Nhỏ và Vừa (SME Legal Assistant)

---

## 1. TÀI LIỆU MÔ TẢ DỮ LIỆU (DATA)

### 1.1. Nguồn dữ liệu sử dụng
Dữ liệu sử dụng trong hệ thống bao gồm 2 phần chính:
- **Dữ liệu Pháp điển (phapdien.jsonl):** Chứa các Điều/Khoản luật, Nghị định, Thông tư quy định về hỗ trợ Doanh nghiệp Nhỏ và Vừa.
- **Dữ liệu Án lệ (anle.jsonl):** Các án lệ thực tế của hệ thống Tòa án Việt Nam liên quan đến các tranh chấp và tình huống kinh doanh thực tiễn.
- **Dữ liệu Kiểm thử (R2AIStage1DATA.json):** 2000 câu hỏi pháp lý từ Ban tổ chức cung cấp để truy vấn hệ thống.

### 1.2. Cấu trúc và Định dạng dữ liệu
- **Định dạng:** JSON / JSONL.
- **Cấu trúc sau khi Ingest (Chunking):** Dữ liệu thô được băm nhỏ (chunking) theo đơn vị "Điều luật" độc lập để tối ưu hóa việc tìm kiếm. Mỗi Chunk chứa các metadata bắt buộc: `doc_id`, `article_number`, `formatted_doc`, `formatted_article`.

### 1.3. Hướng dẫn truy cập và Link dữ liệu
- Để hệ thống chạy được, thư mục dữ liệu phải được đặt tại đường dẫn: `data/` trong thư mục mã nguồn.
- **🔗 Link tải dữ liệu thô (Google Drive):** `[ĐIỀN LINK GOOGLE DRIVE CHỨA THƯ MỤC DATA CỦA BẠN VÀO ĐÂY]`

---

## 2. MÔ HÌNH VÀ CHECKPOINT SỬ DỤNG (MODEL)

Dự án tiếp cận theo hướng sử dụng các mô hình Ngôn ngữ Lớn (LLM) mã nguồn mở tiên tiến nhất hiện nay mà không cần huấn luyện lại (zero-shot/few-shot), kết hợp hệ thống RAG mạnh mẽ. Do đó, **không có checkpoint cục bộ nào cần phải tải lên Drive**.

### 2.1. Thông tin về các mô hình sử dụng
1. **Mô hình Sinh ngôn ngữ (Generation LLM):** `unsloth/gemma-2-9b-it-bnb-4bit` (Dựa trên kiến trúc Gemma-2 9B của Google, áp dụng lượng tử hóa 4-bit BitsAndBytes).
2. **Mô hình Nhúng Vector (Dense Embedding):** `mainguyen9/vietlegal-e5` (Tối ưu riêng cho dữ liệu pháp lý tiếng Việt).
3. **Mô hình Chấm điểm lại (Cross-Encoder Reranker):** `BAAI/bge-reranker-v2-m3` (Hỗ trợ đa ngôn ngữ, tăng độ chính xác tìm kiếm).

### 2.2. Hướng dẫn tải và sử dụng Checkpoint
- Khác với các mô hình tự huấn luyện (fine-tune) cục bộ, toàn bộ các mô hình của hệ thống được lưu trữ công khai (public) trên nền tảng **HuggingFace**.
- **Cách tự động tải:** Giám khảo không cần tải checkpoint thủ công. Khi chạy lệnh khởi chạy pipeline (`python run.py pipeline`), mã nguồn (thư viện `transformers` và `unsloth`) sẽ tự động kết nối và tải các Checkpoint trực tiếp từ HuggingFace Cache về máy.

---

## 3. MÃ NGUỒN (SOURCE CODE)

### 3.1. Tổng quan
- Toàn bộ mã nguồn (Source Code) đã được đóng gói và nén trong file nộp bài (hoặc link Github đính kèm). Mã nguồn tuân thủ tiêu chuẩn thiết kế hướng đối tượng (OOP) và kiến trúc dạng Modular (chia nhỏ thành các module Ingest, Index, Retrieve, Generate).
- Kiến trúc chi tiết được trình bày tại file: `ARCHITECTURE.md`.

### 3.2. Danh sách thư viện và Dependencies
Môi trường Python yêu cầu `Python 3.10`. Danh sách các package chính được khai báo chi tiết tại file `requirements.txt`:
- `transformers`, `torch`, `sentence-transformers` (Chạy mô hình AI)
- `chromadb`, `rank-bm25`, `underthesea` (Xử lý Vector DB và tiếng Việt)
- `unsloth` (Hỗ trợ load Gemma-2-9B lên các GPU VRAM thấp như T4/L4).

### 3.3. Các tệp cấu hình triển khai
Toàn bộ siêu tham số (Hyperparameters), đường dẫn file và cấu hình hệ thống (như Batch size, Device, Thresholds chống ảo giác) được tập trung tại một tệp cấu hình duy nhất:
- 📄 `src/vpl/settings.py`

---

## 4. TÀI LIỆU HƯỚNG DẪN CÀI ĐẶT & VẬN HÀNH (README)

Tài liệu hướng dẫn (Instruction Manual) đã được chúng tôi biên soạn cực kỳ chi tiết, đảm bảo giám khảo có thể tái hiện (reproduce) 100% quá trình ra kết quả từ con số 0.

- 📄 **File Hướng dẫn:** Vui lòng xem tệp `README.md` ở thư mục gốc của mã nguồn.
- **Nội dung bao gồm trong README:**
  - Hướng dẫn tạo môi trường ảo (Conda) và cài đặt thư viện (`pip install`).
  - Lệnh thực thi chạy tự động toàn bộ luồng xử lý (End-to-End Pipeline).
  - Lệnh thực thi chạy từng bước đơn lẻ (Ingest -> Index -> Retrieve -> Generate -> Submit) dành cho môi trường Google Colab.
  - Hướng dẫn sử dụng công cụ tinh chỉnh (Retune Threshold).
