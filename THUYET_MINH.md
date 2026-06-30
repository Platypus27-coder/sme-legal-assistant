# BẢN THUYẾT MINH SẢN PHẨM & TÀI LIỆU KIỂM THỬ
**Đội thi:** impact  
**Dự án:** Hệ thống RAG Trợ lý Pháp lý Doanh nghiệp Nhỏ và Vừa (SME Legal Assistant)

Tài liệu này được biên soạn nhằm đáp ứng đầy đủ các tiêu chí phục vụ Giai đoạn Kiểm thử riêng (01/07/2026 - 03/07/2026) của Ban Tổ chức.

---

## 1. TÀI LIỆU MÔ TẢ DỮ LIỆU (DATA)

### 1.1. Nguồn gốc và Phạm vi sử dụng
- **Nguồn gốc:** Hệ thống sử dụng 100% dữ liệu do Ban tổ chức cung cấp phục vụ riêng cho cuộc thi. Không sử dụng dữ liệu bên ngoài.
- **Thành phần:**
  - `phapdien.jsonl`: Dữ liệu Pháp điển chứa các quy phạm pháp luật, sử dụng làm cơ sở tri thức (Knowledge Base) để tìm kiếm (Retrieval).
  - `anle.jsonl`: Dữ liệu Án lệ, được gộp chung vào Knowledge Base để LLM có thể tham chiếu thực tiễn xét xử.
  - `R2AIStage1DATA.json`: Tập dữ liệu kiểm thử gồm các câu hỏi đầu vào (Input) để hệ thống xử lý sinh ra kết quả cuối cùng.

### 1.2. Cấu trúc và Định dạng dữ liệu
- **Cấu trúc Thư mục:** Tất cả các file dữ liệu thô nói trên phải được đặt tại thư mục `data/` nằm ở thư mục gốc của mã nguồn.
- **Định dạng file:** Dữ liệu có định dạng `.json` và `.jsonl`.
- **Ý nghĩa các trường chính (Tiền xử lý):** Trong quá trình chạy hệ thống (bước Ingest), dữ liệu thô sẽ được tự động băm nhỏ (chunking) theo từng "Điều luật" thành định dạng chuẩn với các trường Metadata:
  - `doc_id`: Số hiệu văn bản.
  - `article_number`: Số hiệu Điều luật.
  - `formatted_doc`: Định dạng ép buộc `[mã VB]|[Loại VB] [Mã VB] [Trích yếu]`.
  - `formatted_article`: Định dạng ép buộc `[mã VB]|[Loại VB] [Mã VB] [Trích yếu]|[Điều X]`.

### 1.3. Đường dẫn chia sẻ và Hướng dẫn sử dụng
- **🔗 Đường dẫn tải dữ liệu (Google Drive):** `https://drive.google.com/drive/u/0/folders/1Gq2JFetYk7VXH5EtSMQMu1V4kFesVorr`
- **Hướng dẫn tải và sử dụng:** Ban Giám khảo truy cập đường link trên để tải file nén `.zip` chứa dữ liệu. Sau khi tải về, vui lòng giải nén và copy toàn bộ các file `.json`/`.jsonl` vào thư mục `data/` của dự án. Hệ thống sẽ tự động đọc dữ liệu từ thư mục này.

---

## 2. CHECKPOINT VÀ MÔ HÌNH SỬ DỤNG (MODEL)

Dự án tiếp cận theo hướng sử dụng Mô hình Ngôn ngữ Lớn mã nguồn mở kết hợp kỹ thuật truy xuất RAG siêu phân luồng (Hybrid Search), do đó **không tiến hành Fine-tuning và KHÔNG SẢN SINH CHECKPOINT CỤC BỘ**.

### 2.1. Thông tin Kiến trúc Mô hình
Hệ thống sử dụng bộ 3 mô hình mã nguồn mở nguyên bản:
1. **Mô hình Sinh (LLM Generation):** `unsloth/gemma-2-9b-it-bnb-4bit`
   - **Kiến trúc:** Dựa trên cấu trúc Gemma-2 (9 Tỷ tham số) của Google.
   - **Thư viện tối ưu:** Sử dụng thư viện `Unsloth` và `BitsAndBytes` để áp dụng Lượng tử hóa 4-bit (4-bit Quantization), giúp mô hình chạy mượt mà trên các GPU có bộ nhớ thấp (15GB VRAM) mà không làm suy giảm chất lượng suy luận.
2. **Mô hình Nhúng Vector (Dense Embedding):** `mainguyen9/vietlegal-e5`
   - Dùng để chuyển đổi văn bản pháp lý thành Vector đa chiều, tối ưu ngữ nghĩa tiếng Việt.
3. **Mô hình Chấm điểm lại (Cross-Encoder Reranker):** `BAAI/bge-reranker-v2-m3`
   - Dùng để chấm điểm (Sigmoid score) và sắp xếp lại kết quả, loại bỏ các kết quả ảo (Anti-hallucination).

### 2.2. Đường dẫn Checkpoint và Hướng dẫn tải
- Do sử dụng bản quyền mã nguồn mở, toàn bộ trọng số (Checkpoint) của 3 mô hình trên được lưu trữ trực tiếp trên máy chủ của HuggingFace.
- **🔗 Đường dẫn mô hình:**
  - https://huggingface.co/unsloth/gemma-2-9b-it-bnb-4bit
  - https://huggingface.co/BAAI/bge-reranker-v2-m3
  - https://huggingface.co/mainguyen9/vietlegal-e5
- **Hướng dẫn cài đặt:** Ban giám khảo **KHÔNG CẦN TẢI THỦ CÔNG** hay giải nén bất kỳ Checkpoint nào. Khi thực thi hệ thống bằng lệnh `python run.py pipeline`, mã nguồn (thư viện `transformers`) sẽ tự động tải các Checkpoint này về Cache của máy chủ ảo và nạp thẳng vào VRAM.

---

## 3. MÃ NGUỒN VÀ THƯ VIỆN ĐÍNH KÈM (SOURCE CODE)

### 3.1. Tổ chức cấu trúc Mã nguồn
Mã nguồn được chúng tôi tổ chức theo nguyên lý Modular hướng đối tượng (OOP), chia thành 5 khâu xử lý riêng biệt (Pipeline 5 bước) để Ban giám khảo dễ dàng rà soát quy trình:
- `src/vpl/corpus/`: Tiền xử lý dữ liệu (Chunking, làm sạch văn bản).
- `src/vpl/store/`: Quá trình Vector hóa và lưu trữ (Tạo từ điển BM25 và ChromaDB).
- `src/vpl/search/`: Quá trình Chạy mô hình truy xuất và chấm điểm (Hybrid Retriever + BGE-Reranker). Sinh ra `artifacts/cache/retrieval.db`.
- `src/vpl/answer/`: Quá trình Chạy mô hình LLM để sinh văn bản (Generation) và Hậu xử lý (Lọc ảo giác, chèn trích dẫn). Sinh ra file kết quả đầu ra.
- `src/vpl/submit.py`: Quá trình chuẩn hóa định dạng và nén ra file `submission.zip` cuối cùng.

### 3.2. Cấu hình và Siêu tham số
- Toàn bộ tham số cấu hình (Đường dẫn, Tên Model, Threshold, Batch size) được tập hợp duy nhất tại file: `src/vpl/settings.py` để Ban Giám khảo thuận tiện kiểm tra.

### 3.3. Danh sách Thư viện (Dependencies)
- **Môi trường:** Python 3.10
- Toàn bộ danh sách thư viện (kèm phiên bản chuẩn mực) được cung cấp trong file `requirements.txt` nằm ở thư mục gốc. Bao gồm các thư viện lớn như: `transformers`, `torch`, `unsloth`, `chromadb`, `rank-bm25`.

---

## 4. TÀI LIỆU HƯỚNG DẪN CHẠY LẠI (README.md)

Để đảm bảo Giám khảo có thể tái hiện (reproduce) và vận hành hệ thống một cách độc lập từ A-Z, chúng tôi đã biên soạn một bản **README.md** cực kỳ tỉ mỉ ngay trong thư mục gốc.

**Bản README.md đã đáp ứng 100% các tiêu chí yêu cầu:**
- ✅ Liệt kê chi tiết cấu trúc thư mục.
- ✅ Cung cấp đường dẫn chia sẻ Data và Model công khai.
- ✅ Hướng dẫn từng lệnh `pip install` để thiết lập môi trường.
- ✅ Hướng dẫn thứ tự các câu lệnh thực thi bằng CLI (ví dụ: `python run.py ingest`, `python run.py index`...) để đảm bảo tái lập chính xác file kết quả cuối cùng.
- ✅ Hướng dẫn tham số chạy nếu máy chủ của Ban Giám khảo bị thiếu VRAM.

Kính mời Ban giám khảo xem chi tiết tại tệp `README.md` của mã nguồn.
