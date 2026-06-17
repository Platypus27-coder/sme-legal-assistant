# R2AI2026 — Đề Bài & Yêu Cầu Cuộc Thi

> **Tên cuộc thi**: Truy hồi và Hỏi đáp Văn bản Pháp luật Tiếng Việt  
> **Vietnamese Legal Information Retrieval & Question Answering**  
> **Tổ chức**: AIGuru  
> **Leaderboard**: http://leaderboard.aiguru.com.vn/

---

## 1. Bối Cảnh Bài Toán

Doanh nghiệp SME tại Việt Nam thường gặp khó khăn trong việc tra cứu và áp dụng các quy định pháp lý liên quan đến Luật Doanh nghiệp, thuế, lao động, hợp đồng... Trợ lý pháp lý AI được xây dựng nhằm hỗ trợ chủ doanh nghiệp, kế toán, nhân sự tra cứu nhanh các điều luật, hỏi đáp tình huống pháp lý cụ thể và nhận tư vấn sơ bộ dựa trên hệ thống văn bản pháp luật chính thống.

Cuộc thi hướng tới việc xây dựng các hệ thống AI có khả năng:
- **Tìm kiếm điều luật liên quan** (Information Retrieval)
- **Tự động trả lời các câu hỏi pháp lý** (Legal Question Answering)

### Định nghĩa bài toán

**IR**: Cho tập câu hỏi `Q = {q1, q2, ..., qn}` và kho điều luật `A = {a1, a2, ..., an}`, xác định tập con `A' ⊆ A` trong đó mỗi điều luật `ai ∈ A'` được coi là "liên quan" đến câu hỏi tương ứng `q`.

Một điều luật được coi là **"Liên quan"** nếu câu truy vấn có thể được trả lời Có/Không, được suy ra từ ý nghĩa của điều luật đó.

---

## 2. Mục Tiêu Hệ Thống

Các đội thi cần xây dựng hệ thống AI có khả năng:

| # | Mục tiêu | Mô tả |
|---|----------|-------|
| 1 | **Tra cứu pháp lý chính xác** | Tìm kiếm điều khoản trong Luật Doanh nghiệp và các văn bản liên quan đến SME. Ưu tiên khả năng retrieval và grounding chính xác |
| 2 | **Hỏi đáp pháp lý bằng tiếng Việt** | Hiểu ngôn ngữ tự nhiên tiếng Việt, hỏi đáp các tình huống pháp lý thường gặp |
| 3 | **Dẫn nguồn điều luật** | Trích dẫn điều/khoản/văn bản liên quan, hiện thị rõ nguồn tham chiếu, hạn chế trả lời không có căn cứ pháp lý |
| 4 | **Tư vấn sơ bộ & cảnh báo giới hạn** | Đưa ra hướng dẫn pháp lý sơ bộ, nhắc nhở các rủi ro, hiển thị cảnh báo giới hạn AI |
| 5 | **Kiểm soát nội dung sai lệch** | Hạn chế AI sinh ra thông tin sai lệch, tránh bịa điều luật hoặc nguồn tham chiếu không tồn tại |

---

## 3. Mốc Thời Gian

| Ngày | Sự kiện |
|------|---------|
| **03/06/2026** | Ngày khai mạc, phát hành tập dữ liệu kiểm thử |
| **30/06/2026 23:59 UTC+7** | Chính thức đóng cổng hệ thống, deadline nộp bài |
| **05/07/2026** | Công bố kết quả Top 10, tiến vào DemoDay |
| **11/07/2026** | DemoDay, công bố kết quả chung cuộc |

---

## 4. Dữ Liệu

### 4.1 Ban tổ chức cung cấp

- **Test set duy nhất**: Tập câu hỏi pháp lý (2000 câu), dùng làm căn cứ chấm điểm
- **Không có train set / dev set**
- **Bộ đáp án chuẩn**: Ban tổ chức giữ kín, chỉ phục vụ quá trình chấm điểm

### 4.2 Các đội tự thu thập

Các đội được toàn quyền chủ động thu thập:
- Văn bản pháp luật, thông tư, nghị định từ các nguồn chính thống
- Dữ liệu liên quan đến doanh nghiệp SME (quy định thuế, lao động, hợp đồng...)
- Các tập dữ liệu mở (open dataset) phục vụ bài toán Legal NLP
- Mọi nguồn dữ liệu hợp pháp khác mà đội thi có thể tiếp cận

> ⚠️ **Không được sử dụng dữ liệu bên ngoài** trong bất kỳ bước xử lý nào — nghĩa là không gọi API bên ngoài (GPT-4o, Gemini...) để tạo ra dữ liệu huấn luyện hay augment data.

---

## 5. Format Dữ Liệu

### 5.1 Input (test set)

```json
{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
}
```

### 5.2 Output (submission format)

```json
{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào...",
  "answer": "Doanh nghiệp được hỗ trợ khi được thành lập, tổ chức và hoạt động theo pháp luật về doanh nghiệp; đáp ứng tiêu chí doanh nghiệp nhỏ và vừa, gồm số lao động tham gia bảo hiểm xã hội bình quân năm không quá 200 người và đáp ứng một trong hai tiêu chí: tổng nguồn vốn không quá 100 tỷ đồng hoặc tổng doanh thu của năm trước liền kề không quá 300 tỷ đồng...",
  "relevant_docs": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
    "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
  ],
  "relevant_articles": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5",
    "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết...|Điều 5"
  ]
}
```

### 5.3 Quy tắc format tên văn bản (QUAN TRỌNG)

Trường `<tên văn bản>` trong cả `relevant_docs` và `relevant_articles` phải theo công thức:

```
<Loại văn bản> + <Mã văn bản> + <Trích yếu>
```

| Trường | Format | Ví dụ |
|--------|--------|-------|
| `relevant_docs` | `<mã VB>\|<Loại VB> <Mã VB> <Trích yếu>` | `04/2017/QH14\|Luật 04/2017/QH14 Luật Hỗ trợ DNNVV` |
| `relevant_articles` | `<mã VB>\|<Loại VB> <Mã VB> <Trích yếu>\|<Điều X>` | `04/2017/QH14\|Luật 04/2017/QH14 Luật Hỗ trợ DNNVV\|Điều 4` |

---

## 6. Format Nộp Bài

### 6.1 Cấu trúc file

File JSON chứa **mảng** kết quả cho toàn bộ 2000 câu hỏi:

```json
[
  {
    "id": 1,
    "question": "...",
    "answer": "...",
    "relevant_docs": ["..."],
    "relevant_articles": ["..."]
  },
  ...
]
```

### 6.2 Quy trình nộp

```bash
# Linux/macOS
zip submission.zip results.json

# Windows PowerShell
Compress-Archive -Path results.json -DestinationPath submission.zip
```

> ⚠️ **File bắt buộc phải tên `results.json`** — sai tên sẽ không được chấm điểm  
> ⚠️ **`results.json` phải nằm ở gốc ZIP** — không được nằm trong thư mục con  
> ⚠️ **Bài nộp thiếu câu hoặc sai định dạng** sẽ không được đánh giá và không bị tính vào số lần nộp tối đa

### 6.3 Giới hạn nộp bài

| Giai đoạn | Giới hạn |
|-----------|---------|
| **Public Phase** | Tối đa 10 bài/ngày/đội |
| **Private Phase** | Tối đa **5 bài tổng cộng** — chọn rất cẩn thận |

---

## 7. Phương Pháp Đánh Giá

### 7.1 Information Retrieval (IR) — Tự động

**Metric chính: Macro F2 Score**

```
Precision  = avg(số điều luật truy hồi đúng / số điều luật đã truy hồi)
Recall     = avg(số điều luật truy hồi đúng / số điều luật liên quan thực tế)
F2         = (5 × Precision × Recall) / (4 × Precision + Recall)
```

> **F2 trọng Recall gấp 4 lần Precision** → thà bắt thêm hơn bỏ sót

**Cách trích xuất điều luật từ answer**:  
Hệ thống chấm điểm tự động tìm pattern `"Điều X"` trong trường `answer` của bài nộp, sau đó so sánh với `relevant_articles` trong đáp án (định danh đầy đủ dạng `law_id|tên văn bản|Điều X` được chuẩn hóa về `Điều X`).

### 7.2 Question Answering (QA) — Bán tự động

**5 tiêu chí đánh giá**:

| # | Tiêu chí | Phương thức |
|---|----------|------------|
| 1 | **Căn cứ chính xác pháp luật** | Tự động: tỷ lệ câu hỏi có ít nhất 1 điều luật được trích xuất đúng từ câu trả lời |
| 2 | **Tính chính xác nội dung** | LLM-as-Judge: mức độ chính xác so với quy định pháp luật |
| 3 | **Tính đầy đủ & toàn diện** | LLM-as-Judge: câu trả lời có bao quát đầy đủ các khía cạnh không |
| 4 | **Tính thực tiễn** | LLM-as-Judge: có thể áp dụng thực tế trong bối cảnh pháp lý không |
| 5 | **Tính rõ ràng, dễ hiểu** | LLM-as-Judge: diễn đạt rõ ràng cho người đọc không chuyên không |

> ⚠️ Tiêu chí 2-5 hiện đặt giá trị 0.0, sẽ được cập nhật điểm số sau khi ban giám khảo hoàn thành đánh giá.

**Chu kỳ đánh giá QA**: Hàng tuần — đội chọn 1 bài từ danh sách đã nộp và **đẩy (promote)** lên leaderboard. Chỉ bài được promote mới được đưa vào kỳ chấm QA.

---

## 8. Ràng Buộc Mô Hình LLM

| Ràng buộc | Yêu cầu |
|-----------|---------|
| **Kích thước** | < 14B parameters |
| **Thời điểm ra mắt** | Phải được công bố chính thức **trước 01/03/2026** |
| **Giấy phép** | Open-source, trọng số được phép tải xuống và sử dụng tự do cho nghiên cứu |
| **Loại hình** | Không được dùng closed LLM (GPT-4o, Gemini, Claude...) |

> Ban tổ chức có quyền yêu cầu các đội cung cấp thông tin xác nhận mô hình sử dụng.  
> Bài nộp không đáp ứng các ràng buộc trên sẽ bị loại khỏi bảng xếp hạng.

---

## 9. Quy Định Khác

- **Dữ liệu bên ngoài**: Không được sử dụng dữ liệu bên ngoài trong bất kỳ bước xử lý nào
- **Tên đại diện**: Mỗi đội chọn 1 tên người dùng đại diện
- **Bài báo mô tả phương pháp**: Kết quả cuối cùng không được coi là chính thức cho đến khi một bài báo mô tả phương pháp (working notes paper) với mô tả đầy đủ về các phương pháp được nộp
- **Quyền loại bỏ**: Ban tổ chức có toàn quyền loại bất kỳ thí sinh nào có bài nộp không tuân thủ tất cả các yêu cầu

---

## 10. Chiến Lược Tối Ưu F2

Vì F2 trọng Recall gấp 4 lần Precision:

```
F2 = 5PR / (4P + R)
→ Thà trả về thêm điều luật (Precision thấp) hơn bỏ sót (Recall thấp)
→ Khi nghi ngờ: thêm điều luật vào relevant_articles thay vì bỏ qua
```

**Hàm ý thiết kế**:
- Threshold retrieval nên **thấp** (ưu tiên recall)
- `MAX_ARTICLES` nên **cao** hơn tối thiểu cần thiết
- Fallback: nếu không đủ high-confidence articles → lấy top-N
- Answer phải **cite điều luật** vì hệ thống chấm auto-extract từ answer

---

## 11. Checklist Trước Khi Nộp

- [ ] File tên đúng: `results.json`
- [ ] ZIP flat (không thư mục con): `submission.zip → results.json`
- [ ] Đủ 2000 entries (id 1 → 2000)
- [ ] 5 fields bắt buộc: `id`, `question`, `answer`, `relevant_docs`, `relevant_articles`
- [ ] `answer` không rỗng
- [ ] `relevant_docs` format: `<mã VB>|<Loại VB> <Mã VB> <Trích yếu>`
- [ ] `relevant_articles` format: `<mã VB>|<Loại VB> <Mã VB> <Trích yếu>|<Điều X>`
- [ ] Các `Điều X` trong `relevant_articles` xuất hiện trong `answer`
- [ ] `relevant_docs` chứa đầy đủ tài liệu tương ứng với `relevant_articles`
- [ ] Local F2 evaluation trước khi dùng 1 trong 5 lượt Private Phase

---

*Nguồn: Tài liệu cuộc thi R2AI2026 — AIGuru*  
*Cập nhật: 2026-06-16*
