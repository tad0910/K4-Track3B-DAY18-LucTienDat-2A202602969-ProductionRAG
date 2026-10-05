# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lục Tiến Đạt  
**MSSV:** 2A202602969  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|:-------------:|:----------:|:--:|
| Faithfulness | 0.0000 | 0.7000 | +0.7000 |
| Answer Relevancy | 0.0000 | 0.1302 | +0.1302 |
| Context Precision | 0.0000 | 0.9583 | +0.9583 |
| Context Recall | 0.0000 | 0.7833 | +0.7833 |

*Ghi chú: Điểm số Production thực tế đạt ngưỡng $\ge 0.70$ trên 3 tiêu chí cốt lõi (Faithfulness 0.70, Context Precision 0.9583, Context Recall 0.7833) nhờ sự kết hợp giữa Hierarchical Chunking, Hybrid Search (BM25 + Dense) và Cross-Encoder Reranker. Riêng chỉ số Answer Relevancy (0.1302) bị sụt giảm do giới hạn ngắt rate limit 50 req/ngày của OpenRouter Free tier khi gọi đánh giá batch 80 jobs.*

---

## Bottom-5 Failures

### #1
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Trích từ tài liệu `nghi_phep_nam_v2023.md`: Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi 5 năm làm việc liên tục...
- **Worst metric:** Context Precision
- **Error Tree:** Output sai → Context sai (lấy nhầm file cũ `nghi_phep_nam_v2023.md`) → Query không chỉ định năm cụ thể → Retrieval bị nhiễu do xung đột phiên bản.
- **Root cause:** Trong kho dữ liệu tồn tại 2 văn bản chính sách cùng chủ đề: bản cũ 2023 và bản mới 2024. Cả hai đều chứa từ khóa trùng khớp cao, nhưng hệ thống chưa có cơ chế lọc bỏ các tài liệu đã hết hiệu lực.
- **Suggested fix:** Bổ sung metadata `version: "2024"` và `status: "active" | "deprecated"`. Sử dụng Metadata Filtering ở tầng Search để loại trừ các chính sách cũ, hoặc bổ sung Query Rewrite để tự động chèn ngữ cảnh "năm hiện hành" vào câu hỏi.

---

### #2
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Got:** Trích xuất đúng `mat_khau_v2.md` (12 ký tự), tuy nhiên tài liệu `mat_khau_v1.md` vẫn lọt vào Top 5 kết quả tìm kiếm sơ bộ.
- **Worst metric:** Context Precision
- **Error Tree:** Output đúng → Context chứa cả chunk cũ và mới → Cross-Encoder đã ưu tiên v2 nhưng khoảng cách điểm số giữa v1 và v2 còn sít sao.
- **Root cause:** Cả 2 tài liệu đều có mật độ từ khóa "mật khẩu tối thiểu ký tự" tương đương nhau.
- **Suggested fix:** Áp dụng kỹ thuật Contextual Prepend (Module 5) để ghi rõ header *"Chính sách này đã hết hiệu lực từ 01/07/2024"* lên chunk của v1, giúp Cross-Encoder nhận diện và hạ điểm sâu hơn.

---

### #3
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** KHÔNG. Nhân viên thử việc KHÔNG được nghỉ phép năm. Nếu cần nghỉ, phải xin nghỉ không lương và được trưởng phòng phê duyệt.
- **Got:** Chunk `thu_viec.md` được truy xuất chính xác: *"Nhân viên thử việc KHÔNG được nghỉ phép năm"*.
- **Worst metric:** Faithfulness (nếu LLM tóm tắt không nhấn mạnh chữ KHÔNG)
- **Error Tree:** Context đúng → Output LLM cần đảm bảo không làm nhẹ đi tính bắt buộc của quy định.
- **Root cause:** Các câu hỏi dạng câu phủ định hoặc cấm đoán đòi hỏi system prompt phải chỉ thị nghiêm ngặt không được suy diễn mềm mỏng.
- **Suggested fix:** Cải thiện System Prompt: *"Nếu tài liệu nêu rõ điều cấm hoặc không áp dụng, phải trả lời rõ ràng dứt khoát bằng chữ KHÔNG trước tiên."*

---

### #4
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG. Nhân viên tuyệt đối không được tự ý xử lý malware. Phải báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline CNTT.
- **Got:** Trích xuất `bao_mat_su_co.md`: *"Tuyệt đối không tự ý xử lý malware..."*.
- **Worst metric:** Answer Relevancy
- **Error Tree:** Context đúng → Output trả lời đúng trọng tâm quy trình xử lý sự cố.
- **Root cause:** Khi câu hỏi chứa tình huống giả định ("có nên không"), nếu chỉ trích xuất đoạn văn thuần túy thì chưa nhấn mạnh đủ thời hạn "trong vòng 1 giờ".
- **Suggested fix:** Sử dụng HyQA (Hypothesis Question Answering) ở Module 5 để sinh trước câu hỏi: *"Nhân viên có được tự xử lý mã độc không?"* và liên kết với chunk này.

---

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 18 ngày (15 ngày cơ bản + 1 ngày cho mỗi 3 năm = 15 + 3 = 18 ngày).
- **Got:** Truy xuất được quy định v2024 nhưng LLM cần thực hiện phép tính số học (15 + 3).
- **Worst metric:** Faithfulness / Reasoning
- **Error Tree:** Context đúng → Output đòi hỏi suy luận số học đa bước.
- **Root cause:** RAG chỉ truy xuất dữ kiện, bản thân dữ kiện không ghi sẵn con số "18 ngày cho 9 năm thâm niên", cần LLM thực hiện phép tính toán.
- **Suggested fix:** Bổ sung Chain-of-Thought (CoT) prompting vào phần LLM Answer generation để LLM trình bày từng bước tính toán rõ ràng trước khi đưa ra kết luận.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"*

**Error Tree walkthrough:**
1. **Output đúng?** $\rightarrow$ Chưa hoàn toàn chuẩn xác nếu dùng Naive RAG (do trích xuất nhầm chính sách 2023 nói 5 năm).
2. **Context đúng?** $\rightarrow$ Chưa chuẩn: Lấy nhầm tài liệu `nghi_phep_nam_v2023.md` thay vì `nghi_phep_nam_v2024.md`.
3. **Query rewrite OK?** $\rightarrow$ Query người dùng không ghi rõ năm 2024, khiến Dense Search và BM25 đều coi tài liệu 2023 là ứng viên hợp lệ.
4. **Fix ở bước:** Tầng **Chunking & Enrichment (M1 & M5)** và **Tầng Search Filtering (M2)**.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Temporal Filtering / Metadata Filtering:** Thêm trường `is_effective: bool` hoặc `valid_until: date`. Khi truy vấn, tự động inject bộ lọc `valid_until >= now()`.
2. **Query Expansion & Disambiguation:** Dùng LLM viết lại câu hỏi: *"Thâm niên bao nhiêu năm thì được cộng thêm ngày phép theo quy định mới nhất hiện hành?"* trước khi đưa vào Hybrid Search.
