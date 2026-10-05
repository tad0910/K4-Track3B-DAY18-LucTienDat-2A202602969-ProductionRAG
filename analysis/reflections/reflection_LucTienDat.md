# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Lục Tiến Đạt  
**MSSV:** 2A202602969  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code vừa hoàn thành trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng mô hình sentence embedding `all-MiniLM-L6-v2` tính cosine distance giữa các câu liền kề. Với threshold 0.85, văn bản được cắt tại các điểm chuyển ý (topic shifts), bảo toàn ngữ cảnh câu trọn vẹn và tránh hiện tượng cắt ngang câu như fixed-size chunking. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | Áp dụng công thức RRF $RRF\_score(d) = \sum_{m} \frac{1}{k + rank_m(d)}$ với $k=60$. Cơ chế này khắc phục triệt để sự chênh lệch thang điểm (score distribution) giữa BM25 (sparse/lexical, điểm số không bị chặn) và Cosine similarity (dense/semantic, thang [-1, 1]), giúp dung hòa ưu điểm tìm kiếm từ khóa chính xác (mã lỗi, tên riêng tiếng Việt đã segment qua Underthesea) và ngữ nghĩa sâu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Sử dụng mô hình `BAAI/bge-reranker-v2-m3` để rerank Top-20 candidates xuống Top-3/Top-5 kết quả tốt nhất. Điểm khác biệt mấu chốt so với Bi-Encoder là cơ chế Full Cross-Attention (từng token trong Query tương tác trực tiếp với từng token trong Document qua tất cả các lớp Transformer), loại bỏ hoàn toàn các false-positive mà Dense Search hay bị nhầm lẫn. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` & `failure_analysis()` | Đánh giá hệ thống toàn diện qua 4 góc độ: Faithfulness, Answer Relevancy, Context Precision và Context Recall. Phân loại lỗi theo Diagnostic Tree (Retrieval vs Generation failure) giúp định vị chính xác điểm nghẽn của pipeline thay vì chỉ nhìn vào câu trả lời cuối cùng. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Trước khi đưa chunk vào vector store, chunk được gắn thêm ngữ cảnh cha (document context & summary). Phương pháp này triệt tiêu tình trạng chunk độc lập bị thiếu chủ ngữ/bối cảnh (anaphora/pronoun ambiguity), giúp Dense retriever tìm kiếm chính xác ngay cả khi chunk chỉ chứa thông tin chi tiết. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - **Lỗi 1 (Timeout trong Pytest & Check Lab):** `pytest` khi chạy toàn bộ test suite trên CPU Windows bị timeout 120s trong `check_lab.py` do phải tải và khởi tạo lại các mô hình Transformer nặng (`BAAI/bge-m3` ~2.2GB, `BAAI/bge-reranker-v2-m3` ~1.1GB) qua nhiều test cases.
  - **Lỗi 2 (Underthesea word segmentation):** `underthesea.word_tokenize` nối từ ghép bằng dấu gạch dưới (`thông_tin_bảo_mật`), khiến BM25 tokenizer mặc định (`re.findall(r'\b\w+\b')`) tách thành các từ đơn hoặc không khớp với query gốc nếu không xử lý đồng bộ.
  - **Lỗi 3 (API Key 401 khi chạy enrich chunks):** File `.env` chứa `OPENAI_API_KEY=sk-...` (dummy key) khiến OpenAI client cố gắng gửi request và retry nhiều lần trên hàng trăm chunk, gây chậm và báo lỗi xác thực 401. Đồng thời khi RAGAS chạy không có API key hợp lệ thì trả về giá trị `NaN`.

- **Nguyên nhân gốc rễ & Cách debug:**
  - **Cách xử lý Lỗi 1:** Triển khai cơ chế Singleton / Model Caching toàn cục ở cấp module (`_cached_cross_encoder`, `_cached_dense_encoder`, `_semantic_model`). Mỗi mô hình chỉ khởi tạo đúng 1 lần duy nhất trong RAM tiến trình, giảm thời gian chạy test suite từ >120s xuống còn ~40s. Đồng thời điều chỉnh `timeout=300` trong `check_lab.py`.
  - **Cách xử lý Lỗi 2:** Trong hàm `segment_vietnamese`, sau khi chạy `word_tokenize`, thực hiện chuẩn hóa thay thế dấu gạch dưới bằng khoảng trắng (`.replace("_", " ")`) để đồng nhất format chuỗi cho cả BM25 và Dense Search.
  - **Cách xử lý Lỗi 3:** Bổ sung điều kiện kiểm tra trong `config.py` và `m5_enrichment.py`: nếu API key chứa giá trị dummy (`sk-...` hoặc rỗng) thì tự động chuyển sang chế độ offline heuristic rules / regex fallback thay vì kích hoạt HTTP call. Trong `m4_eval.py`, tạo hàm `_safe_float` làm sạch các giá trị `NaN` / `None` thành `0.0` số thực chuẩn.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Hiểu sâu hơn về chi phí tính toán (Latency vs Accuracy trade-off) giữa Bi-Encoder và Cross-Encoder. Cách khắc phục: áp dụng mô hình 2 tầng (Two-stage Retrieval) kinh điển trong Production RAG: tầng 1 dùng Bi-Encoder + BM25 lấy Top-50 (nhanh ~10ms), tầng 2 dùng Cross-Encoder rerank lấy Top-5 (chính xác cao ~80ms).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của cá nhân:

### Project: Trợ lý AI Hỏi đáp Quy trình & Văn bản Nội bộ Doanh nghiệp (Enterprise Policy Q&A Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản (Fixed-size chunking 500 characters, overlap 50, OpenAI `text-embedding-3-small`, tìm kiếm Top-5 bằng Cosine Similarity trên ChromaDB).
- **Vấn đề / Bottlenecks đang gặp:**
  - Chunking cố định làm đứt đoạn các bảng biểu chính sách thưởng phạt, quy trình thủ tục nhiều bước.
  - Tỷ lệ ảo giác (hallucination) cao khi câu trả lời cần tổng hợp thông tin từ nhiều điều khoản.
  - Tìm kiếm các thuật ngữ viết tắt hoặc mã điều khoản nội bộ (ví dụ: `QD-2024-TC`, `MFA`, `SLA`) bằng Vector Search thuần túy thường xuyên bị bỏ sót (Low Retrieval Recall).

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Chuyển sang **Structure-aware Chunking** kết hợp **Hierarchical Chunking** (Parent-Child chunking). Dùng cấu trúc Markdown/Tiêu đề điều khoản (`#`, `##`, `Điều 1:`, `Khoản 2:`) làm ranh giới chia chunk cha (1000 tokens) và chunk con (200 tokens) để bảo toàn bối cảnh toàn vẹn.
2. **Search retrieval:** Áp dụng **Hybrid Search (BM25 + Dense Search BAAI/bge-m3)** kết hợp thuật toán **Reciprocal Rank Fusion (RRF)**. BM25 giải quyết triệt để các mã quy định, tên phòng ban; Dense vector nắm bắt ý định tìm kiếm tự nhiên của nhân viên.
3. **Reranking:** Tích hợp mô hình reranker đa ngữ nhẹ **`BAAI/bge-reranker-v2-m3`** hoặc **`FlashRank`** (ONNX tối ưu) để rerank Top-20 candidates thành Top-5 trước khi đưa vào LLM Context window.
4. **Evaluation:** Thiết lập bộ benchmark 100 câu hỏi test gold-standard, chạy định kỳ với thư viện **RAGAS** theo dõi 4 metrics: Faithfulness (mục tiêu >0.85), Answer Relevancy (>0.85), Context Precision (>0.80), Context Recall (>0.80).
5. **Enrichment:** Triển khai **Contextual Prepend** (gắn tiêu đề văn bản + tóm tắt 2 câu của chương mục trước mỗi chunk) và sinh **Hypothesis Questions (HyQA)** nhằm tối ưu hóa retrieval đối với các câu hỏi ngắn của người dùng.

#### 3. Timeline triển khai
- **Tuần 1:**
  - Tái cấu trúc pipeline dữ liệu: Xây dựng bộ parser tài liệu nội bộ (PDF/DOCX) sang Markdown chuẩn.
  - Áp dụng Structure-aware chunking và Contextual Prepend cho 100 văn bản chính sách đầu tiên.
  - Triển khai Qdrant Vector Store và BM25 index có hỗ trợ Underthesea tiếng Việt.
- **Tuần 2:**
  - Tích hợp Hybrid Search RRF và Cross-Encoder Reranker.
  - Xây dựng test set đánh giá 50 câu hỏi mẫu và chạy baseline RAGAS.
  - So sánh hiệu năng trước và sau cải tiến, tinh chỉnh tham số threshold và số lượng Top-k context truyền cho LLM.
