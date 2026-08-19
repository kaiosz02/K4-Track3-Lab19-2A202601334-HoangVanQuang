# Lab 19: Production-Grade GraphRAG vs Flat RAG — Bài tập thực hành

**Thời lượng:** 2 giờ implement + 30 phút reflection & thuyết minh kỹ thuật  
**Tổng điểm:** 100 điểm + 10 điểm bonus  
**File thực thi chính:** [`Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`](Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb)

---

## 🎯 Mục tiêu bài Lab

1. **Hiểu rõ giới hạn của Flat RAG**: Chứng minh qua thực nghiệm các câu hỏi multi-hop và cross-document khiến Vector Search thuần túy bị phân mảnh ngữ cảnh hoặc trích xuất sót thông tin.
2. **Xây dựng Knowledge Graph Extraction Pipeline chuẩn Production**:
   - Tiền xử lý văn bản, khử trùng lặp (Dedup) và phân giải đại từ (Coreference Resolution).
   - Trích xuất Named Entity Recognition (NER) & Relation Extraction (RE) theo strict JSON schema.
   - Phân giải thực thể (Entity Resolution) kết hợp Vector ANN + Lexical Guard + Disjoint-Set Union (Union-Find).
   - Ingestion hiệu năng cao vào Neo4j bằng câu lệnh Cypher `UNWIND` theo batch (không insert từng dòng).
3. **Thiết kế Hybrid GraphRAG Retrieval**:
   - Seed Entity Extraction & Fuzzy Matching fallback.
   - Graph Traversal BFS với cơ chế **Super-node Mitigation** (giảm thiểu bùng nổ đồ thị khi node có bậc > 100).
   - Linearize subgraph thành textual context kèm trích dẫn nguồn gốc (Provenance: `source_chunk_id`, `published_date`, `evidence`).
   - Kết hợp Subgraph Context + Vector Chunks Context.
4. **Đánh giá Benchmark & LLM-as-a-Judge**:
   - Đánh giá trên Golden Dataset phân loại theo 3 nhóm câu hỏi: `factoid`, `multi-hop`, `cross-doc`.
   - Chấm điểm bằng LLM Judge trên 3 tiêu chí: *Comprehensiveness*, *Faithfulness*, *Multi-hop Reasoning*.
   - Đo lường đánh đổi giữa Quality vs Latency vs Token usage.
5. **Thuyết minh & Kiểm soát AI Coding Agent**: Bảo vệ các quyết định kỹ thuật và giải thích các failure modes.

---

## ⏳ Phân bổ thời gian (Timeline)

| Thời gian | Hoạt động | Trọng tâm bàn giao |
|-----------|-----------|-------------------|
| **0:00–0:15** | **Setup & Preprocessing (M1)** | Khởi tạo kết nối, stream dataset từ Hugging Face, exact dedup, text chunking, Coreference Resolution |
| **0:15–0:45** | **Triple Extraction & Ingestion (M2)** | NER + RE extraction, Schema Validation, Cypher `UNWIND` bulk insert vào Neo4j, verify 0 edge thiếu provenance |
| **0:45–1:05** | **Entity Resolution (M3)** | Vector Embedding ANN + Lexical Guard + Union-Find clustering, xuất audit table |
| **1:05–1:30** | **Retrieval Architecture (M4)** | Xây dựng FAISS Flat RAG Index, Seed Extraction, BFS Graph Traversal, Super-node degree cap, Hybrid Answer Generator |
| **1:30–1:45** | **Golden Evaluation & Benchmark (M5)** | Điền gold answers, chạy LLM-as-a-Judge đánh giá, export `graphrag_eval_results.csv` & `graphrag_vs_flatrag_summary.csv` |
| **1:45–2:00** | **Failure Analysis & Bonus** | Kiểm tra Super-node policy, phân tích các ca lỗi (Bottom-N), chạy thử Community Detection hoặc Self-Correction |
| **2:00–2:30** | **Thuyết minh kỹ thuật & Reflection** | Trả lời 10 câu hỏi bảo vệ kiến trúc + Viết báo cáo cá nhân |

---

## 🛡️ Scale Guard (Giới hạn dữ liệu trong giờ Lab)

Để hoàn thành toàn bộ pipeline trong 120 phút mà không gặp lỗi rate-limit hay OOM (Out-of-Memory):
- `LAB_MAX_ARTICLES = 1500` (Tối đa 1500 bài viết được đưa vào xử lý)
- `LAB_MAX_CHUNKS = 3000` (Tối đa 3000 chunk cho Flat RAG Vector Index)
- `EXTRACTION_MAX_CHUNKS = 400` (Tối đa 400 chunk gửi qua LLM để trích xuất Knowledge Graph)
- `CHUNK_WORDS = 220`, `CHUNK_OVERLAP_WORDS = 40`

---

## 🛠️ Chi tiết từng Module

### Module 1: Setup, Ingestion & Coreference Resolution (15 phút)

**Vị trí trong Notebook:** Phần 1 (Cell 1.1 → Cell 1.7)

#### Các bước thực hiện:
1. **Streaming Dataset:** Kết nối stream `HackerNoon/tech-company-news-data-dump` qua `HF_TOKEN` và lưu ra CSV cục bộ (ưu tiên `PRIORITIZE_MB = True`, dừng ở mức ~300MB hoặc số dòng tối đa).
2. **Exact Dedup & Chunking:**
   - Chuẩn hóa text (`norm_space`, NFKC unicode).
   - Khử trùng lặp chính xác bằng SHA-1 hash (`title + text`).
   - Cắt văn bản theo rolling window: 220 từ/chunk, overlap 40 từ.
3. **Conservative Coreference Resolution:**
   - Sử dụng LLM phân giải các đại từ nhân xưng (`it`, `they`, `he`, `she`, `the company`, `the startup`) về tên thực thể đích thực sự.
   - **Quy tắc an toàn (Conservative rule):** CHỈ phân giải khi tiền ngữ (antecedent) xuất hiện rõ ràng trong cùng một chunk; KHÔNG suy diễn hoặc bịa đặt (hallucinate).
   - Nếu mơ hồ (ambiguous) → giữ nguyên văn bản gốc và ghi nhận vào danh sách `unresolved_mentions`.

> [!CAUTION]
> **Cảnh báo Failure Mode:** Phân giải đại từ sai (False Coreference) sẽ dẫn đến việc trích xuất liên kết sai (False Edge) trong đồ thị tri thức, làm suy giảm nghiêm trọng độ tin cậy của GraphRAG.

---

### Module 2: Triple Extraction & Neo4j Bulk Ingestion (30 phút)

**Vị trí trong Notebook:** Phần 2 (Cell 2.1 → Cell 2.4)

#### Schema Knowledge Graph:
- **Node Labels:** `Company`, `Person`, `Technology` (tất cả đều có base label `Entity`).
- **Allowed Relation Types:**
  ```python
  ALLOWED_RELATIONS = {
      "ACQUIRED", "DEVELOPED", "INVESTED_IN", "FOUNDED",
      "WORKED_AT", "PARTNERED_WITH", "USES", "LEADS"
  }
  ```
- **Thuộc tính bắt buộc trên mọi Relationship (Edge Provenance):**
  - `source_chunk_id`: ID của chunk xuất xứ của thông tin.
  - `published_date`: Ngày xuất bản bài viết (hỗ trợ phân tích dòng thời gian / temporal validity).
  - `evidence`: Câu văn trích dẫn làm bằng chứng từ văn bản gốc.
  - `confidence`: Điểm tự tin của mô hình trích xuất (0.0 – 1.0).

#### Kỹ thuật nạp dữ liệu (Bulk Ingestion):
- **Cấm:** Không chạy câu lệnh `CREATE` hoặc `MERGE` đơn lẻ cho từng row (gây nghẽn mạng và latency cao).
- **Yêu cầu:** Sử dụng `UNWIND $rows AS row` với batch size 1000 records.
- **Constraints & Indexes:** Thiết lập unique constraint trên `(n:Entity).id` và index trên `name_norm`.
- **Sanity Check:** Chạy kiểm tra đồ thị:
  ```cypher
  MATCH ()-[r]->()
  WHERE r.source_chunk_id IS NULL OR r.published_date IS NULL
  RETURN count(r) AS invalid_provenance_edges
  ```
  *(Số lượng cạnh thiếu provenance bắt buộc phải bằng 0)*.

---

### Module 3: Entity Resolution & Canonicalization (20 phút)

**Vị trí trong Notebook:** Cell 2.2

Thực thể trong tin tức công nghệ thường xuất hiện dưới nhiều biến thể (ví dụ: *MSFT*, *Microsoft Corp*, *Microsoft Corporation*, *Google LLC*, *Alphabet*).

```
[Raw Entity Mentions] 
        │
        ├── 1. Manual Aliases Map (Ticker & Major Tech)
        │
        ├── 2. Vector Embedding Candidate Search (FAISS ANN, cosine similarity >= 0.90)
        │
        ├── 3. Lexical Guard (Loại bỏ hậu tố Inc, Corp; SequenceMatcher ratio >= 0.72)
        │
        └── 4. Disjoint-Set Union (Union-Find) → Gán Canonical Entity ID
```

#### Tiêu chí vượt qua:
- [ ] Tạo được bảng audit `entity_resolution_audit_df` ghi rõ lý do merge (`MERGE_MANUAL`, `MERGE_VECTOR`) hoặc từ chối (`REJECT_GUARD`).
- [ ] Ngăn chặn được các trường hợp False Merge nguy hiểm (ví dụ: người trùng họ như *Sam Altman* vs *Steve Altman*, hoặc sản phẩm mang tên công ty như *Apple Watch* vs *Apple*).

---

### Module 4: Retrieval Architecture — Flat RAG vs Hybrid GraphRAG (25 phút)

**Vị trí trong Notebook:** Phần 3 (Cell 3.1 → Cell 3.4)

#### 1. Flat RAG Baseline:
- Index toàn bộ `chunks_df` vào FAISS (`IndexFlatIP`) sử dụng embedding `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieve top-k ($k=6$) chunks tương đồng ngữ nghĩa cao nhất.

#### 2. Hybrid GraphRAG Retrieval Flow:
1. **Seed Entity Extraction:** LLM trích xuất danh sách thực thể hạt nhân (Seed Entities) từ câu hỏi người dùng.
2. **Seed Resolution & Fuzzy Matching:** Khớp seed với node trong Neo4j (Exact match trên `name_norm`/`aliases_norm`, fallback bằng vector similarity >= 0.66).
3. **Graph Traversal (BFS):** Khám phá các node lân cận trong bán kính $N$ bước (`max_hops = 2`).
4. **Super-Node Mitigation (Giảm thiểu nút siêu kết nối):**
   - Nếu một node có bậc $degree > 100$ (ví dụ: các công ty lớn như *Google*, *Microsoft* kết nối với hàng nghìn thực thể khác):
   - **Chính sách:** Giới hạn tối đa **50 cạnh** có `published_date` mới nhất (`ORDER BY published_date DESC LIMIT 50`).
   - Tổng số cạnh thu thập trong toàn bộ context bị chặn ở `GLOBAL_EDGE_CAP = 250`.
   - Giới hạn độ dài ngữ cảnh đồ thị: `MAX_GRAPH_CONTEXT_CHARS = 14000`.
5. **Linearize Subgraph (Textualization):** Chuyển đồ thị thành văn bản có cấu trúc kèm trích dẫn:
   ```text
   Microsoft [Company] -INVESTED_IN-> OpenAI [Company] | date=2023-01-23 | chunk=art_01::c0002 | evidence=Microsoft invested $10 billion...
   ```
6. **Hybrid Context Generation:** Kết hợp cả `=== GRAPH ===` context và `=== VECTOR ===` context đưa vào Prompt trả lời.

---

### Module 5: Golden Dataset Evaluation & LLM-as-a-Judge (15 phút)

**Vị trí trong Notebook:** Phần 4 (Cell 4.1 → Cell 4.4)

#### Golden Dataset Schema:
Tối thiểu 5 câu hỏi trải dài trên 3 nhóm:
- **`factoid`**: Câu hỏi tra cứu 1 sự thật duy nhất (Single-hop).
- **`multi-hop`**: Câu hỏi đòi hỏi suy luận nối chuỗi giữa ≥ 2 quan hệ (ví dụ: *Startup nào được sáng lập bởi cựu nhân viên Microsoft và sau đó nhận vốn từ Google?*).
- **`cross-doc`**: Câu hỏi so sánh xu hướng/dòng thời gian từ nhiều bài báo khác nhau.

#### LLM-as-a-Judge Rubric (Thang điểm 1–5):
1. **Comprehensiveness (Độ đầy đủ):** Câu trả lời có bao quát hết tất cả các khía cạnh/thực thể được hỏi không?
2. **Faithfulness (Độ trung thực):** Mọi luận điểm trong câu trả lời có được chứng minh bởi context cung cấp không? (Trừ điểm nặng---

## 📝 Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật (30 phút)

Học viên mở file `reports/lab_report.md` và hoàn thiện đầy đủ 2 phần:

### Phần 1: Thuyết minh Kỹ thuật & Phân tích Ca lỗi (10 câu hỏi)
1. **Coreference Resolution:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu mà cơ chế Coreference Resolution phân giải sai. Hậu quả của nó đối với Knowledge Graph là gì?
2. **Entity Resolution Threshold & Lexical Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp và giải thích lý do.
3. **Super-node Analysis:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy 50 cạnh mới nhất mang lại ưu điểm và rủi ro gì?
4. **Bảng so sánh Benchmark & 2 Ca lỗi Điển hình:** Điền bảng so sánh Quality vs Latency vs Token usage. Phân tích 1 ca Flat RAG thất bại (GraphRAG thành công) và 1 ca GraphRAG thất bại/khó khăn.
5. **Trade-offs, Agent Control & Scale 350MB:** So sánh đánh đổi chi phí/thời gian; nêu đề xuất của AI Coding Agent mà bạn từ chối; giải pháp kiến trúc khi scale 350MB.

### Phần 2: Suy ngẫm Cá nhân & Kế hoạch Đồ án (Reflection & Action Plan)
1. **Mapping bài giảng:** Điền bảng mapping 5 modules bài giảng vào các hàm code tương ứng.
2. **Debugging & Bài học:** Mô tả lỗi kỹ thuật khó nhất và cách xử lý.
3. **Kế hoạch Đồ án:** Đánh giá bài toán thực tế của bạn có cần GraphRAG không, thiết kế sơ bộ Node/Relation và chiến lược xử lý dữ liệu.

---

## 📦 Deliverables (Bài nộp)

Học viên commit và push lên GitHub repo cá nhân:

```
Day19-Track3-GraphRAG/
├── Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb   # ★ Notebook đã chạy đầy đủ output các cell
├── outputs/
│   ├── graphrag_eval_results.csv                         # ★ Kết quả benchmark chi tiết từng câu
│   └── graphrag_vs_flatrag_summary.csv                   # ★ Bảng so sánh tổng hợp Flat vs Graph
└── reports/
    └── lab_report.md                                     # ★ Báo cáo hoàn chỉnh (Thuyết minh + Reflection)
```

---

## 📋 Submission Checklist (Kiểm tra trước khi nộp)

- [ ] Kết nối Neo4j thành công và schema đã được khởi tạo.
- [ ] Chạy hoàn tất toàn bộ các cell trong Notebook `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` không có cell nào bị crash.
- [ ] Kiểm tra đồ thị: Đảm bảo **0 cạnh bị thiếu `source_chunk_id` hoặc `published_date`**.
- [ ] Bảng `entity_resolution_audit_df` có ít nhất 10+ dòng audit minh bạch.
- [ ] Đã kiểm tra cơ chế Super-node cap (node bậc $> 100$ chỉ lấy $\le 50$ cạnh).
- [ ] Đã điền đầy đủ `reference_answer` cho các câu hỏi trong Golden Dataset (`data/golden_dataset.csv`).
- [ ] Chạy hoàn tất LLM Judge và xuất đủ 2 file CSV vào thư mục `outputs/`.
- [ ] Đã hoàn thành đầy đủ cả 2 phần trong file báo cáo `reports/lab_report.md`.
- [ ] Push toàn bộ repo lên GitHub và nộp đường link.
 (Action Plan)
- Đồ án của bạn có cần đến GraphRAG không, hay Flat RAG / Hybrid RAG là đủ?
- Nếu áp dụng GraphRAG, cấu trúc Node/Relation của bài toán bạn là gì?
- Chiến lược giải quyết Entity Resolution và Super-node trong bài toán cụ thể của bạn.

## 📦 Deliverables (Cấu trúc bài nộp)

Học viên commit và push lên GitHub repo cá nhân:

```
Day19-Track3-GraphRAG/
├── Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb   # ★ Notebook đã chạy đầy đủ output các cell
├── outputs/
│   ├── graphrag_eval_results.csv                         # ★ Kết quả benchmark chi tiết từng câu
│   └── graphrag_vs_flatrag_summary.csv                   # ★ Bảng so sánh tổng hợp Flat vs Graph
└── reports/
    ├── technical_defense.md                              # ★ 10 câu trả lời thuyết minh kỹ thuật
    ├── failure_analysis.md                               # ★ Phân tích ca lỗi Flat RAG vs GraphRAG
    └── reflection_[HọTên].md                             # ★ Mapping bài giảng + Action plan đồ án
```

---

## 📋 Submission Checklist (Kiểm tra trước khi nộp)

- [ ] Kết nối Neo4j thành công và schema đã được khởi tạo.
- [ ] Chạy hoàn tất toàn bộ các cell trong Notebook `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` không có cell nào bị crash.
- [ ] Kiểm tra đồ thị: Đảm bảo **0 cạnh bị thiếu `source_chunk_id` hoặc `published_date`**.
- [ ] Bảng `entity_resolution_audit_df` có ít nhất 10+ dòng audit minh bạch.
- [ ] Đã kiểm tra cơ chế Super-node cap (node bậc $> 100$ chỉ lấy $\le 50$ cạnh).
- [ ] Đã điền đầy đủ `reference_answer` cho các câu hỏi trong Golden Dataset (`data/golden_dataset.csv`).
- [ ] Chạy hoàn tất LLM Judge và xuất đủ 2 file CSV vào thư mục `outputs/`.
- [ ] Đã trả lời đầy đủ 10 câu hỏi trong `reports/technical_defense.md`.
- [ ] Đã viết báo cáo phân tích ca lỗi trong `reports/failure_analysis.md`.
- [ ] Đã viết reflection cá nhân trong `reports/reflection_[HọTên].md`.
- [ ] Push toàn bộ repo lên GitHub và nộp đường link.
