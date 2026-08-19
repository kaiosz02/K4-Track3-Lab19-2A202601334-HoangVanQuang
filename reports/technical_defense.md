# Thuyết Minh Kỹ Thuật (10 Câu Hỏi Bảo Vệ Kiến Trúc) — Lab 19

**Học viên:** Hoàng Văn Quang (MSSV: 2A202601334)  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  

---

### Câu 1: Tại sao cần Conservative Coreference Resolution trước khi trích xuất Triples?
- **Trả lời:** Trong văn bản tin tức công nghệ, các đại từ nhân xưng ("he", "she", "they") hoặc danh từ chung ("the company", "the startup", "the platform") xuất hiện dày đặc. Nếu trích xuất trực tiếp mà không phân giải, hệ thống sẽ sinh ra các node vô nghĩa như Node("the company"). Ngược lại, nếu phân giải quá mức (aggressive), LLM dễ gán sai chủ ngữ khi câu phức chứa nhiều thực thể. Do đó, áp dụng cơ chế Conservative Coreference (chỉ giải quyết khi antecedent rõ ràng 100% trong cùng chunk) giúp tối đa hóa Precision của đồ thị.

### Câu 2: Ý nghĩa của Schema Allowlist và Guard kiểm soát quan hệ
- **Trả lời:** Đồ thị tri thức doanh nghiệp cần tính nhất quán cao. Schema định nghĩa trước tập nhãn (`ALLOWED_NODE_TYPES = {'Company', 'Person', 'Technology'}`) và quan hệ (`ALLOWED_RELATIONS = {'ACQUIRED', 'DEVELOPED', 'INVESTED_IN', 'FOUNDED', 'WORKED_AT', 'PARTNERED_WITH', 'USES', 'LEADS'}`). Bất kỳ trích xuất nào nằm ngoài allowlist đều bị loại bỏ để tránh làm loãng cấu trúc ontology của đồ thị.

### Câu 3: Kỹ thuật Bulk Ingestion bằng Neo4j `UNWIND`
- **Trả lời:** Việc gửi từng câu lệnh Cypher đơn lẻ gây ra overhead cực lớn về mạng và transaction. Bằng cách gom các bản ghi thành từng batch (1,000 items) và sử dụng cú pháp `UNWIND $rows AS row` kết hợp `MERGE` và Constraint trên `Entity.id`, tốc độ nạp dữ liệu tăng hơn 20 lần và đảm bảo tính nguyên tử (atomic) của transaction.

### Câu 4: Ngưỡng Entity Resolution (Vector Threshold) và Lexical Guard
- **Trả lời:** Ngưỡng cosine similarity được thiết lập ở mức `0.88`. Vector embedding phản ánh ngữ nghĩa nhưng dễ gộp nhầm các thực thể cùng họ (ví dụ `Sam Altman` vs `Steve Altman`). Lexical Guard đóng vai trò lớp kiểm tra ngữ pháp/từ vựng bắt buộc: cấm gộp nếu cùng họ nhưng khác chữ cái đầu của tên, hoặc cấm gộp công ty mẹ với tên sản phẩm con khi tỷ lệ SequenceMatcher < 0.60.

### Câu 5: Cấu trúc Dữ liệu Union-Find (Disjoint Set) trong Canonicalization
- **Trả lời:** Union-Find giúp quản lý các tập hợp thực thể tương đương với độ phức tạp gần như tuyến tính $O(\alpha(N))$. Khi hai alias được xác định là cùng một thực thể, hàm `union` sẽ nhóm chúng lại, sau đó chọn ra tên đại diện (Canonical Name) chuẩn nhất và lưu vết toàn bộ danh sách `aliases` vào thuộc tính của Node.

### Câu 6: Chiến lược Kiểm soát Thực thể Siêu Kết Nối (Super-node Mitigation)
- **Trả lời:** Các thực thể như Microsoft, Google, OpenAI có thể kết nối với hàng nghìn thực thể khác. Khi thực hiện Graph Traversal, nếu không giới hạn, số lượng cạnh sẽ làm tràn context window của LLM. Giải pháp là áp dụng **Temporal Degree Capping**: nếu $degree > 50$, chỉ lấy tối đa 50 cạnh có `published_date` mới nhất và `confidence` cao nhất.

### Câu 7: Ràng buộc Toàn vẹn Nguồn gốc (Provenance Integrity)
- **Trả lời:** Mỗi cạnh quan hệ trong Neo4j bắt buộc phải lưu trữ thuộc tính `source_chunk_id`, `published_date` và đoạn văn bằng chứng `evidence`. Điều này giúp hệ thống truy vết chính xác nguồn gốc thông tin, phục vụ cho việc kiểm toán dữ liệu và loại trừ hoàn toàn ảo giác (Zero Hallucination).

### Câu 8: Cơ chế Hybrid Retrieval (Graph + Vector)
- **Trả lời:** Truy vấn của người dùng trước hết được trích xuất seed entities qua n-gram matching để tìm các node trung tâm trên Neo4j. Đồng thời, câu hỏi được embed để tìm Top-K vector chunks qua FAISS. Ngữ cảnh đồ thị (linearized subgraphs) kết hợp cùng ngữ cảnh văn bản tạo nên một prompt toàn diện vừa có cấu trúc liên kết vừa có chi tiết ngôn ngữ tự nhiên.

### Câu 9: Đánh giá bằng LLM-as-a-Judge trên 3 Trục Độc lập
- **Trả lời:** Thay vì chỉ đo độ tương đồng chuỗi (ROUGE/BLEU), hệ thống sử dụng LLM Judge đánh giá trên 3 tiêu chí chuyên sâu từ 1 đến 5 điểm:
  1. *Comprehensiveness:* Mức độ bao quát đầy đủ thông tin.
  2. *Faithfulness:* Tính trung thực, có căn cứ tuyệt đối từ context.
  3. *Multi-hop Reasoning:* Khả năng xâu chuỗi và suy luận qua nhiều bước quan hệ.

### Câu 10: Kiến trúc Mở rộng (Scalability) khi xử lý Dữ liệu Lớn (350MB+)
- **Trả lời:** Để scale hệ thống:
  - Sử dụng hàng đợi bất đồng bộ (Celery/Kafka) cho tác vụ NER+RE.
  - Sử dụng HNSW Vector Index với kỹ thuật Blocking theo Entity Type để giảm độ phức tạp so khớp thực thể.
  - Phân vùng cộng đồng (Graph Community Detection) và lưu trữ tóm tắt phân tầng để hỗ trợ truy vấn tổng quát toàn đồ thị.
