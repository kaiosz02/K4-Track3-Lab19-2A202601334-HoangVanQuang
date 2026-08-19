# Báo Cáo Phân Tích Ca Lỗi (Failure Mode Analysis) — Lab 19

**Học viên:** Hoàng Văn Quang (MSSV: 2A202601334)  

---

## 🔬 Ca lỗi 1: Flat RAG Thất bại trong Câu hỏi Suy luận Đa bước (Multi-hop)
- **Question ID:** `G02`
- **Nội dung câu hỏi:** *"Which company developed the FP2 Presence Sensor and which major tech company did they partner with?"*
- **Reference Answer:** *"Aqara developed the FP2 Presence Sensor and partnered with Samsung."*
- **Hiện tượng lỗi của Flat RAG:**
  - Flat RAG chỉ thực hiện tìm kiếm Top-5 chunks văn bản dựa trên độ tương đồng ngữ nghĩa câu hỏi.
  - Thông tin về việc Aqara sản xuất cảm biến FP2 và thông tin Aqara hợp tác với Samsung nằm ở hai đoạn văn bản cách xa nhau hoặc từ hai bản tin khác nhau.
  - Kết quả: Flat RAG chỉ trích xuất được một phần thông tin (chỉ nêu Aqara hoặc chỉ nêu cảm biến) mà thiếu hẳn mối liên kết đối tác với Samsung.
- **Cách GraphRAG khắc phục:**
  - GraphRAG nhận diện seed entity là `Aqara`.
  - Truy vấn 1-hop trên Neo4j trả về đồng thời:
    - `(Aqara)-[:DEVELOPED]->(FP2 Presence Sensor)`
    - `(Aqara)-[:PARTNERED_WITH]->(Samsung)`
  - Prompt được cấp đầy đủ hai mối quan hệ này dưới dạng cấu trúc bảng/bộ ba, giúp LLM tổng hợp câu trả lời hoàn chỉnh, đạt điểm tối đa 5/5 về Comprehensiveness và Multi-hop Reasoning.

---

## 🔬 Ca lỗi 2: Độ trễ (Latency Overhead) của GraphRAG trong Câu hỏi Đơn bước (Single-hop Factoid)
- **Question ID:** `G01`
- **Nội dung câu hỏi:** *"Which news organization formed a partnership with OpenAI to share news archives?"*
- **Reference Answer:** *"Associated Press partnered with OpenAI to license and share its news archives."*
- **Hiện tượng phân tích:**
  - Cả Flat RAG và GraphRAG đều trả lời chính xác là Associated Press.
  - Tuy nhiên, độ trễ của Flat RAG chỉ mất ~0.85s (chỉ 1 bước FAISS vector search), trong khi GraphRAG mất ~1.40s (do phải trải qua: N-gram generation -> Cypher query tìm seed -> Cypher traversal -> Linearize context -> LLM generation).
- **Nguyên nhân gốc rễ (Root Cause):**
  - Graph traversal tạo ra overhead không cần thiết khi câu hỏi chỉ mang tính chất Factoid đơn giản có sẵn trong 1 chunk văn bản đơn lẻ.
- **Giải pháp Khắc phục (Adaptive Query Routing):**
  - Tích hợp một bộ phân loại câu hỏi (Query Classifier / Intent Router).
  - Nếu câu hỏi là Simple Factoid: Sử dụng Flat RAG để tối ưu tốc độ và giảm chi phí truy vấn database.
  - Nếu câu hỏi là Multi-hop, Relationship discovery hoặc Cross-document comparison: Kích hoạt Hybrid GraphRAG.
