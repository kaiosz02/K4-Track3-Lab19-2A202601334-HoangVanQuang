# Báo Cáo Reflection & Kế Hoạch Đồ Án — Lab 19

**Học viên:** Hoàng Văn Quang (MSSV: 2A202601334)  
**Khóa học:** AICB-K4 · Track 3: GraphRAG  

---

## 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|:--------------------------|:-----------------|:-----------------------|:----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | Giúp liên kết đúng thực thể đại từ mà không sinh ảo giác facts mới. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Giữ đồ thị chuẩn hóa, loại bỏ các quan hệ rác hoặc nhãn không xác định. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | Dùng `UNWIND $rows` nạp theo lô 1,000 bản ghi, tốc độ nạp nhanh gấp 20 lần so với insert từng dòng. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UnionFind` | Khử trùng lặp tên thực thể (canonicalization) đồng thời lưu danh sách `aliases`. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `recent_edges()` | Cắt tỉa cạnh quá mức tại các node trung tâm, bảo vệ Context Window của LLM. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `run_evaluation()` | Đánh giá khách quan trên thang điểm 1-5 với đầy đủ 3 tiêu chí cốt lõi. |

---

## 2. Quá trình Debugging & Bài học Rút ra
- **Thách thức lớn nhất:** Vấn đề Rate Limit (429) và Token Per Day (TPD) khi chạy batch liên tục trên Groq API, cùng với việc xử lý format JSON không đồng nhất giữa các LLM Provider.
- **Giải pháp:** 
  - Xây dựng cơ chế **Multi-Model Fallback Resilience**: hệ thống tự động fallback qua model dự phòng khi model chính chạm ngưỡng rate limit.
  - Viết wrapper `parse_json_object` có khả năng bóc tách JSON an toàn từ phản hồi tự do của LLM.

---

## 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án:** Hệ thống Trợ lý Pháp lý & Phân tích Rủi ro Doanh nghiệp dựa trên GraphRAG.
- **Tại sao cần GraphRAG:**
  - Văn bản luật có tính dẫn chiếu và phân cấp sâu (Luật -> Nghị định -> Thông tư -> Điều khoản).
  - Flat RAG chỉ tìm theo từ khóa/ngữ nghĩa nên dễ lấy phải điều khoản đã hết hiệu lực hoặc không liên kết được quy định sửa đổi bổ sung.
  - GraphRAG cho phép tạo các quan hệ `[:AMENDS]`, `[:SUPERSEDES]`, `[:REFERENCES]` để duyệt chính xác luồng hiệu lực pháp luật.
- **Cấu trúc Ontology dự kiến:**
  - *Nodes:* `LegalDocument`, `Clause`, `Company`, `Person`, `Domain`.
  - *Edges:* `AMENDS`, `REFERENCES`, `SUPERSEDES`, `OWNED_BY`, `OPERATES_IN`.
- **Chiến lược xử lý Dữ liệu lớn:**
  - Dùng Mã số thuế (MST) / Số hiệu văn bản pháp luật làm mã định danh duy nhất (Unique ID).
  - Áp dụng Super-node cap cho các điều khoản định nghĩa chung.
