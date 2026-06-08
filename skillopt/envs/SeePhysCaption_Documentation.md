# SeePhysCaption - Session Summary & Documentation

Tài liệu này ghi lại chi tiết các thay đổi, cơ chế hoạt động, luồng nạp dữ liệu, gọi LLM và cơ chế Logging của môi trường huấn luyện `SeePhysCaption` vừa được xây dựng.

## 1. Các file đã tạo/chỉnh sửa và Vai trò

1. **`configs/SeePhysCaption/default.yaml`**: File cấu hình chính. Chỉ định load 600 cặp câu từ 3 level (`level2`, `level3`, `level4`), mục tiêu là `level1`. Không đặt timeout cứng để tránh ngắt kết nối với LLM. Dùng model `qwen`.
2. **`skillopt/envs/SeePhysCaption/dataloader.py`**: Chịu trách nhiệm load dữ liệu. Đã tinh chỉnh để có thể nhận chuỗi `input_levels = "level2,level3,level4"`, đọc đồng thời từ nhiều thư mục và map với `level1` thông qua `question_id`.
3. **`skillopt/envs/SeePhysCaption/rollout.py`**: Quản lý bước sinh Caption. Tại đây, hệ thống ráp ảnh và văn bản khuyết thiếu thành Prompt gửi lên LLM Target, sau đó nhận về văn bản do LLM sinh ra và đẩy sang Evaluator.
4. **`skillopt/envs/SeePhysCaption/hybrid_evaluator.py`**: Trái tim của quá trình tự động chấm điểm. Thực hiện tính toán ROUGE, BLEU, và dùng Regex rút trích số lượng vật lý. Sau đó, nó tổng hợp thành một báo cáo và gọi LLM Judge để đưa ra phán quyết cuối cùng.
5. **`skillopt/envs/SeePhysCaption/evaluator.py`**: Wrapper kết nối kết quả của `hybrid_evaluator.py` vào chuẩn của hệ thống SkillOpt.
6. **`skillopt/envs/SeePhysCaption/skills/initial.md`**: File chứa đoạn Prompt ban đầu (System Instruction) dành cho LLM, dặn dò các nguyên tắc nền tảng (không đoán bừa, đọc kỹ trục toạ độ...).

---

## 2. Cấu hình LLM và Input qua từng bước

### Bước 1: Rollout (Sinh Caption)
- **Model Role**: Target Model (Qwen).
- **Cấu hình**: 
  - `max_completion_tokens=32000` (Cho phép gen văn bản dài).
  - `enable_thinking=True` (Bật tính năng Reasoning / Chain of Thought).
  - `timeout=None` (Đợi model suy nghĩ không giới hạn thời gian).
- **Input LLM nhận được**:
  - `System`: Nội dung lấy từ file skill (`skills/initial.md` hoặc các bản skill được update).
  - `User`: Chứa văn bản bị khuyết (Input Problem) + Câu lệnh yêu cầu bổ sung thông tin ("Analyze the provided images...") + Mảng ảnh đính kèm dạng Base64.

### Bước 2: Evaluate (Chấm điểm - LLM Judge)
- **Model Role**: Judge Model (Qwen - Nằm trong `hybrid_evaluator.py`).
- **Cấu hình**:
  - `max_completion_tokens=1024`.
  - `enable_thinking=False` (Chỉ cần đánh giá nhanh, không cần CoT dài dòng để tiết kiệm thời gian).
- **Input LLM nhận được**:
  - `System`: Định nghĩa vai trò là Giám khảo khắt khe, yêu cầu trả về định dạng JSON thuần.
  - `User`: Chứa văn bản Ground Truth, Generated Caption do bước 1 sinh ra, các chỉ số thống kê (ROUGE, BLEU) và đặc biệt là **danh sách các con số bị missing** mà hệ thống dò ra được. LLM sẽ dùng thông tin này để quyết định có đánh trượt (hard=0) hay không và ghi lý do chi tiết.

### Bước 3: Reflect / Optimize (Cải thiện Skill)
*(Bước này do lõi của SkillOpt tự động chạy)*
- **Model Role**: Optimizer Model (Qwen).
- **Cấu hình**: Mặc định theo SkillOpt.
- **Input LLM nhận được**:
  - `System`: SkillOpt system prompt.
  - `User`: Chứa tổng hợp các mẫu sai (gồm Câu hỏi, Ảnh, Câu trả lời sai và **Lý do sai - Reason do Judge viết ở Bước 2**). LLM Optimizer sẽ đọc lý do này để hiểu vì sao model Target làm sai và viết ra đề xuất sửa file Skill.

---

## 3. Hệ thống Logging (Lưu vết)

Toàn bộ quá trình chạy sẽ được lưu vào một thư mục chung có dạng:
`/media/hung/DATA/SkillOpt/outputs/SeePhysCaption_{timestamp}/`

Trong đó, log sẽ được chia làm 2 cụm chính:

**A. Log của từng Task (Từng câu hỏi) - Sinh và Chấm điểm**
Nằm sâu trong: `steps/step_XXXX/batch_Y/rollout/predictions/{item_id}/`
- `target_api_request.json`: Payload chứa Prompt và Base64 ảnh vừa gửi lên LLM Rollout.
- `target_api_response.json`: Phản hồi thô kèm số lượng Token đã dùng.
- `judge_debug.json`: Log toàn bộ quá trình Judge (đầu vào, báo cáo điểm mềm, phán quyết JSON của LLM Judge).

**B. Log của quá trình Optimizer (Reflect / Phân tích rút kinh nghiệm)**
Nằm trực tiếp trong: `steps/step_XXXX/`
- `optimizer_api_request_analyst_*.json`: Log chứa prompt (gộp nhiều câu sai) nhờ LLM phân tích.
- `optimizer_api_response_analyst_*.json`: Phản hồi của LLM gồm các Patch đề xuất.
- `optimizer_api_request_aggregate_*.json`: Log quá trình tổng hợp các Patch.
- `candidate_skill.md`: Bản nháp của file Skill sau khi được sửa đổi.

---

## 4. Các lệnh thực thi (Commands)

Đứng ở thư mục gốc của dự án (`/media/hung/DATA/SkillOpt`), chạy các lệnh sau:

**1. Chạy Dry Run (Test luồng với dữ liệu nhỏ)**
Lệnh này giới hạn chỉ lấy 10 câu để chạy 1 vòng lặp (1 Epoch) giúp kiểm tra xem toàn bộ pipeline, API calls và logs có tạo ra đúng như dự kiến hay không trước khi train thật.
```bash
python -m skillopt.main --config configs/SeePhysCaption/default.yaml env.limit=10 train.num_epochs=1 train.batch_size=2
```

**2. Chạy Full Training (Toàn bộ 600 cặp câu)**
Lệnh này sẽ huấn luyện toàn diện 5 Epochs với batch size là 40 (như đã cấu hình trong yaml).
```bash
python -m skillopt.main --config configs/SeePhysCaption/default.yaml
```
