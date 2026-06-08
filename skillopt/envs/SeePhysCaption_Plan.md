# Kế hoạch Xây dựng Môi trường SeePhysCaption

Tài liệu này trình bày kế hoạch chi tiết để xây dựng một môi trường mới trong hệ thống `SkillOpt` nhằm tối ưu hóa mô hình Qwen cho tác vụ Image Captioning (Khôi phục thông tin văn bản từ ảnh).

## 1. Mục tiêu (Objective)
Xây dựng môi trường (tạm gọi là `SeePhysCaption`) để huấn luyện LLM (Qwen). Mô hình sẽ nhận vào văn bản đã bị rút gọn (`problem`) và hình ảnh (`images`) từ các Level khó (Level 2, 3, hoặc 4) và học cách sinh ra một đoạn mô tả (caption) sao cho sát với văn bản gốc đầy đủ thông tin nhất (trường `problem` ở Level 1).

## 2. Cấu trúc Thư mục Đề xuất (Directory Structure)
Môi trường mới sẽ được tạo tại `/media/hung/DATA/SkillOpt/skillopt/envs/SeePhysCaption/`:
```text
skillopt/envs/SeePhysCaption/
├── __init__.py
├── dataloader.py    # Logic đọc và ghép nối dữ liệu giữa Level 1 và Level 2/3/4
├── rollout.py       # Logic gọi Target Model (Qwen) sinh caption và Judge Model chấm điểm
├── evaluator.py     # Tính toán các chỉ số đánh giá (metrics)
├── adapter.py       # (Tùy chọn) Chuyển đổi định dạng dữ liệu
└── skills/
    └── initial.md   # Skill khởi tạo, chứa prompt cơ bản về cách caption ảnh vật lý
```

## 3. Chi tiết Từng Thành phần (Component Design)

### 3.1. DataLoader (`dataloader.py`)
Khác với `SeePhys2025` thông thường chỉ load 1 tập dữ liệu, `dataloader.py` mới cần khả năng ghép nối (join) dữ liệu:
*   **Input Data**: Tải hai bộ dataset song song từ `/media/hung/DATA/SkillOpt/SeePhys_2026_data`.
    *   *Reference (Ground Truth)*: Luôn là `level1`.
    *   *Source (Input)*: Được cấu hình là `level2`, `level3`, hoặc `level4`.
*   **Logic Ghép nối**: Sử dụng trường `question_id` (hoặc `row_id`) để map thông tin. Với mỗi mẫu, ta tạo ra một bản ghi chuẩn hóa:
    *   `id`: `question_id`
    *   `input_text`: `problem` từ Level N (N = 2, 3, 4).
    *   `images`: `images` từ Level N.
    *   `ground_truth_caption`: `problem` từ Level 1.

### 3.2. Luồng Thực thi (`rollout.py`)
Quá trình đánh giá mỗi mẫu (Rollout) gồm hai bước gọi LLM:
*   **A. Target Model (Sinh Caption):**
    *   *Nhiệm vụ*: Dựa vào skill hiện tại, đóng vai trò là một chuyên gia trích xuất thông tin ảnh. Nhận vào `input_text` và `images`, sinh ra đoạn mô tả chi tiết.
    *   *Prompt Cấu trúc*:
        ```markdown
        ## Skill Instructions
        {skill_content}

        ## Input Problem
        {input_text}

        Hãy phân tích các hình ảnh được cung cấp và bổ sung các thông tin vật lý/hình học bị thiếu vào văn bản trên để tạo thành một đoạn mô tả đầy đủ nhất.
        ```
*   **B. Hàm Đánh Giá (Hybrid Evaluator):**
    *   *Nhiệm vụ*: Thay vì sử dụng LLM Judge tốn kém và có khả năng ảo giác, chúng ta sử dụng một hàm đánh giá lai (Hybrid Evaluator) tính toán trực tiếp bằng Python. Hàm này so sánh `generated_caption` với `ground_truth_caption` (Level 1).
    *   *Cơ chế tính điểm*:
        *   **ROUGE-L**: Đánh giá độ chồng chéo của cấu trúc câu dài.
        *   **BLEU**: Đánh giá độ chuẩn xác của các cụm N-gram.
        *   **Number Extraction (Regex)**: Rút trích tất cả các con số từ Ground Truth và kiểm tra xem Generated Caption có chứa đủ hay không. Đây là metric quan trọng nhất đối với bài toán vật lý.
    *   *Kết quả trả về*:
        *   `soft_score` (0.0 đến 1.0): Điểm trung bình có trọng số của ROUGE, BLEU và tỷ lệ khớp con số.
        *   `hard_score` (0 hoặc 1): Đạt 1 nếu `soft_score >= threshold` VÀ không bỏ sót bất kỳ con số vật lý nào.
        *   `reason`: Tự động sinh ra lý do bằng code (Ví dụ: "Thất bại. Mô hình bỏ sót các thông số: ['50', '2.5']. ROUGE-L thấp: 0.5"). Chuỗi này sẽ được feed lại cho Optimizer.

### 3.3. Đánh giá & Cập nhật (`evaluator.py` & Optimizer)
*   **Evaluator**: Tính điểm trung bình `soft` và tỷ lệ `hard` pass rate trên toàn bộ tập Validation/Test.
*   **Optimizer Flow**: Việc phân tích thất bại (`Reflect`) ở SkillOpt sẽ dựa vào `reason` của Judge. Nếu Judge báo mô hình thiếu khả năng đọc nhãn trên trục đồ thị, Optimizer sẽ tự động đề xuất sửa chữa file Skill để bổ sung quy tắc "Luôn chú ý đọc các nhãn trên trục x và y của biểu đồ".

## 4. Các Bước Thực hiện Cụ thể (Execution Plan)

1.  **Bước 1: Khởi tạo Bộ khung (Scaffolding)**
    *   Tạo thư mục `SeePhysCaption` và sao chép cấu trúc cơ bản từ `SeePhys2025` sang.
2.  **Bước 2: Xây dựng Multi-level DataLoader**
    *   Viết code dùng thư viện `datasets` (hoặc Pandas/Parquet trực tiếp) để đọc `dataset_info.json` và các file parquet tại `level1` và `level2`.
    *   Test log thử 1-2 mẫu dữ liệu sau khi join để đảm bảo dữ liệu khớp nhau.
3.  **Bước 3: Viết Prompts và Rollout**
    *   Cập nhật `rollout.py` để sử dụng cấu trúc Prompt cho tác vụ sinh văn bản thay vì giải toán.
    *   Thiết kế JSON schema trả về cho Judge thật chặt chẽ: `{"hard": 0|1, "soft": float, "reason": "string"}`.
4.  **Bước 4: Cấu hình (Configuration)**
    *   Tạo file `configs/SeePhysCaption/default.yaml` khai báo các tham số data paths và LLM models (tương tự như `SeePhys2025`).
5.  **Bước 5: Chạy thử và Tinh chỉnh (Dry Run)**
    *   Chạy 1 Epoch với `limit=10` mẫu để xác nhận luồng: Dữ liệu vào -> Qwen sinh caption -> Qwen chấm điểm -> Hệ thống phân tích lỗi và sinh Patch mới.

Kế hoạch này tận dụng tối đa cơ sở hạ tầng đã có của SkillOpt (Optimizer, Aggregate, Gating,...) mà chỉ thay đổi cách định nghĩa tác vụ (Task Definition) và hàm mục tiêu (Objective/Judge).
