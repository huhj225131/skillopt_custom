from datasets import load_dataset
import os

# 1. Cấu hình tên dataset và thư mục lưu trữ chính
dataset_name = "Kun-Xiang/SeePhysPro"
save_path = "./SeePhys_2026_data"  # Thư mục cha để chứa dữ liệu các level

# Tạo thư mục cha nếu chưa tồn tại
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 2. Tải và lưu cả 5 level (từ level1 đến level5) cho split "testmini"
for level in range(1, 6):
    config_name = f"level{level}"
    print(f"Đang tải dataset {dataset_name} - {config_name} (split: testmini)...")
    
    try:
        # Tải cấu hình cụ thể và split "testmini"
        dataset = load_dataset(dataset_name, name=config_name, split="testmini")
        
        # Đường dẫn lưu riêng cho từng level
        level_save_path = os.path.join(save_path, config_name)
        
        # Lưu dataset xuống ổ cứng
        dataset.save_to_disk(level_save_path)
        print(f"Đã tải và lưu thành công {config_name} tại: {os.path.abspath(level_save_path)}")
    except Exception as e:
        print(f"Lỗi khi tải hoặc lưu {config_name}: {e}")

print("\nHoàn thành tải và lưu tất cả các level!")

# --- Cách để load lại sau này mà không cần mạng ---
# from datasets import load_from_disk
# for level in range(1, 6):
#     level_path = os.path.join("./SeePhys_2026_data", f"level{level}")
#     if os.path.exists(level_path):
#         local_ds = load_from_disk(level_path)
#         print(f"Loaded {f'level{level}'}:", local_ds)