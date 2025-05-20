import os
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm

# === 配置项 ===
tri_list_file = "tri_testlist.txt"
input_root = Path("vimeo_triplet/sequences/input")
target_root = Path("vimeo_triplet/sequences/target")
output_root = Path("vimeo_triplet_processed")
scales = [1.0, 0.5, 0.25]  # 多尺度训练支持

# 创建输出目录
output_root.mkdir(parents=True, exist_ok=True)

# 读取测试集文件
with open(tri_list_file, "r") as f:
    raw_list = [line.strip() for line in f if line.strip()]
triplet_list = [x for x in raw_list if "/" in x]

skipped = 0
processed = 0
output_entries = []

for item in tqdm(triplet_list):
    try:
        group, subfolder = item.split("/")
        input_dir = input_root / group / subfolder
        target_dir = target_root / group / subfolder

        # 读取图像并转换为灰度
        im1 = Image.open(input_dir / "im1.png").convert("L")
        im3 = Image.open(input_dir / "im3.png").convert("L")
        im2 = Image.open(target_dir / "im2.png").convert("L")

        for scale in scales:
            tag = f"{int(scale*100)}"
            out_dir = output_root / f"{group}_{subfolder}_{tag}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if scale != 1.0:
                new_size = (int(im1.width * scale), int(im1.height * scale))
                im1_resized = im1.resize(new_size, Image.BICUBIC)
                im2_resized = im2.resize(new_size, Image.BICUBIC)
                im3_resized = im3.resize(new_size, Image.BICUBIC)
            else:
                im1_resized, im2_resized, im3_resized = im1, im2, im3

            im1_resized.save(out_dir / "frame_00.png")
            im2_resized.save(out_dir / "frame_01.png")
            im3_resized.save(out_dir / "frame_02.png")

            output_entries.append(f"{out_dir.relative_to(output_root)}")
            processed += 1
    except Exception as e:
        skipped += 1
        print(f"❌ Skip {item}: {e}")
        continue

# 写入路径列表
with open(output_root / "file_list.txt", "w") as f:
    for entry in output_entries:
        f.write(f"{entry}\n")

print(f"✅ Done. Processed {processed}, Skipped {skipped}.")
