# convert_vimeo_testset_complex.py

import os
import shutil
from tqdm import tqdm

def convert_vimeo_complex(
    input_dir="./vimeo_interp_test/input",
    target_dir="./vimeo_interp_test/target",
    output_root="./train_data"
):
    triplet_paths = []

    for root, dirs, files in os.walk(input_dir):
        if "im1.png" in files and "im3.png" in files:
            rel_path = os.path.relpath(root, input_dir)
            im1_path = os.path.join(input_dir, rel_path, "im1.png")
            im3_path = os.path.join(input_dir, rel_path, "im3.png")
            im2_path = os.path.join(target_dir, rel_path, "im2.png")

            if not os.path.exists(im2_path):
                continue

            triplet_paths.append((im1_path, im2_path, im3_path))

    print(f"🔍 找到 {len(triplet_paths)} 个合法样本")

    for idx, (im1, im2, im3) in enumerate(tqdm(triplet_paths, desc="转换数据")):
        out_dir = os.path.join(output_root, f"scene_{idx:04d}")
        os.makedirs(out_dir, exist_ok=True)

        shutil.copy(im1, os.path.join(out_dir, "frame_0.png"))
        shutil.copy(im2, os.path.join(out_dir, "frame_t.png"))
        shutil.copy(im3, os.path.join(out_dir, "frame_1.png"))

    print(f"✅ 完成转换，共生成 {len(triplet_paths)} 个训练样本，输出到 `{output_root}`")

if __name__ == "__main__":
    convert_vimeo_complex(
        input_dir="./vimeo_interp_test/input",
        target_dir="./vimeo_interp_test/target",
        output_root="./train_data"
    )
