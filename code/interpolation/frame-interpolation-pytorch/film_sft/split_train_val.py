# split_train_val.py

import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, out_dir="train_data_split", val_ratio=0.1, seed=42):
    random.seed(seed)
    all_scenes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    random.shuffle(all_scenes)

    val_size = int(len(all_scenes) * val_ratio)
    val_scenes = all_scenes[:val_size]
    train_scenes = all_scenes[val_size:]

    for mode, scenes in [('train', train_scenes), ('val', val_scenes)]:
        mode_dir = os.path.join(out_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        for scene in tqdm(scenes, desc=f"Copying {mode}"):
            src = os.path.join(source_dir, scene)
            dst = os.path.join(mode_dir, scene)
            shutil.copytree(src, dst)

    print(f"✅ 拆分完成：训练集 {len(train_scenes)}，验证集 {len(val_scenes)}，输出目录为 {out_dir}")

if __name__ == "__main__":
    split_dataset("train_data", out_dir="train_data_split", val_ratio=0.1)
