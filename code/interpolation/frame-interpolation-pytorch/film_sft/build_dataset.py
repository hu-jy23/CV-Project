# build_dataset.py

import os
import cv2
from PIL import Image
from tqdm import tqdm

def extract_triplets_from_video(
    video_path,
    output_root="train_data",
    resize=(512, 512),
    stride=2,
    gray=True,
    min_length=3
):
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()

    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        if gray:
            pil_img = pil_img.convert("L").convert("RGB")

        if resize:
            pil_img = pil_img.resize(resize)

        frames.append(pil_img)
        success, frame = cap.read()

    cap.release()

    # 遍历所有连续三帧
    triplet_count = 0
    for i in tqdm(range(0, len(frames) - 2 * stride), desc="Extracting triplets"):
        i0, it, i1 = i, i + stride, i + 2 * stride
        if i1 >= len(frames):
            break

        scene_dir = os.path.join(output_root, f"scene_{triplet_count:04d}")
        os.makedirs(scene_dir, exist_ok=True)

        frames[i0].save(os.path.join(scene_dir, "frame_0.png"))
        frames[it].save(os.path.join(scene_dir, "frame_t.png"))  # 中间帧（GT）
        frames[i1].save(os.path.join(scene_dir, "frame_1.png"))

        triplet_count += 1

    print(f"✅ 生成完成：共生成 {triplet_count} 个三帧训练样本。")

if __name__ == "__main__":
    extract_triplets_from_video(
        video_path="./raw_data/input_video.mp4",
        output_root="./train_data",
        resize=(512, 512),
        stride=2,
        gray=True
    )
