import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def get_sorted_frames(folder_path):
    """获取按编号排序的帧文件列表"""
    frames = [f for f in os.listdir(folder_path) if f.startswith('frame_') and f.endswith('.png')]
    frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return frames

def calculate_frame_distance(frame1, frame2):
    """计算两帧之间的距离度量（针对灰度图像优化）"""
    # 计算结构相似性指数 (SSIM)
    ssim_score, _ = ssim(frame1, frame2, full=True)
    
    # 计算直方图差异
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 计算绝对像素差异
    pixel_diff = np.mean(np.abs(frame1.astype("float") - frame2.astype("float"))) / 255.0
    
    # 组合多个指标（权重可根据需要调整）
    distance = 1 - (0.5 * ssim_score + 0.3 * hist_diff + 0.2 * (1 - pixel_diff))
    return distance

def detect_shot_boundaries(folder_path, threshold=0.4):
    """检测镜头切换边界"""
    frames = get_sorted_frames(folder_path)
    boundaries = [0]  # 第一个镜头从第0帧开始
    
    prev_frame = None
    
    for i in tqdm(range(len(frames)), desc="检测镜头切换"):
        frame_path = os.path.join(folder_path, frames[i])
        current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        if prev_frame is not None:
            distance = calculate_frame_distance(prev_frame, current_frame)
            if distance > threshold:
                boundaries.append(i)
        
        prev_frame = current_frame
    
    return boundaries, frames

def save_first_frames(boundaries, frames, input_folder, output_folder):
    """保存每个镜头的第一帧"""
    os.makedirs(output_folder, exist_ok=True)
    
    for i, boundary in enumerate(boundaries):
        frame_name = frames[boundary]
        src_path = os.path.join(input_folder, frame_name)
        dst_path = os.path.join(output_folder, f"shot_{i+1}_{frame_name}")
        
        img = cv2.imread(src_path)
        cv2.imwrite(dst_path, img)
        print(f"保存: {dst_path}")

if __name__ == "__main__":
    input_folder = "./dataset/temp/test_input/test_input_new"
    output_folder = "./dataset/videvo/test/imgs/test_new"
    
    # 检测镜头切换
    boundaries, frames = detect_shot_boundaries(input_folder)
    print(f"检测到 {len(boundaries)} 个镜头切换点: {boundaries}")
    
    # 保存每个镜头的第一帧
    save_first_frames(boundaries, frames, input_folder, output_folder)
    print("完成！所有镜头的第一帧已保存。")