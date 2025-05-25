import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import shutil
import argparse

def get_sorted_frames(folder_path):
    """获取按编号排序的帧文件列表"""
    frames = [f for f in os.listdir(folder_path) if f.startswith('frame_') and f.endswith('.png')]
    frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(f"找到 {len(frames)} 帧文件")
    return frames

def calculate_frame_distance(frame1, frame2):
    """改进版距离计算（增加局部特征对比）"""
    # 原始指标
    ssim_score, _ = ssim(frame1, frame2, full=True)
    hist_diff = cv2.compareHist(
        cv2.calcHist([frame1], [0], None, [256], [0, 256]),
        cv2.calcHist([frame2], [0], None, [256], [0, 256]),
        cv2.HISTCMP_CORREL
    )
    
    # 新增：局部区块差异（检测渐变）
    h, w = frame1.shape
    block_size = 32
    block_diffs = []
    for y in range(0, h-block_size, block_size):
        for x in range(0, w-block_size, block_size):
            blk1 = frame1[y:y+block_size, x:x+block_size]
            blk2 = frame2[y:y+block_size, x:x+block_size]
            blk_diff = np.mean(np.abs(blk1.astype("float") - blk2.astype("float")))
            block_diffs.append(blk_diff)
    
    # 组合指标（调整权重）
    return 1 - (0.4*ssim_score + 0.3*hist_diff + 0.3*(1 - np.mean(block_diffs)/255))

def detect_shot_boundaries(folder_path, threshold=0.4, window_size=5):
    """检测镜头切换（包含渐变）"""
    frames = get_sorted_frames(folder_path)
    boundaries = [0]
    window = []  # 存储最近几帧的距离值
    
    prev_frame = None
    
    for i in tqdm(range(len(frames)), desc="检测镜头切换"):
        frame_path = os.path.join(folder_path, frames[i])
        current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        if prev_frame is not None:
            distance = calculate_frame_distance(prev_frame, current_frame)
            window.append(distance)
            
            # 当窗口填满时
            if len(window) >= window_size:
                avg_diff = sum(window) / len(window)
                max_diff = max(window)
                
                # 双重条件检测：平均变化小但存在峰值（硬切）
                # 或持续中等变化（渐变）
                if (max_diff > threshold*1.5) or \
                   (0.2*threshold < avg_diff < threshold and max_diff > threshold*0.8):
                    boundaries.append(i - window_size//2)  # 取窗口中间位置
                    window = []  # 重置窗口
        
        prev_frame = current_frame
        window = window[-window_size:]  # 保持窗口大小
    
    boundaries.append(len(frames))
    return boundaries, frames

def classify_transition(frames, boundary_idx, folder_path, window=10):
    """判断切换类型（返回'cut'/'fade'/'dissolve'）"""
    before = boundary_idx - window
    after = boundary_idx + window
    diffs = []
    
    for i in range(max(0, before), min(len(frames), after)-1):
        frame1 = cv2.imread(os.path.join(folder_path, frames[i]), cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(os.path.join(folder_path, frames[i+1]), cv2.IMREAD_GRAYSCALE)
        diffs.append(calculate_frame_distance(frame1, frame2))
    
    if max(diffs) > 0.6:  # 硬切阈值
        return "cut"
    elif np.mean(diffs[:window//2]) < 0.1 and np.mean(diffs[window//2:]) > 0.3:
        return "fade"
    else:
        return "dissolve"

def clear_directory(dir_path):
    """清空目录（如果存在则删除重建）"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def save_key_frames(boundaries, frames, input_folder, output_folder):
    """保存每个镜头的关键帧（对于渐变保存最后一帧）"""
    clear_directory(output_folder)
    
    # 对每个边界点进行分类
    transition_types = []
    for i in range(1, len(boundaries)-1):
        trans_type = classify_transition(frames, boundaries[i], input_folder)
        transition_types.append(trans_type)
    
    # 处理每个镜头
    for i in range(len(boundaries)-1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        
        # 如果是第一个镜头，直接保存第一帧
        if i == 0:
            frame_idx = start_idx
        else:
            # 根据过渡类型决定保存哪一帧
            if transition_types[i-1] == "cut":
                frame_idx = start_idx  # 硬切保存新镜头第一帧
            else:
                frame_idx = start_idx  # 渐变保存新镜头第一帧（即渐变最后一帧）
                # 或者可以更精确地找到渐变结束点：
                # frame_idx = min(start_idx + 5, end_idx - 1)  # 取开始后5帧或结束前1帧
        
        frame_name = frames[frame_idx]
        src_path = os.path.join(input_folder, frame_name)
        dst_path = os.path.join(output_folder, f"shot_{i+1}_{frame_name}")
        
        img = cv2.imread(src_path)
        cv2.imwrite(dst_path, img)
        print(f"保存: {dst_path} (类型: {transition_types[i-1] if i>0 else 'first shot'})")

def save_shot_frames(boundaries, frames, input_folder, all_frame_output_folder):
    """保存每个镜头的所有帧到对应的子文件夹"""
    clear_directory(all_frame_output_folder)

    for i in range(len(boundaries)-1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        
        # 创建子文件夹
        shot_folder = os.path.join(all_frame_output_folder, f"set_{i+1}")
        os.makedirs(shot_folder, exist_ok=True)
        
        # 复制该镜头的所有帧
        for idx in range(start_idx, end_idx):
            frame_name = frames[idx]
            src_path = os.path.join(input_folder, frame_name)
            dst_path = os.path.join(shot_folder, frame_name)
            
            shutil.copy2(src_path, dst_path)
        
        print(f"保存镜头 {i+1}: {end_idx-start_idx} 帧 -> {shot_folder}")

def copy_ground_truth_frames(gt_source_folder, gt_target_folder, boundaries, frames):
    """复制ground truth参考帧到目标文件夹（只复制边界帧）"""
    clear_directory(gt_target_folder)
    
    # 遍历所有边界点（不包括最后一个结束标记）
    for boundary in boundaries[:-1]:
        frame_name = frames[boundary]
        src_path = os.path.join(gt_source_folder, frame_name)
        dst_path = os.path.join(gt_target_folder, frame_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"复制边界参考帧: {frame_name}")
        else:
            print(f"警告: 未找到边界参考帧 {frame_name}")

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='视频关键帧提取工具')
    parser.add_argument('--input', type=str, required=True, help='输入帧序列目录路径')
    parser.add_argument('--output', type=str, required=True, help='输出关键帧保存目录')
    parser.add_argument('--all_frame_output', type=str, required=True, help='输出所有帧保存目录')
    parser.add_argument('--use_ground_truth_ref', action='store_true', help='是否使用ground truth参考帧')
    parser.add_argument('--gt_source', type=str, help='ground truth参考帧源文件夹路径')
    parser.add_argument('--gt_target', type=str, default='gt_frames', help='ground truth参考帧目标文件夹路径')
    
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    
    # 检查输入目录是否存在
    if not os.path.exists(input_folder):
        print(f"错误：输入目录 {input_folder} 不存在!")
        exit(1)
        
    # 获取帧列表并检查是否为空
    frames = get_sorted_frames(input_folder)
    if not frames:
        print(f"错误：在目录 {input_folder} 中没有找到符合格式的帧图像!")
        print("图像文件名应该类似：frame_001.png")
        exit(1)
    
    # 如果使用ground truth参考帧，检查源文件夹
    if args.use_ground_truth_ref:
        if not args.gt_source:
            print("错误：使用ground truth参考帧需要指定--gt_source参数!")
            exit(1)
        if not os.path.exists(args.gt_source):
            print(f"错误：ground truth源目录 {args.gt_source} 不存在!")
            exit(1)
    
    # 检测镜头切换
    boundaries, frames = detect_shot_boundaries(
        input_folder, 
        threshold=0.2,  # 降低基础阈值
        window_size=7    # 更长的观察窗口
    )
    
    print(f"检测到 {len(boundaries)} 个镜头切换点: {boundaries}")

    if frames:
        # 保存每个镜头的关键帧（改进版）
        save_key_frames(boundaries, frames, input_folder, output_folder)
        print("完成！所有镜头的关键帧已保存。")
        
        # 保存所有帧按镜头分组
        save_shot_frames(boundaries, frames, input_folder, args.all_frame_output)
        print("完成！所有镜头帧已按组保存。")
        
        # 如果需要，复制ground truth参考帧
        if args.use_ground_truth_ref:
            copy_ground_truth_frames(args.gt_source, args.gt_target, boundaries, frames)
            print("完成！所有ground truth参考帧已复制。")
    else:
        print("错误：没有找到任何帧可以处理！")