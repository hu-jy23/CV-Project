import os
import cv2
import argparse
import re

def create_video_from_images(image_paths, output_video, fps=30):
    """
    将图片列表转换为视频
    
    参数:
        image_paths: 图片路径列表
        output_video: 输出视频文件路径
        fps: 帧率 (默认30)
    """
    if not image_paths:
        print("没有找到任何图片")
        return
    
    # 读取第一张图片获取尺寸
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'avc1'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 写入所有图片
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        video.write(frame)
        print(f"已处理: {img_path}")
    
    video.release()
    print(f"视频已保存到: {output_video}")

def natural_sort_key(s):
    """
    自然排序辅助函数
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]

def process_sets(root_dir, output_dir="output_videos", fps=30):
    """
    处理所有set目录并合并为一个视频
    
    参数:
        root_dir: 包含set目录的根目录
        output_dir: 输出视频的目录
        fps: 帧率 (默认30)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有set目录并按自然顺序排序
    set_dirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("set_")]
    
    # 使用自然排序确保set_1, set_2,..., set_10的顺序正确
    set_dirs.sort(key=natural_sort_key)
    
    if not set_dirs:
        print(f"在 {root_dir} 中没有找到set目录")
        return
    
    # 收集所有图片路径并按set顺序和文件名排序
    all_images = []
    for set_dir in set_dirs:  # 已经按自然顺序排序
        set_path = os.path.join(root_dir, set_dir)
        images = [os.path.join(set_path, img) for img in os.listdir(set_path) if img.endswith(".png")]
        images.sort(key=natural_sort_key)  # 确保每个set内的图片按顺序处理
        all_images.extend(images)
    
    # 生成输出视频路径
    output_video = os.path.join(output_dir, "combined_video.mp4")
    
    # 创建视频
    print(f"\n正在处理 {len(all_images)} 张图片...")
    create_video_from_images(all_images, output_video, fps)

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将多个set目录的图片序列合并为一个视频')
    parser.add_argument('--input', type=str, required=True, help='包含set目录的根目录')
    parser.add_argument('--output', type=str, default="output_videos", help='输出视频的目录')
    parser.add_argument('--fps', type=int, default=30, help='输出视频的帧率')
    
    args = parser.parse_args()
    
    # 处理所有set目录
    process_sets(args.input, args.output, args.fps)