import cv2
import os
import argparse

def extract_frames(video_path, output_folder_color, output_folder_bw):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder_color, exist_ok=True)
    os.makedirs(output_folder_bw, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 无法打开视频文件")
        return
    
    frame_count = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果没有帧了，退出循环
        if not ret:
            break
        
        # 彩色帧文件名
        color_filename = os.path.join(output_folder_color, f"frame_{frame_count:04d}.png")
        
        # 保存彩色帧
        cv2.imwrite(color_filename, frame)
        
        # 转换为黑白
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 黑白帧文件名（与彩色帧相同，但放在不同文件夹）
        bw_filename = os.path.join(output_folder_bw, f"frame_{frame_count:04d}.png")
        
        # 保存黑白帧
        cv2.imwrite(bw_filename, frame_bw)
        
        frame_count += 1
        
        # 每处理100帧打印一次进度
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧...")
    
    # 释放视频资源
    cap.release()
    print(f"处理完成！共提取 {frame_count} 帧。")
    print(f"彩色帧保存在: {os.path.abspath(output_folder_color)}")
    print(f"黑白帧保存在: {os.path.abspath(output_folder_bw)}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='提取视频帧并生成黑白版本')
    parser.add_argument('--video_path', help='输入视频文件路径')
    parser.add_argument('--color', default='frames_color', 
                       help='彩色帧输出文件夹路径 (默认: frames_color)')
    parser.add_argument('--bw', default='frames_bw', 
                       help='黑白帧输出文件夹路径 (默认: frames_bw)')
    
    args = parser.parse_args()
    
    # 调用函数处理视频
    extract_frames(args.video_path, args.color, args.bw)