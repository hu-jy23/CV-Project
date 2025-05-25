import os
import shutil

def organize_frames(source_dir, ranges):
    """
    将帧图片按照指定范围分类到不同子文件夹
    
    参数:
        source_dir: 源文件夹路径
        ranges: 元组列表，每个元组包含(起始索引, 结束索引, 目标文件夹名)
    """
    # 确保目标文件夹存在
    for _, _, target_folder in ranges:
        os.makedirs(os.path.join(source_dir, target_folder), exist_ok=True)
    
    # 遍历所有帧文件
    for filename in sorted(os.listdir(source_dir)):
        if filename.startswith('frame_') and filename.endswith('.png'):
            try:
                # 提取帧编号
                frame_num = int(filename.split('_')[1].split('.')[0])
                
                # 确定属于哪个范围
                for start, end, target_folder in ranges:
                    if start <= frame_num <= end:
                        src_path = os.path.join(source_dir, filename)
                        dest_path = os.path.join(source_dir, target_folder, filename)
                        shutil.move(src_path, dest_path)
                        break
            except (IndexError, ValueError):
                print(f"跳过无法解析的文件: {filename}")
                continue

if __name__ == "__main__":
    # 设置路径和分类规则
    source_directory = "./dataset/temp/test_input/test_input_new"
    classification_ranges = [
        (0, 43, "set1"),
        (44, 74, "set2"),
        (75, 122, "set3")  # 假设最后一帧是122
    ]
    
    # 执行分类
    organize_frames(source_directory, classification_ranges)
    print("文件分类完成！")