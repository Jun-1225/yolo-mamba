import os
import shutil
from pathlib import Path

def restructure_to_yolo(source_root, dest_root):
    """
    将数据集重组为 YOLO 格式，并确保 txt 文件名与图片文件名严格对应
    """
    for split in ['Train', 'Test']:
        # 确定目标 split 名称
        target_split = 'train2017' if split == 'Train' else 'val2017'
        
        # 源路径
        src_images_dir = Path(source_root) / split / 'images'
        src_txt_dir = Path(source_root) / split / 'bbox' / 'txt'
        
        # 目标路径
        dest_images_path = Path(dest_root) / 'images' / target_split
        dest_labels_path = Path(dest_root) / 'labels' / target_split
        
        # 创建目标目录
        dest_images_path.mkdir(parents=True, exist_ok=True)
        dest_labels_path.mkdir(parents=True, exist_ok=True)
        
        # 以图片为基准进行遍历
        for img_path in src_images_dir.glob('*'):
            if img_path.is_file():
                # 1. 复制图片
                shutil.copy2(img_path, dest_images_path / img_path.name)
                
                # 2. 处理对应的 txt 文件
                # 获取不带后缀的文件名，例如 'frame_001'
                file_stem = img_path.stem 
                
                # 假设对应的 txt 文件在 src_txt_dir 下也叫 'frame_001.txt'
                # 如果源 txt 命名不规则，这里可以根据你的实际规律修改
                target_txt_file = src_txt_dir / f"{file_stem}.jpeg.txt"
                
                if target_txt_file.exists():
                    shutil.copy2(target_txt_file, dest_labels_path / f"{file_stem}.txt")
                else:
                    print(f"警告: 未找到图片 {img_path.name} 对应的标注文件")

        print(f"✅ 完成 {split} -> {target_split}")

# 执行
restructure_to_yolo('/root/autodl-tmp/COCO-Bridge-2021-plus/320x320', '/root/autodl-tmp/coco3')