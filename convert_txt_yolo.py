import os
import yaml
from pathlib import Path
from PIL import Image
import shutil

def convert_csv_to_yolo(
    csv_label_dir,      # 原始CSV格式txt文件目录
    image_dir,          # 图片目录
    output_label_dir,   # 输出YOLO格式txt目录
    class_to_id         # 类别到ID的映射字典
):
    """
    将CSV格式标注转换为YOLO格式
    """
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    missing_image_count = 0
    renamed_count = 0
    
    txt_files = list(Path(csv_label_dir).glob("*.txt"))
    
    print(f"找到 {len(txt_files)} 个标注文件")
    print("="*50)
    
    for txt_path in txt_files:
        txt_name = txt_path.name
        img_stem = None
        
        # 处理特殊文件名
        if '.jpeg.txt' in txt_name:
            img_stem = txt_name.replace('.jpeg.txt', '')
            renamed_count += 1
        elif '.jpg.txt' in txt_name:
            img_stem = txt_name.replace('.jpg.txt', '')
            renamed_count += 1
        elif '.png.txt' in txt_name:
            img_stem = txt_name.replace('.png.txt', '')
            renamed_count += 1
        else:
            img_stem = txt_path.stem
        
        # 查找图片
        img_path = None
        for ext in ['.jpeg', '.jpg', '.JPEG', '.JPG', '.png', '.PNG']:
            candidate = Path(image_dir) / (img_stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            print(f"❌ 未找到图片: {img_stem} (对应文件: {txt_path.name})")
            missing_image_count += 1
            continue
        
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"❌ 无法读取图片 {img_path}: {e}")
            fail_count += 1
            continue
        
        yolo_annotations = []
        invalid_lines = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 【修复2】跳过纯数字的行（即文件开头的图片宽高 320）
                if line.isdigit():
                    continue
                
                parts = line.split(',')
                if len(parts) != 5:
                    invalid_lines.append(f"行{line_num}: 格式错误，应有5个字段，实际{len(parts)}个")
                    continue
                
                class_name = parts[0].strip()
                try:
                    # 【修复1】真实格式是 x_min, y_min, x_max, y_max
                    x_min = float(parts[1])
                    y_min = float(parts[2])
                    x_max = float(parts[3])
                    y_max = float(parts[4])
                except ValueError:
                    invalid_lines.append(f"行{line_num}: 数值转换失败 - {line}")
                    continue
                
                # 计算真实的宽和高
                w = x_max - x_min
                h = y_max - y_min
                
                if w <= 0 or h <= 0:
                    invalid_lines.append(f"行{line_num}: 宽度或高度为负数或零 - w={w}, h={h}")
                    continue
                
                class_id = class_to_id.get(class_name)
                if class_id is None:
                    invalid_lines.append(f"行{line_num}: 未知类别 '{class_name}'")
                    continue
                
                # 转换为YOLO格式并归一化
                x_center = (x_min + x_max) / 2.0 / img_width
                y_center = (y_min + y_max) / 2.0 / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                
                # 确保坐标被严格限制在 [0,1] 范围内，防止框跑到隔壁图片去！
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        if invalid_lines:
            print(f"⚠️  {txt_path.name}: 发现 {len(invalid_lines)} 个问题")
            for err in invalid_lines[:2]:
                print(f"    {err}")
        
        if yolo_annotations:
            output_path = Path(output_label_dir) / (img_stem + '.txt')
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            success_count += 1
        else:
            fail_count += 1
            
    print("="*50)
    print(f"✅ 成功转换: {success_count} 个文件")
    if fail_count > 0: print(f"❌ 失败: {fail_count} 个文件")
    return success_count


def create_data_yaml(output_dir, class_names, train_path, val_path):
    data = {
        'path': output_dir,
        'train': train_path,
        'val': val_path,
        'nc': len(class_names),
        'names': class_names,
    }
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ 已创建配置文件: {yaml_path}")


if __name__ == "__main__":
    
    # 1. 类别映射
    CLASS_MAPPING = {
        'Bearing': 0,
        'Out of Plane Stiffener': 1,
        'Gusset Plate Connection': 2,
        'Cover Plate Termination': 3,
    }
    
    # 2. 路径配置（无需修改，按你原来的设置）
    TRAIN_CSV_DIR = "/root/autodl-tmp/COCO-Bridge-2021-plus/320x320/Train/bbox/txt"
    TRAIN_IMAGE_DIR = "/root/autodl-tmp/COCO-Bridge-2021-plus/320x320/Train/images"
    VAL_CSV_DIR = "/root/autodl-tmp/COCO-Bridge-2021-plus/320x320/Test/bbox/txt"
    VAL_IMAGE_DIR = "/root/autodl-tmp/COCO-Bridge-2021-plus/320x320/Test/images"
    
    OUTPUT_ROOT = "/root/autodl-tmp/coco4"
    TRAIN_LABEL_DIR = f"{OUTPUT_ROOT}/labels/train2017"
    VAL_LABEL_DIR = f"{OUTPUT_ROOT}/labels/val2017"
    TRAIN_IMAGE_OUT = f"{OUTPUT_ROOT}/images/train2017"
    VAL_IMAGE_OUT = f"{OUTPUT_ROOT}/images/val2017"
    
    # 3. 创建目录
    Path(TRAIN_LABEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(VAL_LABEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(TRAIN_IMAGE_OUT).mkdir(parents=True, exist_ok=True)
    Path(VAL_IMAGE_OUT).mkdir(parents=True, exist_ok=True)
    
    # 4. 复制图片
    print("正在复制图片...")
    for img in Path(TRAIN_IMAGE_DIR).glob("*"):
        if img.suffix.lower() in ['.jpeg', '.jpg', '.png']:
            shutil.copy2(img, Path(TRAIN_IMAGE_OUT) / img.name)
            
    for img in Path(VAL_IMAGE_DIR).glob("*"):
        if img.suffix.lower() in ['.jpeg', '.jpg', '.png']:
            shutil.copy2(img, Path(VAL_IMAGE_OUT) / img.name)
    
    # 5. 转换标注
    print("\n开始转换训练集标注...")
    convert_csv_to_yolo(TRAIN_CSV_DIR, TRAIN_IMAGE_OUT, TRAIN_LABEL_DIR, CLASS_MAPPING)
    
    print("\n开始转换验证集标注...")
    convert_csv_to_yolo(VAL_CSV_DIR, VAL_IMAGE_OUT, VAL_LABEL_DIR, CLASS_MAPPING)
    
    # 6. 生成 YAML（【修复3】字典解包顺序修复为 name, id）
    class_names = [name for name, id in sorted(CLASS_MAPPING.items(), key=lambda x: x[1])]
    create_data_yaml(OUTPUT_ROOT, class_names, "images/train2017", "images/val2017")
    
    print("\n🎉 数据集转换完成！请使用你的 Mamba-YOLO 训练脚本重新开始训练。")