#!/usr/bin/env python3
"""
数据集诊断
"""
import os
import yaml
import cv2
import numpy as np
from pathlib import Path

print("="*80)
print("Fashion-MNIST 数据集诊断")
print("="*80)

# 1. 检查数据集配置文件
print("1. 检查数据集配置:")
config_path = 'fashion_full.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"   配置文件: {config_path}")
    print(f"   类别数: {config.get('nc', 'N/A')}")
    print(f"   类别名: {config.get('names', 'N/A')}")
    print(f"   训练集路径: {config.get('train', 'N/A')}")
    print(f"   验证集路径: {config.get('val', 'N/A')}")
    print(f"   测试集路径: {config.get('test', 'N/A')}")
    
    # 检查路径是否存在
    train_path = config.get('train', '')
    val_path = config.get('val', '')
    
    if train_path and os.path.exists(train_path):
        train_images = list(Path(train_path).glob('*.png'))[:5]
        print(f"   训练集示例: {[img.name for img in train_images[:3]]}")
    
    if val_path and os.path.exists(val_path):
        val_images = list(Path(val_path).glob('*.png'))
        print(f"   验证集数量: {len(val_images)}")
        if val_images:
            print(f"   验证集示例: {[img.name for img in val_images[:3]]}")
else:
    print(f"   配置文件不存在: {config_path}")

# 2. 检查图片格式
print(f"\n2. 检查图片格式:")
if os.path.exists('fashion_mnist_yolo/images/val'):
    sample_img = 'fashion_mnist_yolo/images/val/val_00000.png'
    if os.path.exists(sample_img):
        img = cv2.imread(sample_img, cv2.IMREAD_UNCHANGED)
        if img is not None:
            print(f"   图片尺寸: {img.shape}")
            print(f"   数据类型: {img.dtype}")
            print(f"   像素范围: [{img.min()}, {img.max()}]")
            
            # 如果是灰度图，转换为3通道看看
            if len(img.shape) == 2:
                print(f"   ⚠️ 图片是灰度图 (2D)")
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                print(f"   转换后尺寸: {img_color.shape}")
            elif len(img.shape) == 3:
                print(f"   图片是彩色图，通道数: {img.shape[2]}")
        else:
            print(f"   ❌ 无法读取图片: {sample_img}")
    else:
        print(f"   ❌ 示例图片不存在: {sample_img}")

# 3. 检查标签文件
print(f"\n3. 检查标签文件:")
if os.path.exists('fashion_mnist_yolo/labels/val'):
    sample_label = 'fashion_mnist_yolo/labels/val/val_00000.txt'
    if os.path.exists(sample_label):
        with open(sample_label, 'r') as f:
            lines = f.readlines()
        print(f"   标签示例: {[line.strip() for line in lines[:2]]}")
        print(f"   标签数量: {len(lines)}")
    else:
        print(f"   标签文件不存在: {sample_label}")

# 4. 数据集统计
print(f"\n4. 数据集统计:")
if os.path.exists('fashion_mnist_yolo/images/val'):
    val_images = list(Path('fashion_mnist_yolo/images/val').glob('*.png'))
    train_images = list(Path('fashion_mnist_yolo/images/train').glob('*.png'))
    
    print(f"   训练集图片数: {len(train_images)}")
    print(f"   验证集图片数: {len(val_images)}")
    
    # 检查类别分布
    if os.path.exists('fashion_mnist_yolo/labels/val'):
        val_labels = list(Path('fashion_mnist_yolo/labels/val').glob('*.txt'))
        class_counts = {}
        
        for label_file in val_labels[:100]:  # 检查前100个
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        
        print(f"   验证集类别分布: {dict(sorted(class_counts.items()))}")

print("\n" + "="*80)
print("数据集诊断完成")
print("="*80)
