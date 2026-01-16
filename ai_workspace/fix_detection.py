#!/usr/bin/env python3
from ultralytics import YOLO
import os
import numpy as np
import cv2

print("="*60)
print("诊断和修复检测问题")
print("="*60)

# 1. 检查模型文件
model_path = 'fashion_result/train/weights/best.pt'
if not os.path.exists(model_path):
    print("❌ 模型文件不存在")
    exit()

print(f"✅ 模型文件: {model_path}")

# 2. 加载模型
model = YOLO(model_path)
print("✅ 模型加载成功")

# 3. 创建更好的测试图片
print("\n🔧 创建更好的测试图片...")

# 为10个类别创建有明显特征的图片
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(5):  # 创建5张测试图
    # 创建有明显特征的图片（不是纯随机噪声）
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    
    # 为每个类别创建不同图案
    if i == 0:  # T-shirt/top: 横条纹
        for row in range(0, 64, 8):
            img[row:row+4, :] = [200, 100, 100]
    elif i == 1:  # Trouser: 竖条纹
        for col in range(0, 64, 6):
            img[:, col:col+3] = [100, 200, 100]
    elif i == 2:  # Pullover: 网格
        for row in range(0, 64, 10):
            img[row:row+2, :] = [100, 100, 200]
        for col in range(0, 64, 10):
            img[:, col:col+2] = [100, 100, 200]
    elif i == 3:  # Dress: 对角线条纹
        for j in range(64):
            if (j + i*5) % 8 < 4:
                img[j, :] = [200, 150, 100]
    else:  # 其他: 圆形
        center = 32
        cv2.circle(img, (center, center), 20, (150, 200, 150), -1)
    
    # 保存测试图片
    test_path = f'test_img_{i}.png'
    cv2.imwrite(test_path, img)
    
    # 测试识别
    print(f"\n测试图片 {i+1}: {test_path}")
    
    # 尝试极低的置信度阈值
    for conf in [0.001, 0.01, 0.05, 0.1]:
        results = model.predict(test_path, conf=conf, verbose=False)
        
        if results[0].boxes and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # 获取置信度最高的检测
            confidences = boxes.conf.cpu().numpy()
            if len(confidences) > 0:
                best_idx = confidences.argmax()
                cls_id = int(boxes.cls[best_idx])
                conf_score = float(confidences[best_idx])
                
                if 0 <= cls_id < len(classes):
                    print(f"  ✅ 阈值={conf}: 检测到 {classes[cls_id]}, 置信度={conf_score:.4f}")
                else:
                    print(f"  ⚠️ 阈值={conf}: 类别ID超出范围: {cls_id}")
            break
    else:
        print(f"  ❌ 所有阈值下都未检测到目标")

print("\n" + "="*60)
print("如果还是检测不到，建议重新训练：")
print("1. 生成更好的训练数据（有明显特征）")
print("2. 增加训练轮数（epochs=50+）")
print("3. 降低检测阈值（conf=0.01）")
print("="*60)
