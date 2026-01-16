#!/usr/bin/env python3
from ultralytics import YOLO
import os
import numpy as np
import cv2

print("="*60)
print("Fashion-MNIST 最终识别结果")
print("="*60)

# 尝试多个模型路径
model_paths = [
    'fashion_complete/train_100_epochs/weights/best.pt',
    'fashion_50_epochs/train/weights/best.pt',
    'runs/detect/train/weights/best.pt'
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if not model_path:
    print("❌ 没有找到模型文件")
    exit()

print(f"📁 使用模型: {model_path}")

# 加载模型
model = YOLO(model_path)

# 创建测试图片
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("\n生成并测试5张图片:")

for i in range(5):
    # 生成测试图片
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    class_id = i % 10
    
    # 为不同类别生成不同图案
    if class_id == 0:  # 横条纹
        for row in range(0, 64, 8):
            img[row:row+4, :] = [200, 100, 100]
    elif class_id == 1:  # 竖条纹
        for col in range(0, 64, 6):
            img[:, col:col+3] = [100, 200, 100]
    elif class_id == 2:  # 网格
        for row in range(0, 64, 10):
            img[row:row+2, :] = [100, 100, 200]
    else:  # 简单方块
        color = [100 + class_id*15, 150, 200 - class_id*15]
        img[10:54, 10:54] = color
    
    cv2.imwrite(f'test_{i}.png', img)
    
    # 测试识别
    print(f"\n测试图片 {i+1}: test_{i}.png (类别: {classes[class_id]})")
    
    # 使用低阈值
    results = model.predict(f'test_{i}.png', conf=0.01, verbose=False)
    
    if results[0].boxes and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        if len(confidences) > 0:
            best_idx = confidences.argmax()
            detected_cls = int(boxes.cls[best_idx])
            conf = float(confidences[best_idx])
            
            if 0 <= detected_cls < len(classes):
                print(f"  ✅ 识别结果: {classes[detected_cls]}")
                print(f"  ✅ 准确度: {conf:.4f}")
                print(f"  ✅ 置信度: {conf*100:.2f}%")
                
                # 检查是否正确识别
                if detected_cls == class_id:
                    print(f"  ✅ 识别正确!")
                else:
                    print(f"  ❌ 识别错误，应为: {classes[class_id]}")
            else:
                print(f"  ⚠️ 无效类别ID: {detected_cls}")
    else:
        print(f"  ❌ 未检测到目标")

print("\n" + "="*60)
print("✅ 测试完成")
print("="*60)
