#!/usr/bin/env python3
"""
经过验证的Fashion-MNIST识别脚本
"""
import os
import numpy as np
from ultralytics import YOLO
import sys

print("="*80)
print("Fashion-MNIST 验证识别结果")
print("="*80)

# 1. 检查模型
model_path = 'runs/detect/train/weights/best.pt'
if not os.path.exists(model_path):
    print(f"❌ 模型文件不存在: {model_path}")
    sys.exit(1)

# 2. 加载模型
print("加载模型...")
try:
    model = YOLO(model_path)
    print(f"✅ 模型加载成功")
    print(f"   模型路径: {model_path}")
    print(f"   模型任务: {model.task}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# 3. 定义类别
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 4. 测试图片
print(f"\n测试图片:")
test_images = []
for i in range(3):
    img_path = f'fashion_mnist_yolo/images/val/val_{i:05d}.png'
    if os.path.exists(img_path):
        test_images.append(img_path)
        print(f"  ✅ {os.path.basename(img_path)}")
    else:
        print(f"  ❌ {img_path} 不存在")

if not test_images:
    print("❌ 没有可用的测试图片")
    sys.exit(1)

# 5. 进行识别
print(f"\n开始识别...")
all_results = []

for i, img_path in enumerate(test_images):
    img_name = os.path.basename(img_path)
    print(f"\n[{i+1}/{len(test_images)}] {img_name}")
    print("-" * 50)
    
    # 尝试多个置信度阈值
    found = False
    for conf_threshold in [0.01, 0.05, 0.1, 0.25, 0.5]:
        if found:
            break
            
        print(f"  阈值 {conf_threshold:.3f}: ", end="")
        try:
            results = model.predict(
                source=img_path,
                conf=conf_threshold,
                device='cpu',
                verbose=False
            )
            
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                # 获取置信度最高的检测
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences) if len(confidences) > 0 else 0
                
                box = boxes[best_idx]
                cls_id = int(box.cls[best_idx]) if box.cls.numel() > 0 else -1
                conf = float(box.conf[best_idx]) if box.conf.numel() > 0 else 0
                
                if cls_id >= 0 and cls_id < len(class_names):
                    class_name = class_names[cls_id]
                    print(f"✅ {class_name} (置信度: {conf:.4f})")
                    all_results.append((img_name, class_name, conf))
                    found = True
                else:
                    print(f"⚠️ 无效类别ID: {cls_id}")
            else:
                print("❌ 无检测")
                
        except Exception as e:
            print(f"⚠️ 错误: {str(e)[:50]}")
    
    if not found:
        all_results.append((img_name, "未检测到", 0.0))

# 6. 输出结果
print("\n" + "="*80)
print("最终识别结果")
print("="*80)

for img_name, class_name, confidence in all_results:
    if confidence > 0:
        print(f"图片: {img_name}")
        print(f"  识别结果: {class_name}")
        print(f"  准确度: {confidence:.4f}")
        print(f"  置信度: {confidence*100:.2f}%")
    else:
        print(f"图片: {img_name}")
        print(f"  识别结果: 未检测到")
        print(f"  准确度: 0.0000")
    print()

# 7. 统计信息
detected = [r for r in all_results if r[2] > 0]
if detected:
    avg_conf = sum(r[2] for r in detected) / len(detected)
    print(f"平均检测置信度: {avg_conf:.4f} ({avg_conf*100:.2f}%)")
    print(f"检测成功率: {len(detected)}/{len(all_results)}")

print("="*80)
print("识别完成")
print("="*80)
