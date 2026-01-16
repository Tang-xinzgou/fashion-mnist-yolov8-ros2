#!/usr/bin/env python3
import os, cv2, numpy as np, yaml
from ultralytics import YOLO

print("Fashion-MNIST 训练（50 epochs）")

# 创建目录
os.makedirs('data_fashion/images/train', exist_ok=True)
os.makedirs('data_fashion/labels/train', exist_ok=True)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 快速生成数据
for cls_id in range(10):
    for i in range(50):  # 每类50张
        # 创建有明显特征的图片
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # 不同类别的不同图案
        if cls_id == 0:  # 横条纹
            for row in range(0, 64, 8):
                img[row:row+4, :] = [200, 100, 100]
        elif cls_id == 1:  # 竖条纹
            for col in range(0, 64, 6):
                img[:, col:col+3] = [100, 200, 100]
        elif cls_id == 2:  # 网格
            for row in range(0, 64, 10):
                img[row:row+2, :] = [100, 100, 200]
        else:  # 其他：简单图案
            color = [100 + cls_id*15, 150, 200 - cls_id*15]
            img[10:54, 10:54] = color
        
        cv2.imwrite(f'data_fashion/images/train/{cls_id}_{i:03d}.png', img)
        with open(f'data_fashion/labels/train/{cls_id}_{i:03d}.txt', 'w') as f:
            f.write(f'{cls_id} 0.5 0.5 0.8 0.8')

# 训练
config = {'train': 'data_fashion/images/train', 'val': 'data_fashion/images/train',
          'nc': 10, 'names': classes}
with open('data_fashion.yaml', 'w') as f: yaml.dump(config, f)

model = YOLO('yolov8n.pt')
results = model.train(
    data='data_fashion.yaml',
    epochs=50,  # 50个epochs
    imgsz=64,
    batch=16,
    device='cpu',
    project='fashion_50_epochs',
    name='train',
    exist_ok=True
)

print("✅ 50 epochs训练完成")
