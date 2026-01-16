import numpy as np
from PIL import Image
import os

# 创建输出目录
os.makedirs('fashion_mnist_yolo/images/train', exist_ok=True)
os.makedirs('fashion_mnist_yolo/images/val', exist_ok=True)

# 创建10张简单的测试图片（不同灰度值）
for i in range(10):
    # 创建64x64的灰度图片
    img_array = np.full((64, 64), i * 25, dtype=np.uint8)  # 灰度值从0-225
    
    # 保存训练图片
    img_train = Image.fromarray(img_array, mode='L')
    img_train.save(f'fashion_mnist_yolo/images/train/train_{i:05d}.png')
    
    # 保存验证图片
    img_val = Image.fromarray(img_array, mode='L')
    img_val.save(f'fashion_mnist_yolo/images/val/val_{i:05d}.png')
    
    # 创建对应的标签文件
    with open(f'fashion_mnist_yolo/labels/train/train_{i:05d}.txt', 'w') as f:
        f.write(f'{i} 0.5 0.5 0.4375 0.4375\n')
    
    with open(f'fashion_mnist_yolo/labels/val/val_{i:05d}.txt', 'w') as f:
        f.write(f'{i} 0.5 0.5 0.4375 0.4375\n')

print(f"创建了10张训练图片和10张验证图片")
print(f"训练图片路径: {os.path.abspath('fashion_mnist_yolo/images/train/')}")
print(f"验证图片路径: {os.path.abspath('fashion_mnist_yolo/images/val/')}")
