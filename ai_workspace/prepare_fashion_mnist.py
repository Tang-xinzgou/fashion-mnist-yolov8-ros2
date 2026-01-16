#!/usr/bin/env python3
"""
准备Fashion-MNIST数据集用于YOLOv8训练
"""
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

print("="*80)
print("Fashion-MNIST 数据集准备")
print("="*80)

# 创建目录结构
def create_directories():
    directories = [
        'fashion_mnist_yolo/images/train',
        'fashion_mnist_yolo/images/val',
        'fashion_mnist_yolo/labels/train',
        'fashion_mnist_yolo/labels/val',
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    return directories

# 加载Fashion-MNIST数据
def load_fashion_mnist():
    try:
        # 尝试导入Fashion-MNIST
        from tensorflow.keras.datasets import fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        print(f"✅ 成功加载Fashion-MNIST数据集")
        print(f"   训练集: {train_images.shape}")
        print(f"   测试集: {test_images.shape}")
        return train_images, train_labels, test_images, test_labels
    except ImportError as e:
        print(f"❌ 无法导入tensorflow: {e}")
        print("请安装: pip install tensorflow")
        return None, None, None, None

# 预处理图片：灰度转RGB
def preprocess_image(image):
    # 如果图片是全黑的，添加一些噪声
    if np.max(image) == np.min(image):
        image = np.random.randint(0, 255, image.shape, dtype=np.uint8)
    
    # 归一化到0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # 灰度图转RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 调整大小到64x64
    if image.shape[:2] != (64, 64):
        image = cv2.resize(image, (64, 64))
    
    return image

# 生成YOLO格式标签
def create_yolo_label(label, img_size=64):
    """为整个图片创建边界框"""
    # 将类别转换为0-index
    class_id = int(label)
    
    # 创建覆盖整个图片的边界框
    # YOLO格式: class_id x_center y_center width height
    x_center = 0.5
    y_center = 0.5
    width = 0.9
    height = 0.9
    
    return f"{class_id} {x_center} {y_center} {width} {height}"

# 保存图片和标签
def save_data(images, labels, image_dir, label_dir, prefix='img'):
    saved_count = 0
    
    for i, (image, label) in enumerate(zip(images, labels)):
        if i >= 100:  # 限制数量，避免太多
            break
            
        # 预处理图片
        processed_img = preprocess_image(image)
        
        # 保存图片
        img_name = f"{prefix}_{i:05d}.png"
        img_path = os.path.join(image_dir, img_name)
        cv2.imwrite(img_path, processed_img)
        
        # 保存标签
        label_name = f"{prefix}_{i:05d}.txt"
        label_path = os.path.join(label_dir, label_name)
        label_str = create_yolo_label(label)
        
        with open(label_path, 'w') as f:
            f.write(label_str)
        
        saved_count += 1
    
    return saved_count

# 创建YAML配置文件
def create_yaml_config():
    config = {
        'path': '.',
        'train': 'fashion_mnist_yolo/images/train',
        'val': 'fashion_mnist_yolo/images/val',
        'test': '',
        'nc': 10,
        'names': [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    }
    
    with open('fashion_full.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ 创建配置文件: fashion_full.yaml")
    return config

def main():
    # 1. 创建目录
    create_directories()
    
    # 2. 加载数据
    train_images, train_labels, test_images, test_labels = load_fashion_mnist()
    
    if train_images is None:
        print("❌ 无法加载数据集，使用现有数据")
        
        # 检查现有数据
        if os.path.exists('fashion_mnist_yolo'):
            print("使用现有fashion_mnist_yolo目录")
        else:
            print("❌ 没有可用数据")
            return
    else:
        # 3. 分割数据
        print("分割数据集...")
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42
        )
        
        # 4. 保存训练集
        print("保存训练集...")
        train_count = save_data(
            train_images, train_labels,
            'fashion_mnist_yolo/images/train',
            'fashion_mnist_yolo/labels/train',
            'train'
        )
        
        # 5. 保存验证集
        print("保存验证集...")
        val_count = save_data(
            val_images, val_labels,
            'fashion_mnist_yolo/images/val',
            'fashion_mnist_yolo/labels/val',
            'val'
        )
        
        # 6. 保存测试集
        print("保存测试集...")
        test_count = save_data(
            test_images[:20], test_labels[:20],  # 只保存20张测试
            'fashion_mnist_yolo/images/val',  # 与验证集合并
            'fashion_mnist_yolo/labels/val',
            'test'
        )
        
        print(f"✅ 数据保存完成:")
        print(f"   训练集: {train_count} 张")
        print(f"   验证集: {val_count + test_count} 张")
    
    # 7. 创建配置文件
    config = create_yaml_config()
    
    # 8. 验证数据
    print("\n数据验证:")
    train_imgs = list(Path('fashion_mnist_yolo/images/train').glob('*.png'))
    val_imgs = list(Path('fashion_mnist_yolo/images/val').glob('*.png'))
    train_labels = list(Path('fashion_mnist_yolo/labels/train').glob('*.txt'))
    val_labels = list(Path('fashion_mnist_yolo/labels/val').glob('*.txt'))
    
    print(f"   训练图片: {len(train_imgs)} 张")
    print(f"   训练标签: {len(train_labels)} 个")
    print(f"   验证图片: {len(val_imgs)} 张")
    print(f"   验证标签: {len(val_labels)} 个")
    
    if train_imgs:
        sample_img = cv2.imread(str(train_imgs[0]))
        print(f"   样本图片尺寸: {sample_img.shape}")
        print(f"   样本图片像素范围: [{sample_img.min()}, {sample_img.max()}]")
    
    print("\n" + "="*80)
    print("✅ 数据集准备完成")
    print("="*80)

if __name__ == "__main__":
    main()
