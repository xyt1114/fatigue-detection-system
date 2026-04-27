import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

# 确保可以导入项目模块
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from detection.utils.dataset_loader import DatasetLoader


class FatigueImageDataset(Dataset):
    """
    用于 PyTorch 训练的 Dataset 包装类，读取 dataset_loader 收集到的路径
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # 处理可能的视频帧标记 "path/to/video.mp4|frame=0"
        if "|" in path:
            path = path.split("|")[0]
            
        try:
            # 简化起见，我们直接重新读取图片，如果遇到视频帧，我们最好在训练前就把视频抽帧存成图片
            # 这里假定 dataset_raw 里都是图片文件，或者我们这里用 PIL 读取图片
            image = Image.open(path).convert('RGB')
        except Exception as e:
            # 如果出错，返回一张纯黑图片以防崩溃
            image = Image.new('RGB', (112, 112), color='black')
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, int(label)


def create_model(num_classes=3):
    """
    创建一个轻量级 CNN 模型 (MobileNetV3-Small)，适合 CPU 推理
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # 替换最后的分类头
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 添加进度条
            pbar = tqdm(dataloader, desc=f"{phase}")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 更新进度条信息
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 保存最好模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state = model.state_dict().copy()

        print()

    print(f"Best val Acc: {best_acc:4f}")
    model.load_state_dict(best_model_state)
    return model


def main():
    parser = argparse.ArgumentParser(description="训练疲劳检测 CNN 模型")
    parser.add_argument("--dataset-dir", type=str, default=str(BASE_DIR / "dataset"), help="数据集根目录")
    parser.add_argument("--output-model", type=str, default=str(BASE_DIR / "models" / "fatigue_classifier_cnn.pt"), help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数 (为了演示设为 5，实际建议 15-30)")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--img-size", type=int, default=112, help="输入图片尺寸")
    parser.add_argument("--resume", action="store_true", help="是否从现有的 .pt 模型继续训练")
    parser.add_argument("--fine-tune-all", dest="fine_tune_all", action="store_true", help="是否微调所有层（而不仅仅是分类头），推荐在继续训练时使用")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 扫描数据集
    print(f"正在扫描数据集: {args.dataset_dir}")
    loader = DatasetLoader(dataset_root=args.dataset_dir)
    report = loader.validate_dataset()
    if not report["is_valid"]:
        print("数据集不完整或样本不足，请检查！")
        return
        
    print(f"类别分布: {report['distribution']}")

    # 获取所有图片的路径和标签
    all_paths = []
    all_labels = []
    
    # 支持两种目录结构： dataset/raw/class 和 dataset/class
    for class_name, class_id in loader.class_map.items():
        # 优先尝试 dataset/raw 目录
        class_dir = loader.dataset_root / "raw" / class_name
        if not class_dir.exists():
            # 如果没有 raw 目录，直接用 dataset/class
            class_dir = loader.dataset_root / class_name
            
        if class_dir.exists():
            for media_path in loader._iter_media_paths(class_dir):
                if loader._is_image_file(media_path):
                    all_paths.append(str(media_path))
                    all_labels.append(class_id)

    if len(all_paths) == 0:
        print("未找到图片，尝试从视频中抽帧...")
        images, labels, _ = loader.load_dataset(img_size=(args.img_size, args.img_size), augment=False)
        if len(images) == 0:
            print("未找到任何有效的数据，请确保 dataset 目录下有 awake, mild, severe 等子目录且包含视频或图片。")
            return
            
        print(f"共加载 {len(images)} 张图片（可能包含视频抽帧）")
        
        # 对于加载在内存中的数据，我们需要一个特殊的 Dataset 类
        class MemoryDataset(Dataset):
            def __init__(self, imgs, lbls, transform=None):
                self.imgs = imgs
                self.lbls = lbls
                self.transform = transform
                
            def __len__(self):
                return len(self.imgs)
                
            def __getitem__(self, idx):
                # OpenCV/numpy image is HxWxC
                image = self.imgs[idx]
                image = Image.fromarray(image)
                label = int(self.lbls[idx])  # 转换为 Python int
                if self.transform:
                    image = self.transform(image)
                return image, label
                
        dataset_size = len(images)
        val_size = int(dataset_size * 0.2)
        train_size = dataset_size - val_size

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 数据增强
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        train_dataset = MemoryDataset(X_train, y_train, transform=data_transforms['train'])
        val_dataset = MemoryDataset(X_val, y_val, transform=data_transforms['val'])
        train_labels = y_train
        
    else:
        print(f"共发现 {len(all_paths)} 张有效图片")

        # 2. 数据集划分与数据增强
        dataset_size = len(all_paths)
        val_size = int(dataset_size * 0.2)
        train_size = dataset_size - val_size

        # 数据增强
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # 划分路径
        # 简单切分，实际可使用 sklearn 的 train_test_split 做分层抽样
        from sklearn.model_selection import train_test_split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        train_dataset = FatigueImageDataset(train_paths, train_labels, transform=data_transforms['train'])
        val_dataset = FatigueImageDataset(val_paths, val_labels, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. 构建模型、损失函数和优化器
    model = create_model(num_classes=3)
    model = model.to(device)

    # 针对类别不平衡计算权重
    class_counts = np.bincount(train_labels, minlength=3)
    total_samples = len(train_labels)
    # class_weights = total_samples / (3.0 * class_counts) 避免除零
    class_weights = np.zeros(3, dtype=np.float32)
    for i in range(3):
        if class_counts[i] > 0:
            class_weights[i] = total_samples / (3.0 * class_counts[i])
        else:
            class_weights[i] = 1.0
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"类别权重: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    # 仅训练最后一层分类器，可以加快速度和防止过拟合
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 4. 开始训练
    start_time = time.time()
    best_model = train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device)
    
    time_elapsed = time.time() - start_time
    print(f"训练完成，总耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # 5. 保存模型
    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存完整模型以便于直接 load，不需要再定义模型结构类
    torch.save(best_model, out_path)
    print(f"CNN 模型已成功保存到: {out_path}")


if __name__ == "__main__":
    main()