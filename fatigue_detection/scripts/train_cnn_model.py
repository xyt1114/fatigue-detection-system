import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from detection.utils.dataset_loader import DatasetLoader


CLASS_MAP = {"awake": 0, "mild": 1, "severe": 2}
CLASS_NAMES = ["awake", "mild", "severe"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FatigueImageDataset(Dataset):
    def __init__(self, image_paths, labels, image_size, transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.image_size = int(image_size)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if "|" in path:
            path = path.split("|", 1)[0]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size), color="black")
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[idx])


def create_model(num_classes=3):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_transforms(img_size):
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def set_trainable_layers(model, fine_tune_all=False):
    for param in model.parameters():
        param.requires_grad = bool(fine_tune_all)
    for param in model.classifier.parameters():
        param.requires_grad = True


def collect_paths_from_split(split_dir):
    paths = []
    labels = []
    for class_name, class_id in CLASS_MAP.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(image_path))
                labels.append(class_id)
    return paths, labels


def discover_split_dirs(dataset_dir):
    dataset_dir = Path(dataset_dir)
    candidates = [dataset_dir / "splits", dataset_dir]
    for base in candidates:
        train_dir = base / "train"
        valid_dir = base / "valid"
        if train_dir.exists() and valid_dir.exists():
            return {
                "train": train_dir,
                "valid": valid_dir,
                "test": base / "test" if (base / "test").exists() else None,
            }
    return None


def load_split_datasets(dataset_dir, img_size):
    split_dirs = discover_split_dirs(dataset_dir)
    transforms_map = build_transforms(img_size)
    if split_dirs:
        train_paths, train_labels = collect_paths_from_split(split_dirs["train"])
        val_paths, val_labels = collect_paths_from_split(split_dirs["valid"])
        test_dataset = None
        if split_dirs["test"] is not None:
            test_paths, test_labels = collect_paths_from_split(split_dirs["test"])
            if test_paths:
                test_dataset = FatigueImageDataset(
                    test_paths,
                    test_labels,
                    image_size=img_size,
                    transform=transforms_map["eval"],
                )
        if train_paths and val_paths:
            return (
                FatigueImageDataset(train_paths, train_labels, image_size=img_size, transform=transforms_map["train"]),
                FatigueImageDataset(val_paths, val_labels, image_size=img_size, transform=transforms_map["eval"]),
                test_dataset,
                np.array(train_labels, dtype=np.int64),
                "official_split",
            )

    loader = DatasetLoader(dataset_root=dataset_dir)
    report = loader.validate_dataset()
    if not report["is_valid"]:
        raise RuntimeError(f"数据集不完整或样本不足: {report}")

    all_paths = []
    all_labels = []
    for class_name, class_id in loader.class_map.items():
        class_dir = loader.dataset_root / class_name
        if not class_dir.exists():
            class_dir = Path(dataset_dir) / class_name
        if not class_dir.exists():
            continue
        for media_path in loader._iter_media_paths(class_dir):
            if loader._is_image_file(media_path):
                all_paths.append(str(media_path))
                all_labels.append(class_id)

    if not all_paths:
        raise RuntimeError("未找到可用于分类训练的图片，请先运行数据转换脚本。")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels,
    )
    return (
        FatigueImageDataset(train_paths, train_labels, image_size=img_size, transform=transforms_map["train"]),
        FatigueImageDataset(val_paths, val_labels, image_size=img_size, transform=transforms_map["eval"]),
        None,
        np.array(train_labels, dtype=np.int64),
        "random_split",
    )


def evaluate_model(model, dataloader, criterion, device, desc):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    pbar = tqdm(dataloader, desc=desc)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data).item()
            total += batch_size
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return running_loss / max(1, total), running_corrects / max(1, total)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, early_stop_patience=6):
    best_acc = 0.0
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        pbar = tqdm(train_loader, desc="train")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data).item()
            total += batch_size
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / max(1, total)
        train_acc = running_corrects / max(1, total)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, "val")
        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print()

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print(f"Best val Acc: {best_acc:.4f} (epoch {best_epoch})")
    model.load_state_dict(best_model_state)
    return model, best_acc


def build_optimizer(model, learning_rate, weight_decay):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)


def load_checkpoint_if_needed(model, output_path, device):
    if not output_path.exists():
        return 0.0
    loaded = torch.load(output_path, map_location=device, weights_only=False)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        model.load_state_dict(loaded["model_state_dict"])
        return float(loaded.get("best_val_acc", 0.0))
    if isinstance(loaded, dict) and all(torch.is_tensor(v) for v in loaded.values()):
        model.load_state_dict(loaded)
        return 0.0
    if hasattr(loaded, "state_dict"):
        model.load_state_dict(loaded.state_dict())
        return 0.0
    raise RuntimeError(f"无法识别的 checkpoint 格式: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="训练疲劳检测 CNN 模型")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(BASE_DIR / "dataset_yolo"),
        help="分类数据集根目录，优先支持 splits/train|valid|test 结构",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(BASE_DIR / "models" / "fatigue_classifier_cnn.pt"),
        help="模型保存路径",
    )
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数，建议 25-40")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--img-size", type=int, default=224, help="输入图片尺寸")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--early-stop-patience", type=int, default=6, help="早停耐心轮数，0 表示关闭")
    parser.add_argument("--resume", action="store_true", help="是否从现有 checkpoint 继续训练")
    parser.add_argument("--fine-tune-all", dest="fine_tune_all", action="store_true", help="是否解冻骨干网络做全量微调")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"正在读取数据集: {args.dataset_dir}")

    train_dataset, val_dataset, test_dataset, train_labels, split_mode = load_split_datasets(
        args.dataset_dir,
        args.img_size,
    )
    print(f"数据集模式: {split_mode}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    if test_dataset is not None:
        print(f"测试集样本数: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(num_classes=len(CLASS_NAMES))
    set_trainable_layers(model, fine_tune_all=args.fine_tune_all)
    model = model.to(device)

    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    previous_best_acc = 0.0
    if args.resume and out_path.exists():
        previous_best_acc = load_checkpoint_if_needed(model, out_path, device)
        print(f"已加载已有 checkpoint: {out_path}")
        print(f"历史最佳验证准确率: {previous_best_acc:.4f}")

    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    total_samples = int(np.sum(class_counts))
    class_weights = np.ones(len(CLASS_NAMES), dtype=np.float32)
    for idx, count in enumerate(class_counts):
        if count > 0:
            class_weights[idx] = total_samples / (len(CLASS_NAMES) * float(count))
    print(f"类别分布: {class_counts.tolist()}")
    print(f"类别权重: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    learning_rate = args.learning_rate if args.fine_tune_all else max(args.learning_rate, 1e-3)
    optimizer = build_optimizer(model, learning_rate=learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    start_time = time.time()
    best_model, best_val_acc = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        args.epochs,
        device,
        args.early_stop_patience,
    )
    elapsed = time.time() - start_time
    print(f"训练完成，总耗时 {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    if test_loader is not None:
        test_loss, test_acc = evaluate_model(best_model, test_loader, criterion, device, "test")
        print(f"test  Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    checkpoint = {
        "model_state_dict": best_model.state_dict(),
        "class_names": CLASS_NAMES,
        "input_size": [args.img_size, args.img_size],
        "best_val_acc": max(previous_best_acc, float(best_val_acc)),
        "fine_tune_all": bool(args.fine_tune_all),
        "early_stop_patience": int(args.early_stop_patience),
    }
    torch.save(checkpoint, out_path)
    print(f"CNN 模型已成功保存到: {out_path}")


if __name__ == "__main__":
    main()
