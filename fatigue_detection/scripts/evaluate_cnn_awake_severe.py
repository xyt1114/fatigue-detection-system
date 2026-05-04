import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from detection.utils.cnn_classifier import _build_mobilenet_v3_small

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_CLASSES = ("awake", "severe")
PRED_TO_BINARY = {
    "awake": "awake",
    "mild": "severe",
    "severe": "severe",
}


class EvalImageDataset(Dataset):
    def __init__(self, samples, input_size):
        self.samples = list(samples)
        self.transform = transforms.Compose(
            [
                transforms.Resize(tuple(input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, gt_label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), gt_label, str(image_path)


def collect_samples(dataset_root: Path, split: str):
    if split == "raw":
        eval_root = dataset_root / "raw"
    else:
        eval_root = dataset_root / "splits" / split
    if not eval_root.exists():
        raise FileNotFoundError(f"评估目录不存在: {eval_root}")

    samples = []
    for gt_label in GT_CLASSES:
        class_dir = eval_root / gt_label
        if not class_dir.exists():
            raise FileNotFoundError(f"缺少类别目录: {class_dir}")
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, gt_label))
    return eval_root, samples


def load_model(model_path: Path, device):
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    class_names = ["awake", "mild", "severe"]
    input_size = (224, 224)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        classes = loaded.get("class_names")
        if isinstance(classes, (list, tuple)) and classes:
            class_names = [str(item) for item in classes]
        saved_input_size = loaded.get("input_size")
        if isinstance(saved_input_size, (list, tuple)) and len(saved_input_size) == 2:
            input_size = (int(saved_input_size[0]), int(saved_input_size[1]))
        model = _build_mobilenet_v3_small(num_classes=len(class_names))
        model.load_state_dict(loaded["model_state_dict"])
    else:
        raise RuntimeError(f"无法识别的模型格式: {model_path}")
    model = model.to(device)
    model.eval()
    return model, class_names, input_size


def main():
    parser = argparse.ArgumentParser(description="直接评估 CNN 模型的 awake/severe 二分类指标")
    parser.add_argument("--dataset-root", type=str, default=str(BASE_DIR / "dataset_yolo"))
    parser.add_argument("--split", type=str, default="raw", choices=["raw", "train", "valid", "test"])
    parser.add_argument("--model-path", type=str, default=str(BASE_DIR / "models" / "fatigue_classifier_cnn.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(BASE_DIR / "logs" / "eval_cnn_awake_severe.csv"),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_root, samples = collect_samples(dataset_root, args.split)
    model, class_names, input_size = load_model(model_path, device)
    dataset = EvalImageDataset(samples, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true = []
    y_pred = []
    rows = []

    with torch.no_grad():
        for inputs, gt_labels, image_paths in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_indices = torch.argmax(outputs, dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            for gt_label, image_path, pred_idx, prob_vec in zip(gt_labels, image_paths, pred_indices, probs):
                raw_label = class_names[int(pred_idx)] if int(pred_idx) < len(class_names) else "awake"
                pred_binary = PRED_TO_BINARY.get(raw_label, "awake")
                confidence = float(np.max(prob_vec))
                y_true.append(str(gt_label))
                y_pred.append(pred_binary)
                rows.append(
                    {
                        "image_path": str(image_path),
                        "gt_label": str(gt_label),
                        "raw_label": raw_label,
                        "pred_binary": pred_binary,
                        "confidence": round(confidence, 4),
                        "correct": int(pred_binary == str(gt_label)),
                    }
                )

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["awake", "severe"],
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=["awake", "severe"])

    print("=" * 60)
    print("CNN awake / severe 二分类评估")
    print("=" * 60)
    print(f"评估目录: {eval_root}")
    print(f"模型路径: {model_path}")
    print(f"总样本数: {len(samples)}")
    print("-" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("-" * 60)
    print("混淆矩阵:")
    print("labels = ['awake', 'severe']")
    print(cm)
    print("-" * 60)
    print("分类报告:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=["awake", "severe"],
            digits=4,
            zero_division=0,
        )
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "gt_label", "raw_label", "pred_binary", "confidence", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("-" * 60)
    print(f"逐样本结果已保存: {output_csv}")


if __name__ == "__main__":
    main()
