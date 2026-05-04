import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_CLASSES = ("awake", "severe")
STATUS_TO_BINARY = {
    "alert": "awake",
    "fatigue": "severe",
    "severe_fatigue": "severe",
}


def resolve_eval_root(dataset_root: Path, split: str) -> Path:
    split_root = dataset_root / "splits" / split
    if split_root.exists():
        return split_root
    alt_root = dataset_root / split
    if alt_root.exists():
        return alt_root
    raise FileNotFoundError(f"未找到评估目录: {split_root}")


def collect_samples(eval_root: Path):
    samples = []
    for gt_label in GT_CLASSES:
        class_dir = eval_root / gt_label
        if not class_dir.exists():
            raise FileNotFoundError(f"缺少类别目录: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, gt_label))
    return samples


def init_detector(mode: str):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fatigue_detection.settings")
    os.environ["CLASSIFIER_MODE"] = mode

    import django
    django.setup()

    from detection.views import _detect_on_frame, _resolve_classifier_mode
    return _detect_on_frame, _resolve_classifier_mode


def evaluate_one_image(image_path: Path, detect_on_frame):
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "detected": False,
            "status": "read_failed",
            "pred_binary": "undetected",
            "score": "",
            "reasons": "",
            "confidence": "",
        }

    result = detect_on_frame(image, include_annotation=False, retry_static=True)
    if not result:
        return {
            "detected": False,
            "status": "undetected",
            "pred_binary": "undetected",
            "score": "",
            "reasons": "",
            "confidence": "",
        }

    classify = result.get("classify") or {}
    status = str(classify.get("status", "alert"))
    pred_binary = STATUS_TO_BINARY.get(status, "awake")
    reasons = classify.get("reasons", [])
    if isinstance(reasons, (list, tuple)):
        reasons = "|".join(map(str, reasons))
    else:
        reasons = str(reasons)

    return {
        "detected": True,
        "status": status,
        "pred_binary": pred_binary,
        "score": classify.get("score", ""),
        "reasons": reasons,
        "confidence": classify.get("confidence", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(BASE_DIR / "dataset_yolo"),
        help="dataset_yolo 根目录",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "raw"],
        help="使用哪个官方划分做评估，默认 test",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["rule", "ml", "cnn", "fusion"],
        help="评估时使用的项目推理模式，默认 fusion",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(BASE_DIR / "logs" / "eval_awake_severe_predictions.csv"),
        help="逐样本预测结果保存路径",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    eval_root = resolve_eval_root(dataset_root, args.split)
    samples = collect_samples(eval_root)

    detect_on_frame, resolve_classifier_mode = init_detector(args.mode)
    actual_mode = resolve_classifier_mode()

    y_true_valid = []
    y_pred_valid = []
    rows = []

    total_samples = len(samples)
    undetected_count = 0

    for image_path, gt_label in samples:
        pred = evaluate_one_image(image_path, detect_on_frame)
        
        pred_binary = pred["pred_binary"]
        
        if pred["detected"] and pred_binary in {"awake", "severe"}:
            y_true_valid.append(gt_label)
            y_pred_valid.append(pred_binary)
        else:
            undetected_count += 1

        rows.append(
            {
                "image_path": str(image_path),
                "gt_label": gt_label,
                "detected": int(pred["detected"]),
                "raw_status": pred["status"],
                "pred_binary": pred_binary,
                "correct": int(pred_binary == gt_label) if pred["detected"] else 0,
                "score": pred["score"],
                "confidence": pred["confidence"],
                "reasons": pred["reasons"],
            }
        )

    valid_count = len(y_true_valid)
    if valid_count > 0:
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_valid,
            y_pred_valid,
            labels=["awake", "severe"],
            average="weighted",
            zero_division=0,
        )
    else:
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    print("=" * 60)
    print("awake / severe 二分类评估 (忽略未检测到人脸的样本)")
    print("=" * 60)
    print(f"评估目录: {eval_root}")
    print(f"推理模式: {actual_mode}")
    print(f"总输入样本数: {total_samples}")
    print(f"未检测到人脸数 (已忽略): {undetected_count}")
    print(f"有效参与评估样本数: {valid_count}")
    print("-" * 60)
    print(f"有效样本准确率 (Accuracy): {accuracy:.4f}")
    print(f"加权平均精确率 (Precision): {precision:.4f}")
    print(f"加权平均召回率 (Recall): {recall:.4f}")
    print(f"加权平均 F1 值 (F1-score): {f1:.4f}")
    
    if valid_count > 0:
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=["awake", "severe"])
        print("-" * 60)
        print("混淆矩阵:")
        print("labels = ['awake', 'severe']")
        print(cm)

        print("-" * 60)
        print("分类报告:")
        print(
            classification_report(
                y_true_valid,
                y_pred_valid,
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
            fieldnames=[
                "image_path",
                "gt_label",
                "detected",
                "raw_status",
                "pred_binary",
                "correct",
                "score",
                "confidence",
                "reasons",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 60)
    print(f"逐样本结果已保存: {output_csv}")


if __name__ == "__main__":
    main()
