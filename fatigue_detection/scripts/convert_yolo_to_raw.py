from argparse import ArgumentParser
from pathlib import Path
import shutil


CLASS_MAP = {
    0: "mild",
    1: "awake",
    2: "severe",
}

PRIORITY = {"awake": 0, "mild": 1, "severe": 2}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def decide_class(label_file):
    if not label_file.exists():
        return None
    lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    picked = None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            cls_id = int(parts[0])
        except Exception:
            continue
        cls_name = CLASS_MAP.get(cls_id)
        if cls_name is None:
            continue
        if picked is None or PRIORITY[cls_name] > PRIORITY[picked]:
            picked = cls_name
    return picked


def convert_dataset(source_dir, output_root):
    output_raw = output_root / "raw"
    for class_name in ("awake", "mild", "severe"):
        (output_raw / class_name).mkdir(parents=True, exist_ok=True)

    stats = {"awake": 0, "mild": 0, "severe": 0, "skipped": 0}
    for split in ("train", "valid", "test"):
        image_dir = source_dir / split / "images"
        label_dir = source_dir / split / "labels"
        if not image_dir.exists():
            continue
        for image_file in image_dir.glob("*.*"):
            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_file = label_dir / f"{image_file.stem}.txt"
            class_name = decide_class(label_file)
            if class_name is None:
                stats["skipped"] += 1
                continue
            target_file = output_raw / class_name / f"{split}_{image_file.name}"
            shutil.copy2(image_file, target_file)
            stats[class_name] += 1
    return stats, output_raw


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=str,
        default=r"C:\Users\86186\Documents\trae_projects\lunwen\fatigue_detection\dataset\drowsiness driver.v1i.yolov8",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\Users\86186\Documents\trae_projects\lunwen\fatigue_detection\dataset_yolo",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if not source_dir.exists():
        raise RuntimeError(f"源目录不存在: {source_dir}")

    stats, output_raw = convert_dataset(source_dir, output_dir)
    print("转换完成")
    print(f"输出目录: {output_raw}")
    print(f"样本统计: {stats}")


if __name__ == "__main__":
    main()
