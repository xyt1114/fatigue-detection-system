from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from detection.utils.dataset_loader import DatasetLoader


def _split_media_by_class(loader, val_ratio=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    train_media = []
    val_media = []
    for class_name, class_id in loader.class_map.items():
        class_dir = loader.dataset_root / class_name
        media_list = loader._iter_media_paths(class_dir) if class_dir.exists() else []
        if len(media_list) < 2:
            raise RuntimeError(f"类别 {class_name} 样本不足 2 个媒体文件，无法按媒体分组切分")
        perm = rng.permutation(len(media_list))
        val_count = int(round(len(media_list) * val_ratio))
        val_count = max(1, min(len(media_list) - 1, val_count))
        val_idx = set(perm[:val_count].tolist())
        for idx, media_path in enumerate(media_list):
            item = (media_path, class_id, class_name)
            if idx in val_idx:
                val_media.append(item)
            else:
                train_media.append(item)
    return train_media, val_media


def _load_samples(loader, media_items, img_size=(128, 128), augment=False, frame_stride=15, max_frames_per_video=20):
    images = []
    labels = []
    groups = []
    for media_path, class_id, _ in media_items:
        if loader._is_image_file(media_path):
            frames = [loader._safe_read_image(media_path, img_size)]
        else:
            frames = loader._extract_video_frames(
                media_path,
                img_size=img_size,
                frame_stride=frame_stride,
                max_frames=max_frames_per_video,
            )
        for frame in frames:
            if frame is None:
                continue
            images.append(frame)
            labels.append(class_id)
            groups.append(str(media_path))
            if augment:
                for aug_img in loader._augment_image(frame):
                    images.append(aug_img)
                    labels.append(class_id)
                    groups.append(f"{media_path}|aug")
    return np.array(images), np.array(labels), groups


def _print_media_distribution(media_items, split_name):
    stats = {}
    for _, _, class_name in media_items:
        stats[class_name] = stats.get(class_name, 0) + 1
    print(f"{split_name} 媒体分布: {stats}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(BASE_DIR / "dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--max-frames-per-video", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "models" / "fatigue_classifier_grouped.joblib"))
    args = parser.parse_args()

    start_at = time.perf_counter()
    print("[1/7] 初始化数据加载器...")
    loader = DatasetLoader(dataset_root=args.dataset_root)

    print("[2/7] 校验数据集...")
    report = loader.validate_dataset(min_samples_per_class=5)
    print(f"数据集路径: {report['dataset_root']}")
    print(f"类别分布: {report['distribution']}")
    if not report["is_valid"]:
        raise RuntimeError(f"数据集校验未通过: {report}")

    print("[3/7] 按媒体分组切分训练/验证集...")
    train_media, val_media = _split_media_by_class(
        loader,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )
    _print_media_distribution(train_media, "训练集")
    _print_media_distribution(val_media, "验证集")

    print("[4/7] 加载训练样本（含增强）...")
    load_train_at = time.perf_counter()
    train_images, y_train, train_groups = _load_samples(
        loader,
        train_media,
        img_size=(args.img_size, args.img_size),
        augment=True,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
    )
    print(
        f"训练样本数: {len(train_images)}，媒体组数: {len(set(train_groups))}，用时 {time.perf_counter() - load_train_at:.2f}s"
    )

    print("[5/7] 加载验证样本（不增强）...")
    load_val_at = time.perf_counter()
    val_images, y_val, val_groups = _load_samples(
        loader,
        val_media,
        img_size=(args.img_size, args.img_size),
        augment=False,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
    )
    print(f"验证样本数: {len(val_images)}，媒体组数: {len(set(val_groups))}，用时 {time.perf_counter() - load_val_at:.2f}s")

    if len(train_images) == 0 or len(val_images) == 0:
        raise RuntimeError("训练集或验证集为空，请检查数据与参数设置")

    print("[6/7] 训练模型...")
    X_train = train_images.reshape(len(train_images), -1).astype(np.float32) / 255.0
    X_val = val_images.reshape(len(val_images), -1).astype(np.float32) / 255.0
    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500, verbose=1)),
        ]
    )
    fit_at = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"模型训练完成, 用时 {time.perf_counter() - fit_at:.2f}s")

    print("[7/7] 评估并保存模型...")
    preds = model.predict(X_val)
    print(classification_report(y_val, preds, target_names=loader.class_names))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"模型已保存到: {output_path}")
    print(f"训练流程结束，总耗时 {time.perf_counter() - start_at:.2f}s")


if __name__ == "__main__":
    main()
