from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from detection.utils.dataset_loader import DatasetLoader


def main():
    start_at = time.perf_counter()
    print("[1/6] 初始化数据加载器...")
    loader = DatasetLoader(dataset_root=BASE_DIR / "dataset")

    print("[2/6] 校验数据集...")
    report = loader.validate_dataset(min_samples_per_class=5)
    print(f"数据集路径: {report['dataset_root']}")
    print(f"类别分布: {report['distribution']}")
    if not report["is_valid"]:
        raise RuntimeError(f"数据集校验未通过: {report}")

    print("[3/6] 加载与增强样本（该步骤可能耗时）...")
    load_at = time.perf_counter()
    images, labels, _ = loader.load_dataset(img_size=(128, 128), augment=True)
    if len(images) == 0:
        raise RuntimeError("未加载到有效图片")
    print(f"样本加载完成: {len(images)} 条, 用时 {time.perf_counter() - load_at:.2f}s")

    print("[4/6] 构建训练/验证集...")
    X = images.reshape(len(images), -1).astype(np.float32) / 255.0
    X_train, X_val, y_train, y_val = loader.split_train_val(X, labels, val_ratio=0.2)
    print(f"训练集: {len(X_train)} 条, 验证集: {len(X_val)} 条")

    print("[5/6] 训练模型...")
    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500, multi_class="multinomial", verbose=1)),
        ]
    )
    fit_at = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"模型训练完成, 用时 {time.perf_counter() - fit_at:.2f}s")

    print("[6/6] 评估并保存模型...")
    preds = model.predict(X_val)
    print(classification_report(y_val, preds, target_names=loader.class_names))

    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fatigue_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")
    print(f"训练流程结束，总耗时 {time.perf_counter() - start_at:.2f}s")


if __name__ == "__main__":
    main()
