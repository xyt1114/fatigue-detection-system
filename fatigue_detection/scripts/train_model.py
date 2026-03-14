from pathlib import Path
import sys

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
    loader = DatasetLoader(dataset_root=BASE_DIR / "dataset")
    report = loader.validate_dataset(min_samples_per_class=5)
    if not report["is_valid"]:
        raise RuntimeError(f"数据集校验未通过: {report}")

    images, labels, _ = loader.load_dataset(img_size=(128, 128), augment=True)
    if len(images) == 0:
        raise RuntimeError("未加载到有效图片")

    X = images.reshape(len(images), -1).astype(np.float32) / 255.0
    X_train, X_val, y_train, y_val = loader.split_train_val(X, labels, val_ratio=0.2)

    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500, multi_class="multinomial")),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print(classification_report(y_val, preds, target_names=loader.class_names))

    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fatigue_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")


if __name__ == "__main__":
    main()
