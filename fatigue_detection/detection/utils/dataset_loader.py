import os
from pathlib import Path

import cv2
import numpy as np

try:
    from django.conf import settings
    from django.core.exceptions import ImproperlyConfigured
except Exception:
    settings = None
    ImproperlyConfigured = Exception


class DatasetLoader:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}

    def __init__(self, dataset_root=None):
        self.class_map = {"awake": 0, "mild": 1, "severe": 2}
        self.class_names = ["awake", "mild", "severe"]
        self.dataset_root = self._resolve_dataset_root(dataset_root)

    def _resolve_dataset_root(self, dataset_root):
        if dataset_root:
            base = Path(dataset_root)
        elif os.getenv("DATASET_ROOT"):
            base = Path(os.getenv("DATASET_ROOT"))
        elif settings is not None:
            try:
                if settings.configured and hasattr(settings, "DATASET_ROOT"):
                    base = Path(settings.DATASET_ROOT)
                else:
                    base = Path("dataset")
            except ImproperlyConfigured:
                base = Path("dataset")
        else:
            base = Path("dataset")
        if base.name == "raw":
            return base
        return base / "raw"

    def _is_image_file(self, file_path):
        return file_path.suffix.lower() in self.IMAGE_EXTENSIONS

    def _is_video_file(self, file_path):
        return file_path.suffix.lower() in self.VIDEO_EXTENSIONS

    def _iter_media_paths(self, class_dir):
        return sorted(
            [
                p
                for p in class_dir.rglob("*")
                if p.is_file() and (self._is_image_file(p) or self._is_video_file(p))
            ]
        )

    def _safe_read_image(self, img_path, img_size):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception:
            return None

    def _extract_video_frames(self, video_path, img_size, frame_stride=15, max_frames=20):
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return frames
        try:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % frame_stride == 0:
                    frame = cv2.resize(frame, img_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    if len(frames) >= max_frames:
                        break
                index += 1
        except Exception:
            return frames
        finally:
            cap.release()
        return frames

    def _augment_image(self, image):
        aug_images = []
        aug_images.append(cv2.flip(image, 1))
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-15, 15)
        bright = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        aug_images.append(bright)
        return aug_images

    def get_class_distribution(self):
        distribution = {}
        for class_name in self.class_map.keys():
            class_dir = self.dataset_root / class_name
            if class_dir.exists():
                count = len(self._iter_media_paths(class_dir))
                distribution[class_name] = count
            else:
                distribution[class_name] = 0
        return distribution

    def validate_dataset(self, min_samples_per_class=1):
        distribution = self.get_class_distribution()
        missing_classes = [k for k, v in distribution.items() if v == 0]
        insufficient_classes = [k for k, v in distribution.items() if v < min_samples_per_class]
        is_valid = len(missing_classes) == 0 and len(insufficient_classes) == 0
        return {
            "is_valid": is_valid,
            "dataset_root": str(self.dataset_root),
            "distribution": distribution,
            "missing_classes": missing_classes,
            "insufficient_classes": insufficient_classes,
            "min_samples_per_class": min_samples_per_class,
        }

    def load_dataset(self, img_size=(224, 224), augment=False, frame_stride=15, max_frames_per_video=20):
        images = []
        labels = []
        paths = []

        for class_name, class_id in self.class_map.items():
            class_dir = self.dataset_root / class_name
            if not class_dir.exists():
                continue

            for media_path in self._iter_media_paths(class_dir):
                if self._is_image_file(media_path):
                    media_images = [self._safe_read_image(media_path, img_size)]
                else:
                    media_images = self._extract_video_frames(
                        media_path,
                        img_size=img_size,
                        frame_stride=frame_stride,
                        max_frames=max_frames_per_video,
                    )

                for idx, img in enumerate(media_images):
                    if img is None:
                        continue
                    images.append(img)
                    labels.append(class_id)
                    if self._is_image_file(media_path):
                        paths.append(str(media_path))
                    else:
                        paths.append(f"{media_path}|frame={idx}")

                    if augment:
                        for aug_img in self._augment_image(img):
                            images.append(aug_img)
                            labels.append(class_id)
                            paths.append(f"{media_path}|frame={idx}|aug")

        return np.array(images), np.array(labels), paths

    def split_train_val(self, images, labels, val_ratio=0.2, random_state=42):
        from sklearn.model_selection import train_test_split

        return train_test_split(
            images,
            labels,
            test_size=val_ratio,
            random_state=random_state,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )
