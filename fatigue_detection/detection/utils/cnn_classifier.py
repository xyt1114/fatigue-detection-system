import logging
from pathlib import Path

import cv2
import numpy as np

_LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _build_mobilenet_v3_small(num_classes):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


class CNNFatigueClassifier:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._model = None
        self._input_size = (224, 224)
        self._classes = ["awake", "mild", "severe"]
        self._label_to_status = {
            "awake": "alert",
            "mild": "fatigue",
            "severe": "severe_fatigue",
        }
        self._device = None
        self._transform = None

    def is_ready(self):
        return TORCH_AVAILABLE and self.model_path.exists()

    def _ensure_model_loaded(self):
        if self._model is not None:
            return True
        if not TORCH_AVAILABLE:
            _LOGGER.warning("PyTorch 未安装，CNN 分类器无法加载")
            return False
        if not self.model_path.exists():
            _LOGGER.warning(f"CNN 模型文件不存在: {self.model_path}")
            return False

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded = torch.load(self.model_path, map_location=self._device, weights_only=False)

            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                classes = loaded.get("class_names")
                if isinstance(classes, (list, tuple)) and classes:
                    self._classes = [str(item) for item in classes]
                input_size = loaded.get("input_size")
                if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
                    self._input_size = (int(input_size[0]), int(input_size[1]))
                self._model = _build_mobilenet_v3_small(num_classes=len(self._classes))
                self._model.load_state_dict(loaded["model_state_dict"])
            elif isinstance(loaded, dict) and all(torch.is_tensor(v) for v in loaded.values()):
                self._model = _build_mobilenet_v3_small(num_classes=len(self._classes))
                self._model.load_state_dict(loaded)
            else:
                self._model = loaded

            self._model = self._model.to(self._device)
            self._model.eval()

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self._input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return True
        except Exception as e:
            _LOGGER.error(f"CNN 模型加载失败: {e}", exc_info=True)
            return False

    def predict(self, frame_bgr):
        if frame_bgr is None:
            return None
        if not self._ensure_model_loaded():
            return None

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = self._transform(rgb).unsqueeze(0).to(self._device)

            with torch.no_grad():
                outputs = self._model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]

            confidence, class_idx = torch.max(probs, dim=0)
            class_idx = int(class_idx.item())
            confidence = float(confidence.item())

            label = self._classes[class_idx] if class_idx < len(self._classes) else "awake"
            status = self._label_to_status.get(label, "alert")
            score_map = {"alert": 0, "fatigue": 55, "severe_fatigue": 85}

            return {
                "status": status,
                "score": score_map.get(status, 0),
                "reasons": [f"cnn_{label}"],
                "confidence": round(confidence, 4),
                "raw_label": label,
            }
        except Exception as e:
            _LOGGER.error(f"CNN 推理异常: {e}", exc_info=True)
            return None
