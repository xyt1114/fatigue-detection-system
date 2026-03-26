from pathlib import Path

import cv2
import joblib
import numpy as np


class MLFatigueClassifier:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._model = None
        self._input_size = (128, 128)
        self._classes = np.array(["awake", "mild", "severe"], dtype=object)
        self._label_to_status = {
            "awake": "alert",
            "mild": "fatigue",
            "severe": "severe_fatigue",
        }

    def is_ready(self):
        return self.model_path.exists()

    def _ensure_model_loaded(self):
        if self._model is not None:
            return True
        if not self.model_path.exists():
            return False
        self._model = joblib.load(self.model_path)
        model_classes = getattr(self._model, "classes_", None)
        if model_classes is not None and len(model_classes) > 0:
            self._classes = np.array(model_classes, dtype=object)
        classifier = getattr(self._model, "named_steps", {}).get("clf")
        coef = getattr(classifier, "coef_", None)
        if coef is not None and coef.ndim == 2 and coef.shape[1] > 0:
            channels = 3
            side = int(round((coef.shape[1] / channels) ** 0.5))
            if side > 0 and side * side * channels == coef.shape[1]:
                self._input_size = (side, side)
        return True

    def _prepare_input(self, frame_bgr):
        resized = cv2.resize(frame_bgr, self._input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.reshape(1, -1).astype(np.float32) / 255.0
        return x

    def predict(self, frame_bgr):
        if frame_bgr is None:
            return None
        if not self._ensure_model_loaded():
            return None
        x = self._prepare_input(frame_bgr)
        pred = self._model.predict(x)[0]
        label = str(pred)
        status = self._label_to_status.get(label, "alert")
        score_map = {"alert": 0, "fatigue": 55, "severe_fatigue": 85}
        confidence = 0.0
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(x)[0]
            cls_index = 0
            for idx, cls_name in enumerate(self._classes):
                if str(cls_name) == label:
                    cls_index = idx
                    break
            confidence = float(probs[cls_index])
        return {
            "status": status,
            "score": score_map.get(status, 0),
            "reasons": [f"ml_{label}"],
            "confidence": round(confidence, 4),
            "raw_label": label,
        }
