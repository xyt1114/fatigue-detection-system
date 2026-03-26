from pathlib import Path

from django.conf import settings

from .config_manager import ConfigManager
from .face_detector import get_face_detector
from .fatigue_classifier import FatigueClassifier
from .feature_extractor import get_feature_extractor
from .ml_classifier import MLFatigueClassifier
from .warning_system import WarningSystem

_face_detector = None
_static_face_detector = None
_feature_extractor = None
_classifier = None
_warning_system = None
_ml_classifier = None


def get_detector_instances():
    global _face_detector, _static_face_detector, _feature_extractor, _classifier, _warning_system
    config = ConfigManager().get_config()
    if _face_detector is None:
        _face_detector = get_face_detector(static_image_mode=False, max_num_faces=1)
    if _static_face_detector is None:
        _static_face_detector = get_face_detector(static_image_mode=True, max_num_faces=1)
    if _feature_extractor is None:
        _feature_extractor = get_feature_extractor()
    if _classifier is None:
        _classifier = FatigueClassifier(
            ear_threshold=config["ear_threshold"],
            mar_threshold=config["mar_threshold"],
            pitch_threshold=config["pitch_threshold"],
        )
    if _warning_system is None:
        _warning_system = WarningSystem(
            warning_frame_count=config["warning_frame_count"],
            emergency_frame_count=config["emergency_frame_count"],
            fps_hint=config["fps_hint"],
            ml_fatigue_min_frames=config["ml_fatigue_min_frames"],
            ml_severe_min_frames=config["ml_severe_min_frames"],
            blink_max_duration_sec=config["blink_max_duration_sec"],
            yawn_warning_sec=config["yawn_warning_sec"],
            yawn_emergency_sec=config["yawn_emergency_sec"],
        )
    return _face_detector, _static_face_detector, _feature_extractor, _classifier, _warning_system


def get_ml_classifier():
    global _ml_classifier
    if _ml_classifier is None:
        model_path = Path(getattr(settings, "ML_MODEL_PATH", ""))
        _ml_classifier = MLFatigueClassifier(model_path=model_path)
    return _ml_classifier


def rebuild_runtime_instances(config):
    global _classifier, _warning_system
    _classifier = FatigueClassifier(
        ear_threshold=config["ear_threshold"],
        mar_threshold=config["mar_threshold"],
        pitch_threshold=config["pitch_threshold"],
    )
    _warning_system = WarningSystem(
        warning_frame_count=config["warning_frame_count"],
        emergency_frame_count=config["emergency_frame_count"],
        fps_hint=config["fps_hint"],
        ml_fatigue_min_frames=config["ml_fatigue_min_frames"],
        ml_severe_min_frames=config["ml_severe_min_frames"],
        blink_max_duration_sec=config["blink_max_duration_sec"],
        yawn_warning_sec=config["yawn_warning_sec"],
        yawn_emergency_sec=config["yawn_emergency_sec"],
    )
    return _classifier, _warning_system
