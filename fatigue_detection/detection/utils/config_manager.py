from threading import Lock

from django.conf import settings


class ConfigManager:
    _instance = None
    _instance_lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._config_lock = Lock()
                    cls._instance._default_config = {
                        "ear_threshold": float(settings.FATIGUE_CONFIG["EAR_THRESHOLD"]),
                        "mar_threshold": float(settings.FATIGUE_CONFIG["MAR_THRESHOLD"]),
                        "pitch_threshold": float(settings.FATIGUE_CONFIG["PITCH_THRESHOLD"]),
                        "warning_frame_count": int(settings.FATIGUE_CONFIG["WARNING_FRAME_COUNT"]),
                        "emergency_frame_count": int(settings.FATIGUE_CONFIG["EMERGENCY_FRAME_COUNT"]),
                        "fps_hint": float(settings.FATIGUE_CONFIG.get("FPS_HINT", 5.0)),
                        "ml_fatigue_min_frames": int(settings.FATIGUE_CONFIG.get("ML_FATIGUE_MIN_FRAMES", 2)),
                        "ml_severe_min_frames": int(settings.FATIGUE_CONFIG.get("ML_SEVERE_MIN_FRAMES", 3)),
                        "blink_max_duration_sec": float(settings.FATIGUE_CONFIG.get("BLINK_MAX_DURATION_SEC", 0.35)),
                        "yawn_warning_sec": float(settings.FATIGUE_CONFIG.get("YAWN_WARNING_SEC", 0.6)),
                        "yawn_emergency_sec": float(settings.FATIGUE_CONFIG.get("YAWN_EMERGENCY_SEC", 1.2)),
                    }
                    cls._instance._config = cls._instance._default_config.copy()
        return cls._instance

    def _validate_config(self, config):
        if not 0.1 <= float(config["ear_threshold"]) <= 0.4:
            raise ValueError("ear_threshold 超出范围(0.1-0.4)")
        if not 0.3 <= float(config["mar_threshold"]) <= 0.8:
            raise ValueError("mar_threshold 超出范围(0.3-0.8)")
        if not 10 <= float(config["pitch_threshold"]) <= 60:
            raise ValueError("pitch_threshold 超出范围(10-60)")
        if not 1 <= int(config["warning_frame_count"]) <= 30:
            raise ValueError("warning_frame_count 超出范围(1-30)")
        if not 1 <= int(config["emergency_frame_count"]) <= 60:
            raise ValueError("emergency_frame_count 超出范围(1-60)")
        if int(config["emergency_frame_count"]) < int(config["warning_frame_count"]):
            raise ValueError("emergency_frame_count 不能小于 warning_frame_count")
        if not 1 <= float(config["fps_hint"]) <= 60:
            raise ValueError("fps_hint 超出范围(1-60)")
        if not 1 <= int(config["ml_fatigue_min_frames"]) <= 30:
            raise ValueError("ml_fatigue_min_frames 超出范围(1-30)")
        if not 1 <= int(config["ml_severe_min_frames"]) <= 60:
            raise ValueError("ml_severe_min_frames 超出范围(1-60)")
        if int(config["ml_severe_min_frames"]) < int(config["ml_fatigue_min_frames"]):
            raise ValueError("ml_severe_min_frames 不能小于 ml_fatigue_min_frames")
        if not 0.1 <= float(config["blink_max_duration_sec"]) <= 1.5:
            raise ValueError("blink_max_duration_sec 超出范围(0.1-1.5)")
        if not 0.2 <= float(config["yawn_warning_sec"]) <= 3.0:
            raise ValueError("yawn_warning_sec 超出范围(0.2-3.0)")
        if not 0.3 <= float(config["yawn_emergency_sec"]) <= 5.0:
            raise ValueError("yawn_emergency_sec 超出范围(0.3-5.0)")
        if float(config["yawn_emergency_sec"]) < float(config["yawn_warning_sec"]):
            raise ValueError("yawn_emergency_sec 不能小于 yawn_warning_sec")

    def get_config(self):
        with self._config_lock:
            return self._config.copy()

    def update_config(self, new_config):
        with self._config_lock:
            merged = self._config.copy()
            merged.update(new_config or {})
            self._validate_config(merged)
            self._config = {
                "ear_threshold": float(merged["ear_threshold"]),
                "mar_threshold": float(merged["mar_threshold"]),
                "pitch_threshold": float(merged["pitch_threshold"]),
                "warning_frame_count": int(merged["warning_frame_count"]),
                "emergency_frame_count": int(merged["emergency_frame_count"]),
                "fps_hint": float(merged["fps_hint"]),
                "ml_fatigue_min_frames": int(merged["ml_fatigue_min_frames"]),
                "ml_severe_min_frames": int(merged["ml_severe_min_frames"]),
                "blink_max_duration_sec": float(merged["blink_max_duration_sec"]),
                "yawn_warning_sec": float(merged["yawn_warning_sec"]),
                "yawn_emergency_sec": float(merged["yawn_emergency_sec"]),
            }
            return self._config.copy()

    def reset_config(self):
        with self._config_lock:
            self._config = self._default_config.copy()
            return self._config.copy()
