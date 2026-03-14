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
            }
            return self._config.copy()

    def reset_config(self):
        with self._config_lock:
            self._config = self._default_config.copy()
            return self._config.copy()
