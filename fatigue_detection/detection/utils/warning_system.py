class WarningSystem:
    """
    预警状态机。

    按连续帧策略将疲劳状态映射为 normal / warning / emergency。
    """

    def __init__(
        self,
        warning_frame_count=3,
        emergency_frame_count=5,
        fps_hint=5.0,
        ml_fatigue_min_frames=2,
        ml_severe_min_frames=3,
        blink_max_duration_sec=0.35,
        yawn_warning_sec=0.6,
        yawn_emergency_sec=1.2,
    ):
        """
        初始化预警计数器与阈值。

        Args:
            warning_frame_count: 连续 fatigue 触发 warning 的帧数。
            emergency_frame_count: 连续 severe_fatigue 触发 emergency 的帧数。
        """
        self.warning_frame_count = warning_frame_count
        self.emergency_frame_count = emergency_frame_count
        self.fps_hint = float(fps_hint) if float(fps_hint) > 0 else 5.0
        self.ml_fatigue_min_frames = max(1, int(ml_fatigue_min_frames))
        self.ml_severe_min_frames = max(self.ml_fatigue_min_frames, int(ml_severe_min_frames))
        self.blink_max_duration_sec = float(blink_max_duration_sec)
        self.yawn_warning_sec = float(yawn_warning_sec)
        self.yawn_emergency_sec = float(yawn_emergency_sec)
        self.fatigue_frame_count = 0
        self.severe_frame_count = 0
        self.ml_fatigue_streak = 0
        self.ml_severe_streak = 0
        self.eye_close_streak = 0
        self.mouth_open_streak = 0
        self.warning_state = "normal"

    def _frame_duration_sec(self):
        return 1.0 / self.fps_hint

    def _extract_status_payload(self, fatigue_status):
        if isinstance(fatigue_status, dict):
            return (
                fatigue_status.get("status", "alert"),
                str(fatigue_status.get("inference_mode", "rule") or "rule").lower(),
                float(fatigue_status.get("ear", 0.0) or 0.0),
                float(fatigue_status.get("mar", 0.0) or 0.0),
            )
        return fatigue_status, "rule", 0.0, 0.0

    def _apply_ml_temporal_gate(self, status, inference_mode, ear, mar):
        if inference_mode != "ml":
            self.ml_fatigue_streak = 0
            self.ml_severe_streak = 0
            self.eye_close_streak = 0
            self.mouth_open_streak = 0
            return status

        frame_sec = self._frame_duration_sec()
        if ear > 0 and ear < 0.22:
            self.eye_close_streak += 1
        else:
            self.eye_close_streak = 0
        if mar > 0.72:
            self.mouth_open_streak += 1
        else:
            self.mouth_open_streak = 0

        eye_close_sec = self.eye_close_streak * frame_sec
        mouth_open_sec = self.mouth_open_streak * frame_sec

        if status == "severe_fatigue":
            self.ml_fatigue_streak += 1
            self.ml_severe_streak += 1
        elif status == "fatigue":
            self.ml_fatigue_streak += 1
            self.ml_severe_streak = 0
        else:
            self.ml_fatigue_streak = 0
            self.ml_severe_streak = 0
            return "alert"

        if eye_close_sec > 0 and eye_close_sec < self.blink_max_duration_sec:
            return "alert"
        if mouth_open_sec > 0 and mouth_open_sec < self.yawn_warning_sec:
            return "alert"

        if status == "severe_fatigue":
            if mouth_open_sec >= self.yawn_warning_sec and mouth_open_sec < self.yawn_emergency_sec:
                return "fatigue"
            if self.ml_severe_streak < self.ml_severe_min_frames:
                if self.ml_fatigue_streak >= self.ml_fatigue_min_frames:
                    return "fatigue"
                return "alert"
        if status == "fatigue" and self.ml_fatigue_streak < self.ml_fatigue_min_frames:
            return "alert"
        return status

    def update(self, fatigue_status):
        """
        更新预警状态。

        Args:
            fatigue_status: 当前帧疲劳状态字符串或包含 status 字段的字典。

        Returns:
            {
                "warning_level": "normal" | "warning" | "emergency",
                "frame_count": int,
                "trigger_alert": bool
            }
        """
        status, inference_mode, ear, mar = self._extract_status_payload(fatigue_status)
        status = self._apply_ml_temporal_gate(status, inference_mode, ear, mar)
        previous_state = self.warning_state
        if status == "severe_fatigue":
            self.fatigue_frame_count += 1
            self.severe_frame_count += 1
        elif status == "fatigue":
            self.fatigue_frame_count += 1
            self.severe_frame_count = 0
        else:
            self.reset()
            return {"warning_level": "normal", "frame_count": 0, "trigger_alert": previous_state != "normal"}

        if self.severe_frame_count >= self.emergency_frame_count:
            self.warning_state = "emergency"
            frame_count = self.severe_frame_count
        elif self.fatigue_frame_count >= self.warning_frame_count:
            self.warning_state = "warning"
            frame_count = self.fatigue_frame_count
        else:
            self.warning_state = "normal"
            frame_count = self.fatigue_frame_count
        trigger_alert = self.warning_state != previous_state and self.warning_state in {"warning", "emergency"}
        return {
            "effective_status": status,
            "warning_level": self.warning_state,
            "frame_count": int(frame_count),
            "trigger_alert": trigger_alert,
        }

    def reset(self):
        """
        重置所有计数器和状态。
        """
        self.fatigue_frame_count = 0
        self.severe_frame_count = 0
        self.ml_fatigue_streak = 0
        self.ml_severe_streak = 0
        self.eye_close_streak = 0
        self.mouth_open_streak = 0
        self.warning_state = "normal"

    def process_sequence(self, fatigue_status_sequence):
        """
        批量处理视频帧状态序列。

        Args:
            fatigue_status_sequence: 每帧疲劳状态列表。

        Returns:
            每帧预警输出结果列表。
        """
        return [self.update(status) for status in fatigue_status_sequence]

    def usage_example(self):
        """
        演示如何调用预警系统。

        Returns:
            示例状态流的预警输出。
        """
        demo_sequence = ["alert", "fatigue", "fatigue", "fatigue", "severe_fatigue", "severe_fatigue"]
        self.reset()
        return self.process_sequence(demo_sequence)
