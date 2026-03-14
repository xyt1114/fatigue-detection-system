class WarningSystem:
    """
    预警状态机。

    按连续帧策略将疲劳状态映射为 normal / warning / emergency。
    """

    def __init__(self, warning_frame_count=3, emergency_frame_count=5):
        """
        初始化预警计数器与阈值。

        Args:
            warning_frame_count: 连续 fatigue 触发 warning 的帧数。
            emergency_frame_count: 连续 severe_fatigue 触发 emergency 的帧数。
        """
        self.warning_frame_count = warning_frame_count
        self.emergency_frame_count = emergency_frame_count
        self.fatigue_frame_count = 0
        self.severe_frame_count = 0
        self.warning_state = "normal"

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
        status = fatigue_status.get("status") if isinstance(fatigue_status, dict) else fatigue_status
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
