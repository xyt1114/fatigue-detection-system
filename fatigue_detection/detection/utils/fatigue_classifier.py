class FatigueClassifier:
    """
    疲劳状态分类器。

    根据 EAR、MAR 和头部姿态进行规则化判定，输出状态、风险分值和触发原因。
    """

    def __init__(self, ear_threshold=0.25, mar_threshold=0.6, pitch_threshold=30.0):
        """
        初始化阈值配置。

        Args:
            ear_threshold: 眼睛闭合阈值，小于该值判定 eye_closed。
            mar_threshold: 张嘴阈值，大于该值判定 mouth_open。
            pitch_threshold: 低头阈值，大于该值判定 head_down。
        """
        self.ear_threshold = float(ear_threshold)
        self.mar_threshold = float(mar_threshold)
        self.pitch_threshold = float(pitch_threshold)

    def classify(self, ear, mar, head_pose):
        """
        依据当前帧特征判定疲劳等级。

        Args:
            ear: 眼睛纵横比。
            mar: 嘴部纵横比。
            head_pose: 头部姿态字典，至少包含 pitch。

        Returns:
            {
                "status": "alert" | "fatigue" | "severe_fatigue",
                "score": 0-100,
                "reasons": ["eye_closed", "mouth_open", "head_down"]
            }
        """
        reasons = []
        score = 0
        pitch = float(head_pose.get("pitch", 0.0))
        if float(ear) < self.ear_threshold:
            reasons.append("eye_closed")
            score += 45
        if float(mar) > self.mar_threshold:
            reasons.append("mouth_open")
            score += 30
        if pitch > self.pitch_threshold:
            reasons.append("head_down")
            score += 25
        score = min(100, int(score))
        severe = score >= 75 or ("eye_closed" in reasons and "head_down" in reasons)
        if severe:
            status = "severe_fatigue"
        elif score > 0:
            status = "fatigue"
        else:
            status = "alert"
        return {"status": status, "score": score, "reasons": reasons}

    def classify_sequence(self, feature_sequence):
        """
        对视频帧序列进行批量疲劳判定。

        Args:
            feature_sequence: 每帧特征字典列表，元素可为 None。

        Returns:
            判定结果列表，无法判定位置返回默认 alert 结果。
        """
        outputs = []
        for feature in feature_sequence:
            if not feature:
                outputs.append({"status": "alert", "score": 0, "reasons": []})
                continue
            outputs.append(
                self.classify(
                    ear=feature["ear"],
                    mar=feature["mar"],
                    head_pose=feature["head_pose"],
                )
            )
        return outputs

    def get_threshold_config(self):
        """
        返回当前阈值配置。

        Returns:
            包含 EAR/MAR/PITCH 阈值的字典。
        """
        return {
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "pitch_threshold": self.pitch_threshold,
        }

    def usage_example(self):
        """
        演示如何调用分类器。

        Returns:
            示例判定结果字典。
        """
        return self.classify(ear=0.21, mar=0.62, head_pose={"pitch": 35.0, "yaw": 0.0, "roll": 0.0})


_FATIGUE_CLASSIFIER_SINGLETON = None


def get_fatigue_classifier(ear_threshold=0.25, mar_threshold=0.6, pitch_threshold=30.0) -> FatigueClassifier:
    global _FATIGUE_CLASSIFIER_SINGLETON
    if _FATIGUE_CLASSIFIER_SINGLETON is None:
        _FATIGUE_CLASSIFIER_SINGLETON = FatigueClassifier(
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold,
            pitch_threshold=pitch_threshold,
        )
    return _FATIGUE_CLASSIFIER_SINGLETON
