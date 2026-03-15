import cv2
import numpy as np


class FeatureExtractor:
    """
    疲劳特征提取器。

    提供 EAR、MAR、头部姿态角的计算能力，并支持批量帧特征处理。
    """

    def __init__(self):
        """
        初始化特征提取器并定义关键点索引。

        关键点顺序遵循 EAR/MAR 公式中的 p1..p6 约定。
        """
        # 适应 68 点 LBF 模型
        self.left_eye_indices = [36, 37, 38, 39, 40, 41]
        self.right_eye_indices = [42, 43, 44, 45, 46, 47]
        # 嘴部 MAR 计算选取关键点 (68点模型): 48, 50, 52, 54, 56, 58
        self.mouth_indices = [48, 50, 52, 54, 56, 58]
        # 头部姿态 2D 点 (68点模型): 鼻尖(30), 颏(8), 左眼左角(36), 右眼右角(45), 左嘴角(48), 右嘴角(54)
        self.pose_indices = [30, 8, 36, 45, 48, 54]
        self.model_points_3d = np.array(
            [
                (0.0, 0.0, 0.0),             # 鼻尖
                (0.0, -330.0, -65.0),        # 颏
                (-225.0, 170.0, -135.0),     # 左眼左角
                (225.0, 170.0, -135.0),      # 右眼右角
                (-150.0, -150.0, -125.0),    # 左嘴角
                (150.0, -150.0, -125.0),     # 右嘴角
            ],
            dtype=np.float32,
        )

    def _ratio_from_six_points(self, points: np.ndarray) -> float:
        """
        按 6 点几何定义计算比值。

        Args:
            points: 形状为 (6, 2) 的二维点集。

        Returns:
            按公式计算得到的比例值，若分母过小返回 0.0。
        """
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (6, 2):
            raise ValueError("points must have shape (6, 2)")
        p1, p2, p3, p4, p5, p6 = points
        numerator = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        denominator = 2.0 * np.linalg.norm(p1 - p4)
        if denominator <= 1e-6:
            return 0.0
        return float(numerator / denominator)

    def calculate_ear(self, eye_landmarks) -> float:
        """
        计算眼睛纵横比 EAR。

        Args:
            eye_landmarks: 支持三种输入格式：
                1) {"left_eye": (6,2), "right_eye": (6,2)}
                2) (2,6,2) 左右眼点集
                3) (6,2) 单眼点集

        Returns:
            EAR 值。双眼输入时返回左右眼平均值。
        """
        if isinstance(eye_landmarks, dict):
            left = self._ratio_from_six_points(eye_landmarks["left_eye"])
            right = self._ratio_from_six_points(eye_landmarks["right_eye"])
            return float((left + right) / 2.0)
        arr = np.asarray(eye_landmarks, dtype=np.float32)
        if arr.shape == (2, 6, 2):
            left = self._ratio_from_six_points(arr[0])
            right = self._ratio_from_six_points(arr[1])
            return float((left + right) / 2.0)
        return self._ratio_from_six_points(arr)

    def calculate_mar(self, mouth_landmarks) -> float:
        """
        计算嘴部纵横比 MAR。

        Args:
            mouth_landmarks: 形状为 (6,2) 的嘴部关键点。

        Returns:
            MAR 值，值越大通常表示张嘴幅度越大。
        """
        points = np.asarray(mouth_landmarks, dtype=np.float32)
        if points.shape == (6, 2):
            return self._ratio_from_six_points(points)
        if points.shape == (20, 2):
            mapped = points[[0, 2, 4, 6, 8, 10]]
            return self._ratio_from_six_points(mapped)
        return 0.0

    def calculate_head_pose(self, landmarks, image_size):
        """
        使用 PnP 估计头部姿态欧拉角。

        Args:
            landmarks: 关键点字典或数组。
                若为字典，优先使用 landmarks["pose_points_2d"]；
                若提供 landmarks["all_landmarks"]，将按预设索引自动提取。
            image_size: (width, height)。

        Returns:
            包含 pitch/yaw/roll（单位：度）的字典。
        """
        width, height = image_size
        if isinstance(landmarks, dict):
            if "pose_points_2d" in landmarks:
                image_points = np.asarray(landmarks["pose_points_2d"], dtype=np.float32)
            else:
                all_points = np.asarray(landmarks["all_landmarks"], dtype=np.float32)
                image_points = all_points[self.pose_indices]
        else:
            image_points = np.asarray(landmarks, dtype=np.float32)
        if image_points.shape != (6, 2):
            raise ValueError("head pose points must have shape (6, 2)")

        focal_length = float(width)
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        projection = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection)
        pitch = float(euler[0][0])
        yaw = float(euler[1][0])
        roll = float(euler[2][0])
        return {"pitch": pitch, "yaw": yaw, "roll": roll}

    def extract_frame_features(self, landmarks, image_size):
        """
        计算单帧疲劳特征。

        Args:
            landmarks: get_landmarks() 返回的关键点字典。
            image_size: (width, height)。

        Returns:
            包含 ear/mar/head_pose 的特征字典。
        """
        try:
            ear = self.calculate_ear({"left_eye": landmarks["left_eye"], "right_eye": landmarks["right_eye"]})
            mar = self.calculate_mar(landmarks["mouth"])
            head_pose = self.calculate_head_pose(landmarks, image_size)
            return {"ear": ear, "mar": mar, "head_pose": head_pose}
        except Exception:
            return {"ear": 0.0, "mar": 0.0, "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}}

    def extract_sequence_features(self, landmarks_sequence, image_size):
        """
        批量提取视频帧序列特征。

        Args:
            landmarks_sequence: 关键点字典列表，元素可为 None。
            image_size: (width, height)。

        Returns:
            特征列表，无法提取的位置返回 None。
        """
        results = []
        for landmarks in landmarks_sequence:
            if not landmarks:
                results.append(None)
                continue
            results.append(self.extract_frame_features(landmarks, image_size))
        return results

    def usage_example(self):
        """
        演示如何调用 EAR/MAR/头姿估计接口。

        Returns:
            示例输入与对应计算结果。
        """
        eye = np.array(
            [[0.0, 0.0], [1.0, 0.5], [3.0, 0.5], [4.0, 0.0], [3.0, -0.5], [1.0, -0.5]],
            dtype=np.float32,
        )
        mouth = np.array(
            [[0.0, 0.0], [1.0, 1.0], [3.0, 1.0], [4.0, 0.0], [3.0, -1.0], [1.0, -1.0]],
            dtype=np.float32,
        )
        ear = self.calculate_ear(np.stack([eye, eye], axis=0))
        mar = self.calculate_mar(mouth)
        return {"ear_example": ear, "mar_example": mar}


_FEATURE_EXTRACTOR_SINGLETON = None


def get_feature_extractor() -> FeatureExtractor:
    global _FEATURE_EXTRACTOR_SINGLETON
    if _FEATURE_EXTRACTOR_SINGLETON is None:
        _FEATURE_EXTRACTOR_SINGLETON = FeatureExtractor()
    return _FEATURE_EXTRACTOR_SINGLETON
