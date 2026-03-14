from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None


class FaceDetector:
    """
    基于 MediaPipe Face Mesh 的面部关键点检测器。

    提供单帧检测、批量帧检测、以及从 468 点中提取疲劳相关区域关键点的能力。
    """

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    MOUTH_IDX = [78, 82, 13, 308, 14, 312]
    POSE_IDX = [1, 152, 33, 263, 61, 291]

    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1):
        """
        初始化 Face Mesh 模型。

        Args:
            static_image_mode: True 时按静态图像模式运行，False 时适合视频流。
            max_num_faces: 最大检测人脸数量。
        """
        self._face_mesh = None
        if mp is not None:
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def detect(self, image: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        对单帧 BGR 图像执行关键点检测。

        Args:
            image: OpenCV BGR 图像，形状为 (H, W, 3)。

        Returns:
            若检测到人脸，返回包含像素坐标关键点的字典：
            {
                "all_landmarks": np.ndarray, shape=(468, 2),
                "image_size": (width, height)
            }
            若未检测到人脸，返回 None。
        """
        if image is None or image.ndim != 3:
            return None
        if self._face_mesh is None:
            return None
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        face_landmarks = result.multi_face_landmarks[0].landmark
        points = np.array(
            [[lm.x * width, lm.y * height] for lm in face_landmarks],
            dtype=np.float32,
        )
        return {"all_landmarks": points, "image_size": (width, height)}

    def get_landmarks(self, face_result: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """
        从检测结果中提取疲劳相关区域关键点。

        Args:
            face_result: detect() 的返回结果字典。

        Returns:
            包含左右眼、嘴部、头姿关键点的字典，若输入为空返回 None。
        """
        if not face_result or "all_landmarks" not in face_result:
            return None
        pts = face_result["all_landmarks"]
        return {
            "left_eye": pts[self.LEFT_EYE_IDX],
            "right_eye": pts[self.RIGHT_EYE_IDX],
            "mouth": pts[self.MOUTH_IDX],
            "pose_points_2d": pts[self.POSE_IDX],
            "all_landmarks": pts,
            "image_size": face_result.get("image_size"),
        }

    def detect_batch(self, frames: List[np.ndarray]) -> List[Optional[Dict[str, np.ndarray]]]:
        """
        批量处理视频帧序列。

        Args:
            frames: BGR 帧列表。

        Returns:
            与输入等长的检测结果列表，每个元素为 detect() 的输出或 None。
        """
        return [self.detect(frame) for frame in frames]

    def usage_example(self, image: np.ndarray):
        """
        演示如何调用面部检测与区域关键点提取。

        Args:
            image: 单帧 BGR 图像。

        Returns:
            若检测成功，返回提取后的区域关键点字典；否则返回 None。
        """
        face_result = self.detect(image)
        if not face_result:
            return None
        return self.get_landmarks(face_result)


_FACE_DETECTOR_SINGLETON = None


def get_face_detector(static_image_mode: bool = False, max_num_faces: int = 1) -> FaceDetector:
    global _FACE_DETECTOR_SINGLETON
    if _FACE_DETECTOR_SINGLETON is None:
        _FACE_DETECTOR_SINGLETON = FaceDetector(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
        )
    return _FACE_DETECTOR_SINGLETON
