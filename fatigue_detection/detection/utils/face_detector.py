from typing import Dict, List, Optional
import os

import cv2
import numpy as np

class FaceDetector:
    """
    基于 OpenCV Haar Cascade 和 LBF 模型的人脸关键点检测器。
    无需安装 MediaPipe，完全依赖 OpenCV。
    提供单帧检测、批量帧检测、以及提取疲劳相关区域关键点 (68点) 的能力。
    """

    # 基于 68 点模型 (LBF) 的关键点索引
    LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
    MOUTH_IDX = [48, 50, 52, 54, 56, 58]
    POSE_IDX = [30, 8, 36, 45, 48, 54]
    # 头姿估计使用的参考点：鼻尖(30), 颏(8), 左眼左角(36), 右眼右角(45), 左嘴角(48), 右嘴角(5

    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1, min_detection_confidence: float = 0.1):
        """
        初始化 OpenCV 面部检测和关键点模型。

        Args:
            static_image_mode: 在此实现中忽略（仅为兼容原有接口）。
            max_num_faces: 在此实现中忽略（默认检测单脸或多脸，取最大框）。
            min_detection_confidence: 在此实现中忽略。
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "..", "models")
        
        cascade_path = os.path.join(model_dir, "haarcascade_frontalface_alt2.xml")
        lbf_path = os.path.join(model_dir, "lbfmodel.yaml")
        
        if not os.path.exists(cascade_path) or not os.path.exists(lbf_path):
            raise FileNotFoundError(f"OpenCV 模型文件缺失，请检查 {model_dir} 目录下的 haarcascade 和 lbfmodel 文件。")
            
        self._face_detector = cv2.CascadeClassifier(cascade_path)
        self._landmark_detector = cv2.face.createFacemarkLBF()
        self._landmark_detector.loadModel(lbf_path)

    @staticmethod
    def _enhance_gray(gray: np.ndarray) -> np.ndarray:
        if gray is None or gray.ndim != 2:
            return gray
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        mean_val = float(np.mean(enhanced))
        if mean_val < 95:
            gamma = 1.2
            lut = np.array(
                [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                dtype=np.uint8,
            )
            enhanced = cv2.LUT(enhanced, lut)
        return enhanced

    @staticmethod
    def _expand_face_box(face_box, image_width: int, image_height: int, scale_x: float = 0.18, scale_y: float = 0.22):
        x, y, w, h = [int(v) for v in face_box]
        pad_x = int(w * scale_x)
        pad_y = int(h * scale_y)
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image_width, x + w + pad_x)
        bottom = min(image_height, y + h + pad_y)
        return np.array([left, top, max(1, right - left), max(1, bottom - top)], dtype=np.int32)

    def detect(self, image: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        对单帧 BGR 图像执行关键点检测。

        Args:
            image: OpenCV BGR 图像，形状为 (H, W, 3)。

        Returns:
            若检测到人脸，返回包含像素坐标关键点的字典：
            {
                "all_landmarks": np.ndarray, shape=(68, 2),
                "image_size": (width, height)
            }
            若未检测到人脸，返回 None。
        """
        if image is None or image.ndim != 3:
            return None
            
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self._enhance_gray(gray)
        
        # 多尺度人脸检测
        faces = self._face_detector.detectMultiScale(
            enhanced_gray,
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            faces = self._face_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        
        if len(faces) == 0:
            return None
            
        # 如果有多个人脸，选择面积最大的一个
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        face_box = self._expand_face_box(faces[0], width, height)
        
        # 提取 68 个关键点
        ok, landmarks = self._landmark_detector.fit(enhanced_gray, np.array([face_box]))
        if (not ok or landmarks is None or len(landmarks) == 0):
            ok, landmarks = self._landmark_detector.fit(gray, np.array([face_box]))
        if not ok or landmarks is None or len(landmarks) == 0:
            return None
            
        # landmarks 的形状可能是 (1, 68, 2) 或类似结构
        points = np.squeeze(landmarks[0])  # 压缩多余维度，确保形状为 (68, 2)
        if points.shape != (68, 2):
            # 防御性处理：如果形状不对，尝试调整
            points = points.reshape(-1, 2)
            if points.shape[0] < 68:
                return None
        
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
        return [self.detect(frame) for frame in frames]

    def usage_example(self, image: np.ndarray):
        face_result = self.detect(image)
        if not face_result:
            return None
        return self.get_landmarks(face_result)


_FACE_DETECTOR_SINGLETONS = {}


def get_face_detector(static_image_mode: bool = False, max_num_faces: int = 1) -> FaceDetector:
    global _FACE_DETECTOR_SINGLETONS
    key = (bool(static_image_mode), int(max_num_faces))
    if key not in _FACE_DETECTOR_SINGLETONS:
        _FACE_DETECTOR_SINGLETONS[key] = FaceDetector(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
        )
    return _FACE_DETECTOR_SINGLETONS[key]
