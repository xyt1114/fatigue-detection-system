from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from unittest.mock import patch
import json
import numpy as np

from detection.models import DetectionLog, DetectionSession
from detection.utils.config_manager import ConfigManager
from detection.utils.fatigue_classifier import FatigueClassifier
from detection.utils.feature_extractor import FeatureExtractor
from detection.utils.warning_system import WarningSystem


class TestFeatureExtractor(SimpleTestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_ear_calculation(self):
        eye_points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.5],
                [3.0, 0.5],
                [4.0, 0.0],
                [3.0, -0.5],
                [1.0, -0.5],
            ],
            dtype=np.float32,
        )
        ear = self.extractor.calculate_ear(np.stack([eye_points, eye_points], axis=0))
        self.assertAlmostEqual(ear, 0.25, places=6)

    def test_mar_calculation(self):
        mouth_points = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [3.0, 1.0],
                [4.0, 0.0],
                [3.0, -1.0],
                [1.0, -1.0],
            ],
            dtype=np.float32,
        )
        mar = self.extractor.calculate_mar(mouth_points)
        self.assertAlmostEqual(mar, 0.5, places=6)

    def test_mar_calculation_with_twenty_points(self):
        mouth_points = np.zeros((20, 2), dtype=np.float32)
        mouth_points[0] = [0.0, 0.0]
        mouth_points[2] = [1.0, 1.0]
        mouth_points[4] = [3.0, 1.0]
        mouth_points[6] = [4.0, 0.0]
        mouth_points[8] = [3.0, -1.0]
        mouth_points[10] = [1.0, -1.0]
        mar = self.extractor.calculate_mar(mouth_points)
        self.assertAlmostEqual(mar, 0.5, places=6)

    def test_fatigue_classification(self):
        classifier = FatigueClassifier(ear_threshold=0.25, mar_threshold=0.6, pitch_threshold=30)
        alert = classifier.classify(ear=0.3, mar=0.3, head_pose={"pitch": 5})
        fatigue = classifier.classify(ear=0.2, mar=0.35, head_pose={"pitch": 10})
        severe = classifier.classify(ear=0.2, mar=0.7, head_pose={"pitch": 35})
        self.assertEqual(alert["status"], "alert")
        self.assertEqual(fatigue["status"], "fatigue")
        self.assertEqual(severe["status"], "severe_fatigue")

    def test_warning_system_transition(self):
        warning = WarningSystem(warning_frame_count=3, emergency_frame_count=5)
        outputs = warning.process_sequence([
            "alert",
            "fatigue",
            "fatigue",
            "fatigue",
            "severe_fatigue",
            "severe_fatigue",
            "severe_fatigue",
            "severe_fatigue",
            "severe_fatigue",
        ])
        self.assertEqual(outputs[0]["warning_level"], "normal")
        self.assertEqual(outputs[3]["warning_level"], "warning")
        self.assertEqual(outputs[-1]["warning_level"], "emergency")

    @patch("detection.utils.feature_extractor.cv2.solvePnP")
    @patch("detection.utils.feature_extractor.cv2.Rodrigues")
    @patch("detection.utils.feature_extractor.cv2.decomposeProjectionMatrix")
    def test_calculate_head_pose(self, mock_decompose, mock_rodrigues, mock_solvepnp):
        mock_solvepnp.return_value = (
            True,
            np.array([[0.0], [0.0], [0.0]], dtype=np.float32),
            np.array([[0.0], [0.0], [1.0]], dtype=np.float32),
        )
        mock_rodrigues.return_value = (np.eye(3, dtype=np.float32), None)
        mock_decompose.return_value = (
            None,
            None,
            None,
            None,
            None,
            None,
            np.array([[10.0], [20.0], [30.0]], dtype=np.float32),
        )
        points = np.array(
            [[320.0, 240.0], [320.0, 380.0], [250.0, 210.0], [390.0, 210.0], [280.0, 300.0], [360.0, 300.0]],
            dtype=np.float32,
        )
        pose = self.extractor.calculate_head_pose(points, (640, 480))
        self.assertEqual(pose["pitch"], 10.0)
        self.assertEqual(pose["yaw"], 20.0)
        self.assertEqual(pose["roll"], 30.0)

    @patch("detection.utils.feature_extractor.cv2.solvePnP")
    def test_extract_sequence_features(self, mock_solvepnp):
        mock_solvepnp.return_value = (
            False,
            np.array([[0.0], [0.0], [0.0]], dtype=np.float32),
            np.array([[0.0], [0.0], [1.0]], dtype=np.float32),
        )
        eye = np.array(
            [[0.0, 0.0], [1.0, 0.5], [3.0, 0.5], [4.0, 0.0], [3.0, -0.5], [1.0, -0.5]],
            dtype=np.float32,
        )
        mouth = np.array(
            [[0.0, 0.0], [1.0, 1.0], [3.0, 1.0], [4.0, 0.0], [3.0, -1.0], [1.0, -1.0]],
            dtype=np.float32,
        )
        pose_points = np.array(
            [[320.0, 240.0], [320.0, 380.0], [250.0, 210.0], [390.0, 210.0], [280.0, 300.0], [360.0, 300.0]],
            dtype=np.float32,
        )
        landmarks = {"left_eye": eye, "right_eye": eye, "mouth": mouth, "pose_points_2d": pose_points}
        features = self.extractor.extract_sequence_features([None, landmarks], (640, 480))
        self.assertIsNone(features[0])
        self.assertIsNotNone(features[1])
        demo = self.extractor.usage_example()
        self.assertIn("ear_example", demo)
        self.assertIn("mar_example", demo)


class TestAPI(TestCase):
    def test_api_get_config(self):
        resp = self.client.get(reverse("detection:api_get_config"))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("config", data)
        self.assertIn("classifier_mode", data)
        self.assertIn("ml_model_ready", data)

    @patch("detection.views._extract_frame_from_upload")
    @patch("detection.views._detect_on_frame")
    @patch("detection.views._image_to_base64")
    def test_api_detect_image_success(self, mock_b64, mock_detect, mock_extract):
        mock_extract.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
        mock_detect.return_value = {
            "features": {"ear": 0.26, "mar": 0.4, "head_pose": {"pitch": 10.0, "yaw": 0.0, "roll": 0.0}},
            "classify": {"status": "alert", "score": 0, "reasons": []},
            "annotated": np.zeros((32, 32, 3), dtype=np.uint8),
        }
        mock_b64.return_value = "abc"
        file_obj = SimpleUploadedFile("a.jpg", b"fake-bytes", content_type="image/jpeg")
        resp = self.client.post(reverse("detection:api_detect_image"), {"file": file_obj})
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["fatigue_level"], "alert")
        self.assertIn("image_with_landmarks", payload)
        self.assertIn("inference_mode", payload)

    def test_detect_image_api(self):
        with patch("detection.views._extract_frame_from_upload") as mock_extract, patch(
            "detection.views._detect_on_frame"
        ) as mock_detect, patch("detection.views._image_to_base64") as mock_b64:
            mock_extract.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
            mock_detect.return_value = {
                "features": {"ear": 0.26, "mar": 0.4, "head_pose": {"pitch": 10.0, "yaw": 0.0, "roll": 0.0}},
                "classify": {"status": "alert", "score": 0, "reasons": []},
                "annotated": np.zeros((32, 32, 3), dtype=np.uint8),
            }
            mock_b64.return_value = "abc"
            file_obj = SimpleUploadedFile("a.jpg", b"fake-bytes", content_type="image/jpeg")
            resp = self.client.post(reverse("detection:api_detect_image"), {"file": file_obj})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "success")
            self.assertIn("fatigue_level", data)

    @patch("detection.views._process_video_to_artifacts")
    def test_api_detect_image_video_success(self, mock_video_process):
        mock_video_process.return_value = (
            {
                "mode": "video",
                "processed_video_url": "/media/processed/demo.mp4",
                "curves": {"times": [0.0, 0.1], "ear": [0.2, 0.22], "mar": [0.4, 0.5], "score": [20, 55], "levels": ["alert", "fatigue"]},
                "summary": {"fps": 10.0, "frame_count": 2, "duration_sec": 0.2, "max_score": 55, "max_level": "fatigue", "fatigue_segments": [{"start": 0.1, "end": 0.1, "level": "fatigue"}]},
            },
            None,
        )
        file_obj = SimpleUploadedFile("a.mp4", b"fake-video", content_type="video/mp4")
        resp = self.client.post(reverse("detection:api_detect_image"), {"file": file_obj})
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["mode"], "video")
        self.assertEqual(payload["warning_level"], "warning")
        self.assertIn("processed_video_url", payload)
        self.assertIn("curves", payload)
        self.assertIn("warning_basis", payload)

    @patch("detection.views._process_video_to_artifacts")
    def test_api_detect_image_video_severe_not_enough_should_not_emergency(self, mock_video_process):
        levels = ["alert"] * 10 + ["severe_fatigue"] * 3 + ["alert"] * 7
        mock_video_process.return_value = (
            {
                "mode": "video",
                "processed_video_url": "/media/processed/demo2.mp4",
                "curves": {
                    "times": [i * 0.1 for i in range(len(levels))],
                    "ear": [0.25] * len(levels),
                    "mar": [0.45] * len(levels),
                    "score": [10] * len(levels),
                    "levels": levels,
                },
                "summary": {
                    "fps": 10.0,
                    "frame_count": len(levels),
                    "duration_sec": round(len(levels) / 10.0, 2),
                    "max_score": 85,
                    "max_level": "severe_fatigue",
                    "fatigue_segments": [],
                },
            },
            None,
        )
        file_obj = SimpleUploadedFile("b.mp4", b"fake-video", content_type="video/mp4")
        resp = self.client.post(reverse("detection:api_detect_image"), {"file": file_obj})
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["fatigue_level"], "severe_fatigue")
        self.assertEqual(payload["warning_level"], "normal")

    @patch("detection.views._decode_base64_image")
    @patch("detection.views._detect_on_frame")
    def test_api_detect_frame_success(self, mock_detect, mock_decode):
        mock_decode.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
        mock_detect.return_value = {
            "features": {"ear": 0.2, "mar": 0.7, "head_pose": {"pitch": 35.0, "yaw": 1.0, "roll": 2.0}},
            "classify": {"status": "severe_fatigue", "score": 90, "reasons": ["eye_closed", "head_down"]},
            "annotated": np.zeros((32, 32, 3), dtype=np.uint8),
        }
        resp = self.client.post(
            reverse("detection:api_detect_frame"),
            data=json.dumps({"frame": "dummy-base64"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["fatigue_level"], "severe_fatigue")
        self.assertIn("warning_level", payload)
        self.assertIn("inference_mode", payload)

    @patch("detection.views._decode_base64_image")
    @patch("detection.views._detect_on_frame")
    def test_detect_frame_api(self, mock_detect, mock_decode):
        mock_decode.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
        mock_detect.return_value = {
            "features": {"ear": 0.2, "mar": 0.7, "head_pose": {"pitch": 35.0, "yaw": 1.0, "roll": 2.0}},
            "classify": {"status": "severe_fatigue", "score": 90, "reasons": ["eye_closed", "head_down"]},
            "annotated": np.zeros((32, 32, 3), dtype=np.uint8),
        }
        resp = self.client.post(
            reverse("detection:api_detect_frame"),
            data=json.dumps({"frame": "dummy-base64", "persist": True}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIsNotNone(data.get("session_id"))
        self.assertGreaterEqual(DetectionSession.objects.count(), 1)
        self.assertGreaterEqual(DetectionLog.objects.count(), 1)

    def test_api_update_config_success(self):
        payload = {
            "ear_threshold": 0.23,
            "mar_threshold": 0.65,
            "pitch_threshold": 28,
            "warning_frame_count": 4,
            "emergency_frame_count": 7,
        }
        resp = self.client.post(
            reverse("detection:api_update_config"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["config"]["warning_frame_count"], 4)
        self.assertEqual(data["config"]["emergency_frame_count"], 7)

    def test_api_update_config_invalid(self):
        payload = {
            "ear_threshold": 0.23,
            "mar_threshold": 0.65,
            "pitch_threshold": 28,
            "warning_frame_count": 8,
            "emergency_frame_count": 4,
        }
        resp = self.client.post(
            reverse("detection:api_update_config"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)


class ConfigManagerTestCase(SimpleTestCase):
    def test_config_manager_update_and_reset(self):
        manager = ConfigManager()
        manager.update_config(
            {
                "ear_threshold": 0.24,
                "mar_threshold": 0.63,
                "pitch_threshold": 31,
                "warning_frame_count": 3,
                "emergency_frame_count": 5,
            }
        )
        updated = manager.get_config()
        self.assertEqual(updated["ear_threshold"], 0.24)
        manager.reset_config()
        reset = manager.get_config()
        self.assertEqual(reset["ear_threshold"], float(settings.FATIGUE_CONFIG["EAR_THRESHOLD"]))
        self.assertEqual(reset["mar_threshold"], float(settings.FATIGUE_CONFIG["MAR_THRESHOLD"]))
        self.assertEqual(reset["pitch_threshold"], float(settings.FATIGUE_CONFIG["PITCH_THRESHOLD"]))

    def test_config_manager_validation(self):
        manager = ConfigManager()
        with self.assertRaises(ValueError):
            manager.update_config({"ear_threshold": 0.6})

    def test_classifier_helpers(self):
        classifier = FatigueClassifier()
        seq = classifier.classify_sequence(
            [
                {"ear": 0.3, "mar": 0.3, "head_pose": {"pitch": 5}},
                {"ear": 0.2, "mar": 0.7, "head_pose": {"pitch": 40}},
                None,
            ]
        )
        self.assertEqual(len(seq), 3)
        self.assertIn("ear_threshold", classifier.get_threshold_config())
        self.assertIn("status", classifier.usage_example())
