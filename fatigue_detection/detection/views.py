import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from .forms import ConfigForm, UploadForm
from .models import DetectionLog, DetectionSession
from .utils.config_manager import ConfigManager
from .utils.logger import log_config_change, log_detection_event, log_performance_event, log_sampling_event
from .utils.model_loader import get_detector_instances, get_ml_classifier, rebuild_runtime_instances

IMAGE_EXT = {"jpg", "jpeg", "png", "bmp"}
VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

_CONFIG_MANAGER = ConfigManager()
_FACE_DETECTOR, _STATIC_FACE_DETECTOR, _FEATURE_EXTRACTOR, _CLASSIFIER, _WARNING_SYSTEM = get_detector_instances()
_ML_CLASSIFIER = get_ml_classifier()
_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_LOGGER = logging.getLogger(__name__)


def _rebuild_models():
    global _CLASSIFIER, _WARNING_SYSTEM
    runtime_config = _CONFIG_MANAGER.get_config()
    _CLASSIFIER, _WARNING_SYSTEM = rebuild_runtime_instances(runtime_config)


_rebuild_models()


def _image_to_base64(image):
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _decode_base64_image(data):
    if not data:
        return None
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
    except Exception:
        return None
    array = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _compress_image(image, quality=80, max_size=640):
    if image is None or image.ndim != 3:
        return image
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, compressed = cv2.imencode(".jpg", image, encode_param)
    if not ok:
        return image
    return cv2.imdecode(compressed, cv2.IMREAD_COLOR)


def _enhance_frame_for_face_detection(image):
    if image is None or image.ndim != 3:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    mean_val = float(np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)))
    if mean_val < 90:
        gamma = 1.25
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
        enhanced = cv2.LUT(enhanced, lut)
    return enhanced


def _frame_hash(frame):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return ""
    return hashlib.md5(encoded.tobytes()).hexdigest()


def _get_user_id(request):
    return (
        request.headers.get("X-User-Id")
        or request.GET.get("user_id")
        or request.META.get("REMOTE_ADDR")
        or "anonymous"
    )


def _resolve_session(session_id, user_id):
    if session_id:
        try:
            return DetectionSession.objects.get(id=session_id)
        except DetectionSession.DoesNotExist:
            return None
    return DetectionSession.objects.create(user_id=user_id)


def _persist_detection_result(session, features, fatigue_level, warning_level):
    if not session:
        return
    DetectionLog.objects.create(
        session=session,
        ear=float(features.get("ear", 0.0)),
        mar=float(features.get("mar", 0.0)),
        pitch=float((features.get("head_pose") or {}).get("pitch", 0.0)),
        fatigue_level=fatigue_level,
        warning_level=warning_level,
    )
    session.total_frames = int(session.total_frames) + 1
    if fatigue_level in {"fatigue", "severe_fatigue"}:
        session.fatigue_frames = int(session.fatigue_frames) + 1
    rank = {"normal": 0, "warning": 1, "emergency": 2}
    current_rank = rank.get(session.max_warning_level or "normal", 0)
    incoming_rank = rank.get(warning_level, 0)
    if incoming_rank > current_rank:
        session.max_warning_level = warning_level
    session.save(update_fields=["total_frames", "fatigue_frames", "max_warning_level"])


def _close_session_if_needed(session, should_close):
    if session and should_close and session.end_time is None:
        session.end_time = timezone.now()
        session.save(update_fields=["end_time"])


def _draw_landmarks(frame, landmarks):
    output = frame.copy()
    points = landmarks.get("all_landmarks")
    if points is None:
        points = np.concatenate(
            [landmarks["left_eye"], landmarks["right_eye"], landmarks["mouth"]],
            axis=0,
        )
    for point in points:
        cv2.circle(output, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
    return output


def _landmark_cache_key(session_id, user_id):
    identity = session_id or user_id or "anonymous"
    return f"realtime:landmarks:{identity}"


def _head_pose_cache_key(session_id, user_id):
    identity = session_id or user_id or "anonymous"
    return f"realtime:head_pose:{identity}"


def _smooth_landmarks(landmarks, session_id=None, user_id=None, alpha=0.68):
    if not landmarks or "all_landmarks" not in landmarks:
        return landmarks
    points = np.asarray(landmarks["all_landmarks"], dtype=np.float32)
    if points.shape != (68, 2):
        return landmarks
    cache_key = _landmark_cache_key(session_id, user_id)
    previous = cache.get(cache_key)
    smoothed = points
    if isinstance(previous, np.ndarray) and previous.shape == (68, 2):
        smoothed = alpha * points + (1.0 - alpha) * previous.astype(np.float32)
    cache.set(cache_key, smoothed.astype(np.float32), timeout=120)
    updated = dict(landmarks)
    updated["all_landmarks"] = smoothed
    for key, indices in (
        ("left_eye", _FACE_DETECTOR.LEFT_EYE_IDX),
        ("right_eye", _FACE_DETECTOR.RIGHT_EYE_IDX),
        ("mouth", _FACE_DETECTOR.MOUTH_IDX),
        ("pose_points_2d", _FACE_DETECTOR.POSE_IDX),
    ):
        updated[key] = smoothed[indices]
    return updated


def _clear_landmark_smoothing(session_id=None, user_id=None):
    cache.delete(_landmark_cache_key(session_id, user_id))


def _smooth_head_pose(
    head_pose,
    session_id=None,
    user_id=None,
    alpha=0.35,
    yaw_roll_deadband=2.5,
    pitch_deadband=1.2,
):
    if not isinstance(head_pose, dict):
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    current = {
        "pitch": float(head_pose.get("pitch", 0.0) or 0.0),
        "yaw": float(head_pose.get("yaw", 0.0) or 0.0),
        "roll": float(head_pose.get("roll", 0.0) or 0.0),
    }
    cache_key = _head_pose_cache_key(session_id, user_id)
    previous = cache.get(cache_key)
    if not isinstance(previous, dict):
        cache.set(cache_key, current, timeout=120)
        return current

    smoothed = {}
    for axis in ("pitch", "yaw", "roll"):
        prev_val = float(previous.get(axis, 0.0) or 0.0)
        curr_val = current[axis]
        blended = alpha * curr_val + (1.0 - alpha) * prev_val
        deadband = pitch_deadband if axis == "pitch" else yaw_roll_deadband
        smoothed[axis] = prev_val if abs(blended - prev_val) < deadband else blended
    cache.set(cache_key, smoothed, timeout=120)
    return smoothed


def _clear_head_pose_smoothing(session_id=None, user_id=None):
    cache.delete(_head_pose_cache_key(session_id, user_id))


def _clear_realtime_smoothing(session_id=None, user_id=None):
    _clear_landmark_smoothing(session_id=session_id, user_id=user_id)
    _clear_head_pose_smoothing(session_id=session_id, user_id=user_id)


def _extract_face_roi(frame, landmarks, padding_ratio=0.2):
    if frame is None or not landmarks:
        return frame
    points = landmarks.get("all_landmarks")
    if points is None:
        points = np.concatenate(
            [landmarks["left_eye"], landmarks["right_eye"], landmarks["mouth"]],
            axis=0,
        )
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return frame
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)
    pad_x = width * float(padding_ratio)
    pad_y = height * float(padding_ratio)
    h, w = frame.shape[:2]
    left = max(0, int(x_min - pad_x))
    top = max(0, int(y_min - pad_y))
    right = min(w, int(x_max + pad_x))
    bottom = min(h, int(y_max + pad_y))
    if right - left < 16 or bottom - top < 16:
        return frame
    return frame[top:bottom, left:right].copy()


def _prepare_face_roi_for_cnn(frame, landmarks):
    roi = _extract_face_roi(frame, landmarks, padding_ratio=0.28)
    if roi is None or roi.ndim != 3 or roi.size == 0:
        return frame
    return _enhance_frame_for_face_detection(roi)


def _resolve_classifier_mode():
    mode = str(getattr(settings, "CLASSIFIER_MODE", "rule") or "rule").strip().lower()
    if mode not in {"rule", "ml", "cnn", "fusion"}:
        return "rule"
    return mode


def _classify_frame(frame, features):
    mode = _resolve_classifier_mode()
    
    rule_result = _CLASSIFIER.classify(
        ear=features["ear"],
        mar=features["mar"],
        head_pose=features["head_pose"],
    )
    rule_result["confidence"] = 0.0
    rule_result["raw_label"] = "rule"
    rule_result["inference_mode"] = "rule"
    
    if mode in {"cnn", "fusion"}:
        from .utils.model_loader import get_cnn_classifier
        cnn_classifier = get_cnn_classifier()
        
        # --- 动态融合 (Dynamic Fusion) 逻辑 ---
        if mode == "fusion":
            # 1. 评估当前光照条件 (亮度)
            if frame is not None and frame.size > 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray))
            else:
                brightness = 120.0 # 默认正常亮度
            
            # 2. 光照良好判定 (亮度适中)
            is_good_lighting = 80 <= brightness <= 200
            
            # 3. 始终运行 CNN，保证主路判定
            cnn_result = cnn_classifier.predict(frame) if cnn_classifier else None
            
            if cnn_result is not None:
                rank = {"alert": 0, "fatigue": 1, "severe_fatigue": 2}
                
                # 4. 动态置信度加权：如果光照好，大幅提高规则的置信度权重；光照差，依赖 CNN
                if is_good_lighting:
                    # 光照好：规则特征提取极其准确，赋予极高置信度
                    rule_confidence = 0.9
                    cnn_result["confidence"] = min(1.0, cnn_result.get("confidence", 0.5) * 0.8) # 稍微压低CNN置信度
                else:
                    # 光照差：规则失效概率大，置信度调低，依赖CNN的泛化
                    rule_confidence = 0.2
                    cnn_result["confidence"] = min(1.0, cnn_result.get("confidence", 0.5) * 1.2) # 提高CNN置信度
                
                rule_result["confidence"] = rule_confidence
                
                # 5. 融合决策：高风险优先，但参考了环境动态置信度
                # 即使光照差，如果规则仍然极度异常（比如闭眼时间极长），依然覆盖 CNN
                if rank.get(rule_result["status"], 0) > rank.get(cnn_result.get("status", "alert"), 0):
                    cnn_result["status"] = rule_result["status"]
                    
                # 融合最终得分（综合了光照调节后的置信度）
                cnn_result["score"] = max(cnn_result.get("score", 0), rule_result.get("confidence", 0))
                
                reasons = set(cnn_result.get("reasons", []))
                reasons.update(rule_result.get("reasons", []))
                cnn_result["reasons"] = list(reasons)
                
                # 记录是光照好还是光照差的融合
                cnn_result["inference_mode"] = "fusion_dynamic_good_light" if is_good_lighting else "fusion_dynamic_poor_light"
                return cnn_result
        else:
            # 纯 CNN 模式
            cnn_result = cnn_classifier.predict(frame) if cnn_classifier else None
            if cnn_result is not None:
                cnn_result["inference_mode"] = "cnn"
                return cnn_result
            
    if mode == "ml":
        ml_result = _ML_CLASSIFIER.predict(frame) if _ML_CLASSIFIER else None
        if ml_result is not None:
            rank = {"alert": 0, "fatigue": 1, "severe_fatigue": 2}
            if rank.get(rule_result["status"], 0) > rank.get(ml_result.get("status", "alert"), 0):
                ml_result["status"] = rule_result["status"]
            ml_result["score"] = max(ml_result.get("score", 0), rule_result.get("score", 0))
            reasons = set(ml_result.get("reasons", []))
            reasons.update(rule_result.get("reasons", []))
            ml_result["reasons"] = list(reasons)
            ml_result["inference_mode"] = "ml"
            return ml_result
            
    return rule_result


def _detect_with_detector(frame, detector, include_annotation=True):
    face_result = detector.detect(frame)
    if not face_result:
        return None
    landmarks = detector.get_landmarks(face_result)
    if not landmarks:
        return None
    try:
        features = _FEATURE_EXTRACTOR.extract_frame_features(landmarks, face_result["image_size"])
        cnn_input = _prepare_face_roi_for_cnn(frame, landmarks)
        classify_result = _classify_frame(cnn_input, features)
        annotated = _draw_landmarks(frame, landmarks) if include_annotation else None
        return {
            "features": features,
            "classify": classify_result,
            "annotated": annotated,
            "landmarks": landmarks,
        }
    except Exception:
        return None


def _detect_on_frame(frame, include_annotation=True, retry_static=True):
    detect_result = _detect_with_detector(frame, _FACE_DETECTOR, include_annotation=include_annotation)
    if detect_result:
        return detect_result
    if retry_static:
        return _detect_with_detector(frame, _STATIC_FACE_DETECTOR, include_annotation=include_annotation)
    return None


def _detect_with_cache(frame, include_annotation=True, cache_ttl=300, retry_static=True):
    config_signature = str(sorted(_CONFIG_MANAGER.get_config().items()))
    classifier_signature = _resolve_classifier_mode()
    frame_signature = _frame_hash(frame)
    cache_key = (
        f"detect:frame:{frame_signature}:{hash(config_signature)}:{classifier_signature}:{int(include_annotation)}:{int(retry_static)}"
    )
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result, True
    detect_result = _detect_on_frame(
        frame,
        include_annotation=include_annotation,
        retry_static=retry_static,
    )
    if detect_result is not None:
        cache.set(cache_key, detect_result, timeout=cache_ttl)
    return detect_result, False


def _sample_video_frames(video_path, sample_count=16):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 1:
        anchors = np.linspace(0.05, 0.95, num=max(4, sample_count))
        indices = sorted(
            {
                min(max(0, int(total_frames * anchor)), max(0, total_frames - 1))
                for anchor in anchors
            }
        )
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        sampled = 0
        step = 2
        index = 0
        while sampled < sample_count:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                frames.append(frame)
                sampled += 1
            index += 1
    cap.release()
    return frames


def _extract_frames_from_upload(uploaded_file, sample_count=16):
    suffix = uploaded_file.name.lower().split(".")[-1] if "." in uploaded_file.name else ""
    if suffix in IMAGE_EXT:
        data = uploaded_file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return [], suffix
        return [frame], suffix
    if suffix in VIDEO_EXT:
        import os
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False)
        temp_path = temp_file.name
        try:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file.close()
            sampled_frames = _sample_video_frames(temp_path, sample_count=sample_count)
            if sampled_frames:
                debug_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, "debug_first_frame.jpg"), sampled_frames[0])
            return sampled_frames, suffix
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    return [], suffix


def _extract_frame_from_upload(uploaded_file):
    frames, _ = _extract_frames_from_upload(uploaded_file, sample_count=1)
    return frames[0] if frames else None


def _extract_candidate_frames(uploaded_file, sample_count=16):
    suffix = uploaded_file.name.lower().split(".")[-1] if "." in uploaded_file.name else ""
    if suffix in IMAGE_EXT:
        frame = _extract_frame_from_upload(uploaded_file)
        return ([frame] if frame is not None else []), suffix
    return _extract_frames_from_upload(uploaded_file, sample_count=sample_count)


def _detect_from_candidates(
    frames,
    include_annotation=True,
    cache_ttl=300,
    retry_static=True,
    compress_max_size=720,
    compress_quality=82,
    endpoint="api_detect_image",
):
    cache_hit_any = False
    for idx, frame in enumerate(frames):
        if frame is None:
            continue
        detect_result, cache_hit = _detect_with_cache(
            frame,
            include_annotation=include_annotation,
            cache_ttl=cache_ttl,
            retry_static=retry_static,
        )
        log_sampling_event(endpoint, idx, "raw", bool(detect_result))
        cache_hit_any = cache_hit_any or cache_hit
        if detect_result:
            return detect_result, cache_hit_any
        
        # 移除额外的_compress_image步骤，仅使用CLAHE增强，防止因resize导致人脸太小
        enhanced = _enhance_frame_for_face_detection(frame)
        detect_result, cache_hit = _detect_with_cache(
            enhanced,
            include_annotation=include_annotation,
            cache_ttl=cache_ttl,
            retry_static=retry_static,
        )
        log_sampling_event(endpoint, idx, "enhanced", bool(detect_result))
        cache_hit_any = cache_hit_any or cache_hit
        if detect_result:
            return detect_result, cache_hit_any
            
    return None, cache_hit_any


def _warning_from_level(level):
    if level == "severe_fatigue":
        return "emergency"
    if level == "fatigue":
        return "warning"
    return "normal"


def _warning_from_video_levels(levels, max_level):
    levels = list(levels or [])
    if not levels:
        return _warning_from_level(max_level), {
            "severe_frame_count": 0,
            "total_frame_count": 0,
            "severe_ratio": 0.0,
        }
    severe_count = sum(1 for level in levels if level == "severe_fatigue")
    fatigue_count = sum(1 for level in levels if level in {"fatigue", "severe_fatigue"})
    total_count = len(levels)
    severe_ratio = severe_count / total_count if total_count else 0.0
    fatigue_ratio = fatigue_count / total_count if total_count else 0.0
    severe_min_frames = int(settings.FATIGUE_CONFIG.get("VIDEO_SEVERE_EMERGENCY_MIN_FRAMES", 8))
    severe_min_ratio = float(settings.FATIGUE_CONFIG.get("VIDEO_SEVERE_EMERGENCY_MIN_RATIO", 0.35))
    warning_min_ratio = float(settings.FATIGUE_CONFIG.get("VIDEO_WARNING_MIN_RATIO", 0.25))
    if severe_count >= severe_min_frames and severe_ratio >= severe_min_ratio:
        warning_level = "emergency"
    elif fatigue_ratio >= warning_min_ratio:
        warning_level = "warning"
    else:
        warning_level = "normal"
    return warning_level, {
        "severe_frame_count": int(severe_count),
        "total_frame_count": int(total_count),
        "severe_ratio": round(float(severe_ratio), 4),
    }


def _build_fatigue_segments(times, levels):
    segments = []
    start_idx = None
    current_level = None
    for idx, level in enumerate(levels):
        if level in {"fatigue", "severe_fatigue"}:
            if current_level is None:
                current_level = level
                start_idx = idx
            elif current_level != level:
                segments.append(
                    {
                        "start": round(float(times[start_idx]), 3),
                        "end": round(float(times[idx - 1]), 3),
                        "level": current_level,
                    }
                )
                current_level = level
                start_idx = idx
        else:
            if current_level is not None:
                segments.append(
                    {
                        "start": round(float(times[start_idx]), 3),
                        "end": round(float(times[idx - 1]), 3),
                        "level": current_level,
                    }
                )
                current_level = None
                start_idx = None
    if current_level is not None and start_idx is not None and times:
        segments.append(
            {
                "start": round(float(times[start_idx]), 3),
                "end": round(float(times[-1]), 3),
                "level": current_level,
            }
        )
    return segments


def _max_level(levels):
    rank = {"alert": 0, "fatigue": 1, "severe_fatigue": 2}
    best = "alert"
    for level in levels:
        if rank.get(level, 0) > rank.get(best, 0):
            best = level
    return best


def _draw_video_overlay(frame, classify_result, features, timestamp_sec):
    level = classify_result.get("status", "alert")
    score = int(classify_result.get("score", 0))
    color = (46, 204, 113)
    if level == "fatigue":
        color = (39, 181, 255)
    elif level == "severe_fatigue":
        color = (68, 68, 255)
    cv2.rectangle(frame, (8, 8), (540, 110), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (540, 110), color, 2)
    pose = features.get("head_pose") or {}
    text1 = f"t={timestamp_sec:.2f}s  level={level}  score={score}"
    text2 = f"EAR={features.get('ear', 0.0):.4f}  MAR={features.get('mar', 0.0):.4f}  pitch={pose.get('pitch', 0.0):.2f}"
    cv2.putText(frame, text1, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, text2, (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)


def _validate_video_file(path):
    if not path.exists() or path.stat().st_size <= 0:
        return False
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None and frame.size > 0)


def _resolve_ffmpeg_bin():
    configured = getattr(settings, "FFMPEG_BIN", "")
    if configured:
        configured_path = str(configured)
        if os.path.isfile(configured_path):
            return configured_path
        resolved = shutil.which(configured_path)
        if resolved:
            return resolved
    for candidate in ["ffmpeg", r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
        if os.path.isfile(candidate):
            return candidate
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return ""


def _transcode_video_for_web(source_path, target_path):
    ffmpeg_bin = _resolve_ffmpeg_bin()
    if not ffmpeg_bin:
        return False, "未找到ffmpeg可执行文件"
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(target_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception as exc:
        return False, f"调用ffmpeg失败[{ffmpeg_bin}]: {exc}"
    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip()[-600:]
        return False, f"ffmpeg转码失败[{ffmpeg_bin}](code={result.returncode}): {stderr_tail}"
    if not _validate_video_file(target_path):
        return False, "ffmpeg转码后视频校验失败"
    return True, ""


def _process_video_to_artifacts(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "无法解析上传视频"
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return None, "视频分辨率异常"

    output_dir = settings.MEDIA_ROOT / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"processed_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = output_dir / output_name
    raw_path = output_dir / f"raw_{uuid.uuid4().hex[:8]}.mp4"

    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return None, "处理视频写入失败"

    times = []
    ear_series = []
    mar_series = []
    score_series = []
    level_series = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        detect_result = _detect_on_frame(frame, include_annotation=True, retry_static=True)
        if detect_result:
            features = detect_result["features"]
            classify = detect_result["classify"]
            annotated = detect_result.get("annotated")
            if annotated is None:
                annotated = frame.copy()
        else:
            features = {"ear": 0.0, "mar": 0.0, "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}}
            classify = {"status": "alert", "score": 0, "reasons": ["no_face"]}
            annotated = frame.copy()

        ts = frame_index / fps
        _draw_video_overlay(annotated, classify, features, ts)
        writer.write(annotated)

        times.append(round(float(ts), 3))
        ear_series.append(round(float(features.get("ear", 0.0)), 6))
        mar_series.append(round(float(features.get("mar", 0.0)), 6))
        score_series.append(int(classify.get("score", 0)))
        level_series.append(classify.get("status", "alert"))
        frame_index += 1

    cap.release()
    writer.release()

    if frame_index == 0:
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None, "视频没有可处理帧"

    final_ready, transcode_error = _transcode_video_for_web(raw_path, output_path)
    try:
        raw_path.unlink(missing_ok=True)
    except Exception:
        pass
    if not final_ready:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        _LOGGER.error("视频转码失败: %s", transcode_error)
        return None, "视频转码失败，请安装ffmpeg并重试"

    if not _validate_video_file(output_path):
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None, "处理后视频不可播放，请重试"

    segments = _build_fatigue_segments(times, level_series)
    max_level = _max_level(level_series)
    duration = round(frame_index / fps, 3)

    return {
        "mode": "video",
        "processed_video_url": f"/media/processed/{output_path.name}",
        "curves": {
            "times": times,
            "ear": ear_series,
            "mar": mar_series,
            "score": score_series,
            "levels": level_series,
        },
        "summary": {
            "fps": round(fps, 3),
            "frame_count": frame_index,
            "duration_sec": duration,
            "max_score": int(max(score_series) if score_series else 0),
            "max_level": max_level,
            "fatigue_segments": segments,
        },
    }, None


def index(request):
    return redirect("detection:upload_detect")


def upload_detect(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    context = {"form": form}
    if request.method == "POST" and form.is_valid():
        upload = form.cleaned_data["file"]
        frames, suffix = _extract_candidate_frames(upload, sample_count=16)
        if not frames:
            context["error"] = "无法解析上传文件，请检查文件格式是否正确"
            return render(request, "detection/upload.html", context)
        detect_result, _ = _detect_from_candidates(
            frames,
            include_annotation=True,
            cache_ttl=300,
            retry_static=True,
            compress_max_size=960,
            compress_quality=84,
            endpoint="upload_detect",
        )
        if not detect_result:
            if suffix in VIDEO_EXT:
                context["error"] = "视频采样帧均未检测到人脸，请确保人脸清晰可见并位于画面中"
            else:
                context["error"] = "未检测到人脸，请使用正脸且光照充足的图片"
            return render(request, "detection/upload.html", context)
        features = detect_result["features"]
        classify = detect_result["classify"]
        context["result"] = {
            "fatigue_level": classify["status"],
            "score": classify["score"],
            "reasons": classify["reasons"],
            "ear": round(features["ear"], 4),
            "mar": round(features["mar"], 4),
            "head_pose": {k: round(v, 2) for k, v in features["head_pose"].items()},
            "image_with_landmarks": _image_to_base64(detect_result["annotated"]),
        }
    return render(request, "detection/upload.html", context)


def realtime_detect(request):
    return render(request, "detection/realtime.html")


@csrf_exempt
@api_view(["POST"])
def api_detect_image(request):
    started_at = time.perf_counter()
    user_id = _get_user_id(request)
    upload = request.FILES.get("file") or request.FILES.get("image")
    if upload is None:
        return JsonResponse({"status": "error", "message": "未找到上传文件(file/image)"}, status=400)
    suffix = upload.name.lower().split(".")[-1] if "." in upload.name else ""
    image_limit = 10 * 1024 * 1024
    video_limit = 100 * 1024 * 1024
    if suffix in IMAGE_EXT and upload.size > image_limit:
        return JsonResponse({"status": "error", "message": "图片大小超过10MB"}, status=400)
    if suffix in VIDEO_EXT and upload.size > video_limit:
        return JsonResponse({"status": "error", "message": "视频大小超过100MB"}, status=400)
    if suffix in VIDEO_EXT:
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False)
        temp_path = temp_file.name
        try:
            for chunk in upload.chunks():
                temp_file.write(chunk)
            temp_file.close()
            payload, error_message = _EXECUTOR.submit(_process_video_to_artifacts, temp_path).result()
        except Exception:
            payload, error_message = None, "检测过程异常，请重试或更换素材"
        finally:
            try:
                temp_file.close()
            except Exception:
                pass
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
        if payload is None:
            return JsonResponse({"status": "error", "message": error_message or "视频处理失败"}, status=400)

        max_level = payload["summary"].get("max_level", "alert")
        warning_level, warning_basis = _warning_from_video_levels(payload["curves"].get("levels"), max_level)
        response_payload = {
            "status": "success",
            "mode": "video",
            "inference_mode": _resolve_classifier_mode(),
            "fatigue_level": max_level,
            "warning_level": warning_level,
            "score": int(payload["summary"].get("max_score", 0)),
            "ear": float(np.mean(payload["curves"].get("ear") or [0.0])),
            "mar": float(np.mean(payload["curves"].get("mar") or [0.0])),
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "reasons": ["video_timeline"],
            "processed_video_url": payload["processed_video_url"],
            "curves": payload["curves"],
            "summary": payload["summary"],
            "warning_basis": warning_basis,
        }
        elapsed = (time.perf_counter() - started_at) * 1000
        log_detection_event(user_id, max_level, response_payload["warning_level"])
        log_performance_event("api_detect_image", elapsed, cache_hit=False)
        return JsonResponse(response_payload)

    frames, suffix = _extract_candidate_frames(upload, sample_count=16)
    if not frames:
        return JsonResponse({"status": "error", "message": "无法解析上传文件"}, status=400)
    try:
        detect_result, cache_hit = _EXECUTOR.submit(
            _detect_from_candidates,
            frames,
            True,
            300,
            True,
            960,
            84,
            "api_detect_image",
        ).result()
    except Exception:
        _LOGGER.exception("api_detect_image执行失败")
        return JsonResponse(
            {"status": "error", "message": "检测过程异常，请重试或更换素材"},
            status=400,
        )
    if not detect_result:
        elapsed = (time.perf_counter() - started_at) * 1000
        log_performance_event("api_detect_image", elapsed, cache_hit=cache_hit)
        return JsonResponse({"status": "error", "message": "未检测到人脸"}, status=400)
    features = detect_result["features"]
    classify = detect_result["classify"]
    log_detection_event(user_id, classify["status"], "normal")
    response_payload = {
        "status": "success",
        "mode": "image",
        "inference_mode": classify.get("inference_mode", _resolve_classifier_mode()),
        "fatigue_level": classify["status"],
        "ear": float(features["ear"]),
        "mar": float(features["mar"]),
        "head_pose": {k: float(v) for k, v in features["head_pose"].items()},
        "image_with_landmarks": _image_to_base64(detect_result["annotated"]),
        "score": int(classify["score"]),
        "reasons": classify["reasons"],
        "confidence": float(classify.get("confidence", 0.0)),
    }
    elapsed = (time.perf_counter() - started_at) * 1000
    log_performance_event("api_detect_image", elapsed, cache_hit=cache_hit)
    return JsonResponse(response_payload)


@csrf_exempt
@api_view(["POST"])
def api_detect_frame(request):
    started_at = time.perf_counter()
    user_id = _get_user_id(request)
    payload = request.data
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    frame_b64 = payload.get("frame") if isinstance(payload, dict) else None
    persist = bool(payload.get("persist")) if isinstance(payload, dict) else False
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    close_session = bool(payload.get("close_session")) if isinstance(payload, dict) else False
    render_mode = str(payload.get("render_mode") or "frontend").strip().lower() if isinstance(payload, dict) else "frontend"
    if render_mode not in {"frontend", "backend"}:
        render_mode = "frontend"
    include_annotation = render_mode == "backend"
    session = None
    if persist or session_id:
        try:
            session = _resolve_session(session_id=session_id, user_id=user_id)
        except Exception:
            session = None
    frame = _decode_base64_image(frame_b64)
    if frame is None:
        return JsonResponse({"status": "error", "message": "frame参数无效，需base64图像"}, status=400)
    frame = _compress_image(frame, quality=75, max_size=640)
    try:
        detect_result = _EXECUTOR.submit(
            _detect_on_frame,
            frame,
            include_annotation,
            True,  # 开启静态兜底重试
        ).result()
        cache_hit = False
        
        # 第一次如果没检出，尝试用 CLAHE 增强图像对比度后再次检测
        if not detect_result:
            import cv2
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            detect_result = _EXECUTOR.submit(
                _detect_on_frame,
                enhanced_frame,
                include_annotation,
                True,
            ).result()
            cache_hit = False
            
    except Exception:
        _LOGGER.exception("api_detect_frame执行失败")
        return JsonResponse(
            {"status": "error", "message": "检测过程异常，请重试"},
            status=400,
        )
    if not detect_result:
        _clear_realtime_smoothing(session_id=session.id if session else session_id, user_id=user_id)
        warning = _WARNING_SYSTEM.update("alert")
        try:
            if session:
                _persist_detection_result(
                    session=session,
                    features={"ear": 0.0, "mar": 0.0, "head_pose": {"pitch": 0.0}},
                    fatigue_level="alert",
                    warning_level=warning["warning_level"],
                )
                _close_session_if_needed(session, close_session)
        except Exception:
            pass
        log_detection_event(user_id, "alert", warning["warning_level"])
        elapsed = (time.perf_counter() - started_at) * 1000
        log_performance_event("api_detect_frame", elapsed, cache_hit=cache_hit)
        return JsonResponse(
            {
                "status": "success",
                "inference_mode": _resolve_classifier_mode(),
                "fatigue_level": "alert",
                "warning_level": warning["warning_level"],
                "ear": 0.0,
                "mar": 0.0,
                "frame_count": warning["frame_count"],
                "session_id": session.id if session else None,
                "message": "未检测到人脸",
                "face_detected": False,  # 明确标记给前端
                "annotated_frame": f"data:image/jpeg;base64,{_image_to_base64(frame)}" if include_annotation else None,
                "render_mode": render_mode,
            }
        )
    raw_landmarks = detect_result.get("landmarks", {})
    smoothed_landmarks = _smooth_landmarks(
        raw_landmarks,
        session_id=session.id if session else session_id,
        user_id=user_id,
    )
    if smoothed_landmarks:
        detect_result["landmarks"] = smoothed_landmarks
        detect_result["features"] = _FEATURE_EXTRACTOR.extract_frame_features(
            smoothed_landmarks,
            smoothed_landmarks.get("image_size"),
        )
        if include_annotation:
            detect_result["annotated"] = _draw_landmarks(frame, smoothed_landmarks)
    detect_result["features"]["head_pose"] = _smooth_head_pose(
        detect_result["features"].get("head_pose"),
        session_id=session.id if session else session_id,
        user_id=user_id,
    )
    if detect_result.get("landmarks"):
        detect_result["classify"] = _classify_frame(
            _prepare_face_roi_for_cnn(frame, detect_result["landmarks"]),
            detect_result["features"],
        )
    features = detect_result["features"]
    classify = detect_result["classify"]
    raw_landmarks = detect_result.get("landmarks", {})
    json_landmarks = {}
    for k, v in raw_landmarks.items():
        if isinstance(v, np.ndarray):
            json_landmarks[k] = v.tolist()
        elif isinstance(v, list):
            json_landmarks[k] = v

    warning = _WARNING_SYSTEM.update(
        {
            "status": classify["status"],
            "inference_mode": classify.get("inference_mode", _resolve_classifier_mode()),
            "ear": float(features.get("ear", 0.0)),
            "mar": float(features.get("mar", 0.0)),
            "pitch": float(features.get("head_pose", {}).get("pitch", 0.0)),
        }
    )
    effective_status = warning.get("effective_status", classify["status"])
    try:
        if session:
            _persist_detection_result(
                session=session,
                features=features,
                fatigue_level=effective_status,
                warning_level=warning["warning_level"],
            )
            _close_session_if_needed(session, close_session)
            if close_session:
                _clear_realtime_smoothing(session_id=session.id, user_id=user_id)
    except Exception:
        pass
    log_detection_event(user_id, effective_status, warning["warning_level"])
    
    annotated_b64 = None
    if detect_result.get("annotated") is not None:
        annotated_b64 = _image_to_base64(detect_result["annotated"])

    payload = {
        "status": "success",
        "inference_mode": classify.get("inference_mode", _resolve_classifier_mode()),
        "fatigue_level": effective_status,
        "warning_level": warning["warning_level"],
        "ear": float(features["ear"]),
        "mar": float(features["mar"]),
        "frame_count": int(warning["frame_count"]),
        "trigger_alert": bool(warning["trigger_alert"]),
        "head_pose": {k: float(v) for k, v in features["head_pose"].items()},
        "score": int(classify["score"]),
        "reasons": classify["reasons"],
        "face_detected": True,  # 明确标记给前端
        "landmarks": json_landmarks,
        "annotated_frame": f"data:image/jpeg;base64,{annotated_b64}" if include_annotation and annotated_b64 else None,
        "render_mode": render_mode,

        "confidence": float(classify.get("confidence", 0.0)),
        "session_id": session.id if session else None,
    }
    elapsed = (time.perf_counter() - started_at) * 1000
    log_performance_event("api_detect_frame", elapsed, cache_hit=cache_hit)
    return JsonResponse(payload)


@csrf_exempt
@api_view(["GET"])
def api_get_config(request):
    from .utils.model_loader import get_cnn_classifier
    cnn_classifier = get_cnn_classifier()
    return JsonResponse(
        {
            "status": "success",
            "config": _CONFIG_MANAGER.get_config(),
            "classifier_mode": _resolve_classifier_mode(),
            "ml_model_ready": bool(_ML_CLASSIFIER and _ML_CLASSIFIER.is_ready()),
            "cnn_model_ready": bool(cnn_classifier and cnn_classifier.is_ready()),
        }
    )


@csrf_exempt
@api_view(["POST"])
def api_update_config(request):
    user_id = _get_user_id(request)
    form = ConfigForm(request.data)
    if not form.is_valid():
        return JsonResponse({"status": "error", "errors": form.errors}, status=400)
    data = form.cleaned_data
    old_config = _CONFIG_MANAGER.get_config()
    try:
        new_config = _CONFIG_MANAGER.update_config(
            {
                "ear_threshold": float(data["ear_threshold"]),
                "mar_threshold": float(data["mar_threshold"]),
                "pitch_threshold": float(data["pitch_threshold"]),
                "warning_frame_count": int(data["warning_frame_count"]),
                "emergency_frame_count": int(data["emergency_frame_count"]),
            }
        )
    except ValueError as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=400)
    _rebuild_models()
    _WARNING_SYSTEM.reset()
    log_config_change(user_id, old_config, new_config)
    return JsonResponse({"status": "success", "config": new_config})
