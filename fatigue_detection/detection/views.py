import base64
import hashlib
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from .forms import ConfigForm, UploadForm
from .models import DetectionLog, DetectionSession
from .utils.config_manager import ConfigManager
from .utils.logger import log_config_change, log_detection_event, log_performance_event
from .utils.model_loader import get_detector_instances, rebuild_runtime_instances

IMAGE_EXT = {"jpg", "jpeg", "png", "bmp"}
VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

_CONFIG_MANAGER = ConfigManager()
_FACE_DETECTOR, _FEATURE_EXTRACTOR, _CLASSIFIER, _WARNING_SYSTEM = get_detector_instances()
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


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
    for key in ("left_eye", "right_eye", "mouth"):
        for point in landmarks[key]:
            cv2.circle(output, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
    return output


def _detect_on_frame(frame, include_annotation=True):
    face_result = _FACE_DETECTOR.detect(frame)
    if not face_result:
        return None
    landmarks = _FACE_DETECTOR.get_landmarks(face_result)
    if not landmarks:
        return None
    features = _FEATURE_EXTRACTOR.extract_frame_features(landmarks, face_result["image_size"])
    classify_result = _CLASSIFIER.classify(
        ear=features["ear"], mar=features["mar"], head_pose=features["head_pose"]
    )
    annotated = _draw_landmarks(frame, landmarks) if include_annotation else None
    return {
        "features": features,
        "classify": classify_result,
        "annotated": annotated,
    }


def _detect_with_cache(frame, include_annotation=True, cache_ttl=300):
    config_signature = str(sorted(_CONFIG_MANAGER.get_config().items()))
    frame_signature = _frame_hash(frame)
    cache_key = f"detect:frame:{frame_signature}:{hash(config_signature)}:{int(include_annotation)}"
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result, True
    detect_result = _detect_on_frame(frame, include_annotation=include_annotation)
    cache.set(cache_key, detect_result, timeout=cache_ttl)
    return detect_result, False


def _extract_frame_from_upload(uploaded_file):
    suffix = uploaded_file.name.lower().split(".")[-1] if "." in uploaded_file.name else ""
    if suffix in IMAGE_EXT:
        data = uploaded_file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if suffix in VIDEO_EXT:
        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=True) as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file.flush()
            cap = cv2.VideoCapture(temp_file.name)
            ok, frame = cap.read()
            cap.release()
            if ok:
                return frame
    return None


def index(request):
    return redirect("detection:upload_detect")


def upload_detect(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    context = {"form": form}
    if request.method == "POST" and form.is_valid():
        upload = form.cleaned_data["file"]
        frame = _extract_frame_from_upload(upload)
        if frame is None:
            context["error"] = "无法解析上传文件，请检查文件格式是否正确"
            return render(request, "detection/upload.html", context)
        frame = _compress_image(frame, quality=82, max_size=720)
        detect_result, _ = _detect_with_cache(frame, include_annotation=True, cache_ttl=300)
        if not detect_result:
            context["error"] = "未检测到人脸"
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
    frame = _extract_frame_from_upload(upload)
    if frame is None:
        return JsonResponse({"status": "error", "message": "无法解析上传文件"}, status=400)
    frame = _compress_image(frame, quality=80, max_size=640)
    config_signature = str(sorted(_CONFIG_MANAGER.get_config().items()))
    frame_signature = _frame_hash(frame)
    response_cache_key = f"detect:image:response:{frame_signature}:{hash(config_signature)}"
    cached_response = cache.get(response_cache_key)
    if cached_response is not None:
        elapsed = (time.perf_counter() - started_at) * 1000
        log_performance_event("api_detect_image", elapsed, cache_hit=True)
        return JsonResponse(cached_response)
    detect_result, cache_hit = _EXECUTOR.submit(
        _detect_with_cache,
        frame,
        True,
        300,
    ).result()
    if not detect_result:
        return JsonResponse({"status": "error", "message": "未检测到人脸"}, status=400)
    features = detect_result["features"]
    classify = detect_result["classify"]
    log_detection_event(user_id, classify["status"], "normal")
    response_payload = {
        "status": "success",
        "fatigue_level": classify["status"],
        "ear": float(features["ear"]),
        "mar": float(features["mar"]),
        "head_pose": {k: float(v) for k, v in features["head_pose"].items()},
        "image_with_landmarks": _image_to_base64(detect_result["annotated"]),
        "score": int(classify["score"]),
        "reasons": classify["reasons"],
    }
    cache.set(response_cache_key, response_payload, timeout=300)
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
    detect_result, cache_hit = _EXECUTOR.submit(
        _detect_with_cache,
        frame,
        False,
        15,
    ).result()
    if not detect_result:
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
                "fatigue_level": "alert",
                "warning_level": warning["warning_level"],
                "ear": 0.0,
                "mar": 0.0,
                "frame_count": warning["frame_count"],
                "session_id": session.id if session else None,
                "message": "未检测到人脸",
            }
        )
    features = detect_result["features"]
    classify = detect_result["classify"]
    warning = _WARNING_SYSTEM.update(classify["status"])
    try:
        if session:
            _persist_detection_result(
                session=session,
                features=features,
                fatigue_level=classify["status"],
                warning_level=warning["warning_level"],
            )
            _close_session_if_needed(session, close_session)
    except Exception:
        pass
    log_detection_event(user_id, classify["status"], warning["warning_level"])
    payload = {
        "status": "success",
        "fatigue_level": classify["status"],
        "warning_level": warning["warning_level"],
        "ear": float(features["ear"]),
        "mar": float(features["mar"]),
        "frame_count": int(warning["frame_count"]),
        "trigger_alert": bool(warning["trigger_alert"]),
        "head_pose": {k: float(v) for k, v in features["head_pose"].items()},
        "score": int(classify["score"]),
        "reasons": classify["reasons"],
        "session_id": session.id if session else None,
    }
    elapsed = (time.perf_counter() - started_at) * 1000
    log_performance_event("api_detect_frame", elapsed, cache_hit=cache_hit)
    return JsonResponse(payload)


@csrf_exempt
@api_view(["GET"])
def api_get_config(request):
    return JsonResponse({"status": "success", "config": _CONFIG_MANAGER.get_config()})


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
