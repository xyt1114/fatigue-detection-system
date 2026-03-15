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
from .utils.logger import log_config_change, log_detection_event, log_performance_event, log_sampling_event
from .utils.model_loader import get_detector_instances, rebuild_runtime_instances

IMAGE_EXT = {"jpg", "jpeg", "png", "bmp"}
VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

_CONFIG_MANAGER = ConfigManager()
_FACE_DETECTOR, _STATIC_FACE_DETECTOR, _FEATURE_EXTRACTOR, _CLASSIFIER, _WARNING_SYSTEM = get_detector_instances()
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
    for key in ("left_eye", "right_eye", "mouth"):
        for point in landmarks[key]:
            cv2.circle(output, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
    return output


def _detect_with_detector(frame, detector, include_annotation=True):
    face_result = detector.detect(frame)
    if not face_result:
        return None
    landmarks = detector.get_landmarks(face_result)
    if not landmarks:
        return None
    try:
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
    frame_signature = _frame_hash(frame)
    cache_key = (
        f"detect:frame:{frame_signature}:{hash(config_signature)}:{int(include_annotation)}:{int(retry_static)}"
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
        return JsonResponse(
            {"status": "error", "message": "检测过程异常，请重试或更换素材"},
            status=400,
        )
    if not detect_result:
        message = "未检测到人脸"
        if suffix in VIDEO_EXT:
            message = "视频采样帧均未检测到人脸，请确保人脸清晰可见并位于画面中"
        elapsed = (time.perf_counter() - started_at) * 1000
        log_performance_event("api_detect_image", elapsed, cache_hit=cache_hit)
        return JsonResponse({"status": "error", "message": message}, status=400)
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
    try:
        detect_result, cache_hit = _EXECUTOR.submit(
            _detect_with_cache,
            frame,
            False,
            15,
            False,
        ).result()
    except Exception:
        return JsonResponse(
            {"status": "error", "message": "检测过程异常，请重试"},
            status=400,
        )
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
