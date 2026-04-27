import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-secret-key-change-in-production")
DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "detection.apps.DetectionConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.gzip.GZipMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "fatigue_detection.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "fatigue_detection.context_processors.static_version",
            ],
        },
    },
]

WSGI_APPLICATION = "fatigue_detection.wsgi.application"
ASGI_APPLICATION = "fatigue_detection.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "zh-hans"
TIME_ZONE = "Asia/Shanghai"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_VERSION = os.getenv("STATIC_VERSION", "20260318-ml-badge")
if not DEBUG:
    STATICFILES_STORAGE = "django.contrib.staticfiles.storage.ManifestStaticFilesStorage"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
FFMPEG_BIN = os.getenv("FFMPEG_BIN", r"C:\ffmpeg\bin\ffmpeg.exe")
CLASSIFIER_MODE = os.getenv("CLASSIFIER_MODE", "cnn").strip().lower()
ML_MODEL_PATH = Path(os.getenv("ML_MODEL_PATH", BASE_DIR / "models" / "fatigue_classifier_grouped.joblib"))
CNN_MODEL_PATH = Path(os.getenv("CNN_MODEL_PATH", BASE_DIR / "models" / "fatigue_classifier_cnn.pt"))

CORS_ALLOW_ALL_ORIGINS = True

FILE_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 120 * 1024 * 1024

DATASET_ROOT = Path(os.getenv("DATASET_ROOT", BASE_DIR / "dataset"))

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "fatigue-detection-cache",
        "TIMEOUT": 300,
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

FATIGUE_CONFIG = {
    "EAR_THRESHOLD": 0.25,
    "MAR_THRESHOLD": 0.6,
    "PITCH_THRESHOLD": 30,
    "WARNING_FRAME_COUNT": 5,
    "EMERGENCY_FRAME_COUNT": 7,
    "FPS_HINT": 5.0,
    "ML_FATIGUE_MIN_FRAMES": 3,
    "ML_SEVERE_MIN_FRAMES": 4,
    "BLINK_MAX_DURATION_SEC": 0.28,
    "YAWN_WARNING_SEC": 0.8,
    "YAWN_EMERGENCY_SEC": 1.4,
    "VIDEO_SEVERE_EMERGENCY_MIN_FRAMES": 8,
    "VIDEO_SEVERE_EMERGENCY_MIN_RATIO": 0.25,
    "VIDEO_WARNING_MIN_RATIO": 0.20,
}

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOG_DIR / "project.log",
            "formatter": "standard",
            "encoding": "utf-8",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "root": {
        "handlers": ["file", "console"],
        "level": "INFO",
    },
}
