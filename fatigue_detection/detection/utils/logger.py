import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from django.conf import settings


def _build_detection_logger():
    logger = logging.getLogger("detection")
    if logger.handlers:
        return logger
    log_dir = Path(getattr(settings, "LOG_DIR", settings.BASE_DIR / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "detection.log"
    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    handler.suffix = "%Y-%m-%d"
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    return logger


detection_logger = _build_detection_logger()


def log_detection_event(user_id, fatigue_level, warning_level):
    detection_logger.info(
        "event=detection user_id=%s fatigue_level=%s warning_level=%s",
        user_id,
        fatigue_level,
        warning_level,
    )


def log_config_change(user_id, old_config, new_config):
    detection_logger.info(
        "event=config_change user_id=%s old_config=%s new_config=%s",
        user_id,
        old_config,
        new_config,
    )


def log_performance_event(endpoint, elapsed_ms, cache_hit=False):
    detection_logger.info(
        "event=performance endpoint=%s elapsed_ms=%.2f cache_hit=%s",
        endpoint,
        elapsed_ms,
        cache_hit,
    )
