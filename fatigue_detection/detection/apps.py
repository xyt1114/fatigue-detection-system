from django.apps import AppConfig


class DetectionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detection"

    def ready(self):
        try:
            from .utils.model_loader import get_detector_instances

            get_detector_instances()
        except Exception:
            pass
