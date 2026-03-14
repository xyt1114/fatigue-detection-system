from django.db import models


class DetectionRecord(models.Model):
    image = models.ImageField(upload_to="uploads/")
    result = models.CharField(max_length=32)
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.result} ({self.confidence:.2f})"


class DetectionSession(models.Model):
    user_id = models.CharField(max_length=100)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    total_frames = models.IntegerField(default=0)
    fatigue_frames = models.IntegerField(default=0)
    max_warning_level = models.CharField(max_length=20, default="normal")

    class Meta:
        ordering = ["-start_time"]
        indexes = [
            models.Index(fields=["user_id", "start_time"]),
        ]

    def __str__(self):
        return f"{self.user_id}-{self.start_time:%Y%m%d%H%M%S}"


class DetectionLog(models.Model):
    session = models.ForeignKey(DetectionSession, on_delete=models.CASCADE, related_name="logs")
    timestamp = models.DateTimeField(auto_now_add=True)
    ear = models.FloatField(default=0.0)
    mar = models.FloatField(default=0.0)
    pitch = models.FloatField(default=0.0)
    fatigue_level = models.CharField(max_length=20, default="alert")
    warning_level = models.CharField(max_length=20, default="normal")

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["session", "timestamp"]),
            models.Index(fields=["warning_level"]),
        ]

    def __str__(self):
        return f"{self.session_id}-{self.fatigue_level}-{self.warning_level}"
