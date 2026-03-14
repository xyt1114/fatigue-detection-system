from django.contrib import admin

from .models import DetectionLog, DetectionRecord, DetectionSession

admin.site.register(DetectionRecord)
admin.site.register(DetectionSession)
admin.site.register(DetectionLog)
