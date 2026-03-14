from django.urls import path

from . import views

app_name = "detection"

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload_detect, name="upload_detect"),
    path("realtime/", views.realtime_detect, name="realtime_detect"),
    path("api/detect_image/", views.api_detect_image, name="api_detect_image"),
    path("api/detect_frame/", views.api_detect_frame, name="api_detect_frame"),
    path("api/get_config/", views.api_get_config, name="api_get_config"),
    path("api/update_config/", views.api_update_config, name="api_update_config"),
]
