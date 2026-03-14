from django import forms


class UploadForm(forms.Form):
    file = forms.FileField(label="选择文件", help_text="支持图片(jpg/png)或视频(mp4/avi)")

    def clean_file(self):
        file = self.cleaned_data["file"]
        suffix = file.name.lower().split(".")[-1] if "." in file.name else ""
        image_ext = {"jpg", "jpeg", "png", "bmp"}
        video_ext = {"mp4", "avi", "mov", "mkv"}
        if suffix not in image_ext | video_ext:
            raise forms.ValidationError("仅支持图片(jpg/png/bmp)或视频(mp4/avi/mov/mkv)")
        image_limit = 10 * 1024 * 1024
        video_limit = 100 * 1024 * 1024
        if suffix in image_ext and file.size > image_limit:
            raise forms.ValidationError("图片大小不能超过10MB")
        if suffix in video_ext and file.size > video_limit:
            raise forms.ValidationError("视频大小不能超过100MB")
        return file


class ConfigForm(forms.Form):
    ear_threshold = forms.FloatField(min_value=0.1, max_value=0.4)
    mar_threshold = forms.FloatField(min_value=0.3, max_value=0.8)
    pitch_threshold = forms.FloatField(min_value=10, max_value=60)
    warning_frame_count = forms.IntegerField(min_value=1, max_value=30)
    emergency_frame_count = forms.IntegerField(min_value=1, max_value=60)

    def clean(self):
        cleaned_data = super().clean()
        warning_count = cleaned_data.get("warning_frame_count")
        emergency_count = cleaned_data.get("emergency_frame_count")
        if warning_count is not None and emergency_count is not None and emergency_count < warning_count:
            raise forms.ValidationError("紧急帧阈值必须大于或等于预警帧阈值")
        return cleaned_data
