# 疲劳驾驶检测系统 - 使用文档

## 安装步骤

1. 创建虚拟环境并激活

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 数据库迁移

```bash
python manage.py migrate
```

4. 运行服务

```bash
python manage.py runserver
```

## 功能说明

1. 文件上传检测
   - 入口：`/upload/`
   - 支持图片和视频上传
   - 展示疲劳等级、EAR、MAR、头姿态、预警信息
2. 实时摄像头检测
   - 入口：`/realtime/`
   - 支持摄像头切换
   - 10fps 上报帧检测，实时预警
3. 阈值配置
   - 读取/更新/恢复默认配置
   - 更新后立即生效，无需重启服务
4. 预警系统
   - 连续疲劳帧进入 warning
   - 连续重度疲劳帧进入 emergency

## API 文档

### 1) POST `/api/detect_image/`

- 请求：`multipart/form-data`
  - 字段：`file` 或 `image`
- 响应：

```json
{
  "status": "success",
  "fatigue_level": "alert|fatigue|severe_fatigue",
  "ear": 0.26,
  "mar": 0.41,
  "head_pose": {"pitch": 11.2, "yaw": 0.6, "roll": -0.3},
  "image_with_landmarks": "base64...",
  "score": 30,
  "reasons": ["mouth_open"]
}
```

### 2) POST `/api/detect_frame/`

- 请求：`application/json`

```json
{
  "frame": "data:image/jpeg;base64,...",
  "persist": true,
  "session_id": 1
}
```

- 响应：

```json
{
  "status": "success",
  "fatigue_level": "fatigue",
  "warning_level": "warning",
  "ear": 0.21,
  "mar": 0.65,
  "frame_count": 4,
  "trigger_alert": true,
  "head_pose": {"pitch": 33.0, "yaw": 1.1, "roll": 2.0},
  "score": 70,
  "reasons": ["eye_closed", "head_down"],
  "session_id": 1
}
```

### 3) GET `/api/get_config/`

- 响应：

```json
{
  "status": "success",
  "config": {
    "ear_threshold": 0.25,
    "mar_threshold": 0.6,
    "pitch_threshold": 30,
    "warning_frame_count": 3,
    "emergency_frame_count": 5
  }
}
```

### 4) POST `/api/update_config/`

- 请求：

```json
{
  "ear_threshold": 0.23,
  "mar_threshold": 0.62,
  "pitch_threshold": 28,
  "warning_frame_count": 4,
  "emergency_frame_count": 6
}
```

- 响应：

```json
{
  "status": "success",
  "config": {
    "ear_threshold": 0.23,
    "mar_threshold": 0.62,
    "pitch_threshold": 28,
    "warning_frame_count": 4,
    "emergency_frame_count": 6
  }
}
```

## 常见问题

1. 摄像头无法使用
   - 检查浏览器权限是否允许摄像头
   - 检查是否有其他应用占用摄像头
   - 通过 HTTPS 或 localhost 访问页面
2. 检测速度慢
   - 降低前端上报帧率（如 8fps）
   - 关闭不必要后台进程
   - 降低分辨率（如 640x360）
3. 准确率偏低
   - 提升光照条件，确保人脸完整入镜
   - 增加数据集并重新训练分类器
   - 调整阈值并结合实际场景标定
4. 配置更新不生效
   - 检查 `/api/update_config/` 返回是否成功
   - 检查阈值范围是否合法
   - 查看 `logs/detection.log` 配置变更日志
