# 疲劳驾驶检测系统

基于 Django + MediaPipe + OpenCV 的疲劳驾驶检测系统，支持本地文件检测、实时摄像头检测、阈值在线配置和预警状态管理。

## 技术栈

- Django 4.2
- MediaPipe
- OpenCV
- NumPy / scikit-learn
- Bootstrap 5

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

打开：

- 上传检测：http://127.0.0.1:8000/upload/
- 实时检测：http://127.0.0.1:8000/realtime/

## 功能演示

- 本地文件上传检测（图片/视频）
- 实时摄像头检测（10fps）
- 疲劳等级与预警级别显示
- 阈值在线读取/更新/恢复默认
- 检测会话与逐帧日志持久化（可选）

## 项目结构

```text
fatigue_detection/
├── detection/                # 核心应用
│   ├── templates/detection/  # 页面模板
│   ├── utils/                # 检测算法与配置/日志模块
│   ├── models.py             # 检测记录与会话日志模型
│   ├── views.py              # 页面与 API 接口
│   └── tests.py              # 单元测试
├── static/                   # 前端资源（css/js/audio）
├── dataset/                  # 数据集目录
├── scripts/                  # 训练与校验脚本
└── docs/                     # 使用、优化、部署文档
```

## 文档

- 使用文档：`docs/USAGE.md`
- 性能优化建议：`docs/PERFORMANCE_OPTIMIZATION.md`
- 部署指南：`docs/DEPLOYMENT.md`

## 许可证

MIT
