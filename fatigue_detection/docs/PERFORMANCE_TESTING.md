# 性能测试说明

## 目标

- 首页加载时间 < 1 秒
- 图片检测响应 < 500ms
- 实时检测延迟 < 100ms（单帧平均）

## 测试准备

1. 启动服务

```bash
python manage.py runserver
```

2. 打开浏览器开发者工具
   - Network 面板查看资源加载耗时
   - Performance 面板查看主线程开销

3. 准备测试素材
   - 1 张标准人脸图片（640~1280分辨率）
   - 实时摄像头场景（稳定光照）

## 测试步骤

### 1) 首页与页面加载

- 访问 `/upload/` 与 `/realtime/`
- 清空缓存后硬刷新 3 次
- 记录 `DOMContentLoaded` 与 `Load` 时间
- 对比静态资源缓存命中情况（含 `?v=` 版本号）

### 2) 图片检测接口延迟

```bash
curl -X POST "http://127.0.0.1:8000/api/detect_image/" -F "file=@C:/path/demo.jpg"
```

- 连续执行 10 次
- 记录平均响应时间
- 第 2 次起应出现缓存收益
- 查看 `logs/detection.log` 的 `event=performance endpoint=api_detect_image`

### 3) 实时检测延迟

- 打开 `/realtime/` 点击“开始检测”
- 默认帧率为 5fps
- 观察前端流畅度与预警切换延迟
- 查看 `logs/detection.log` 的 `event=performance endpoint=api_detect_frame`

## 关键验证点

- 模型实例是否预加载：应用启动后首次请求延迟下降
- 压缩是否生效：上传帧最长边不超过 640
- 缓存是否生效：重复帧请求延迟显著降低
- 前端按需上报是否生效：静止场景请求频率下降

## 监控建议

- 每日汇总：
  - `api_detect_image` p50/p95
  - `api_detect_frame` p50/p95
  - 缓存命中率
- 异常阈值：
  - 图片接口 > 800ms 连续 5 次触发告警
  - 实时接口 > 200ms 连续 20 次触发告警
