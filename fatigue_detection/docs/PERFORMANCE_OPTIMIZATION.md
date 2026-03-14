# 性能优化建议

## 目标

- 降低实时检测延迟
- 提高帧处理吞吐
- 降低数据库与 API 压力

## 优化方向

1. 模型优化（ONNX Runtime）
   - 将关键推理模型导出 ONNX
   - 使用 ONNX Runtime CPU Execution Provider
   - 启用图优化和线程参数调优

2. 缓存策略
   - 对配置接口 `/api/get_config/` 做短时缓存
   - 对重复帧（近似哈希）跳过重复计算
   - 对检测会话统计结果做缓存

3. 异步任务化（Celery）
   - 大视频文件检测通过 Celery + Redis 异步执行
   - 前端轮询任务状态
   - 主 API 专注实时小请求，避免阻塞

4. 前端优化（WebWorker）
   - 帧压缩和 base64 编码放入 WebWorker
   - 主线程仅负责渲染与交互
   - 降低 UI 卡顿

5. 数据库优化
   - 对 `DetectionLog(session, timestamp)` 建联合索引
   - 对 `DetectionSession(start_time, user_id)` 建索引
   - 日志定期归档，减少热数据体量

6. API 与传输优化
   - 前端固定 8~10fps，动态降帧
   - JPEG 质量 0.6~0.75
   - 异常时退避重试，避免抖动风暴

## 可执行优化路线

### 阶段1（低成本）

- 降帧/降分辨率
- 开启日志轮转
- 为高频查询字段加索引

### 阶段2（中成本）

- 接入 Celery 异步视频检测
- 接入 Redis 配置/会话缓存

### 阶段3（高收益）

- ONNX Runtime 推理替换
- 前端 Worker 化 + 指标监控闭环
