# 生产环境部署指南（Gunicorn + Nginx）

## 部署拓扑

- Nginx：静态资源、反向代理、TLS
- Gunicorn：运行 Django WSGI
- 数据库：SQLite（开发）/ PostgreSQL（生产建议）

## 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
python manage.py migrate
python manage.py collectstatic --noinput
```

## 2. Django 生产配置建议

- `DEBUG=False`
- 配置强随机 `SECRET_KEY`
- 限制 `ALLOWED_HOSTS`
- 使用 PostgreSQL 并配置备份
- 配置 `CSRF_TRUSTED_ORIGINS` 与 HTTPS

## 3. Gunicorn 启动

```bash
gunicorn fatigue_detection.wsgi:application --bind 127.0.0.1:8001 --workers 3 --threads 2 --timeout 60
```

建议通过 systemd 托管：

```ini
[Unit]
Description=Fatigue Detection Gunicorn
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/srv/fatigue_detection
Environment="PATH=/srv/fatigue_detection/.venv/bin"
ExecStart=/srv/fatigue_detection/.venv/bin/gunicorn fatigue_detection.wsgi:application --bind 127.0.0.1:8001 --workers 3 --threads 2 --timeout 60
Restart=always

[Install]
WantedBy=multi-user.target
```

## 4. Nginx 配置示例

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 120m;

    location /static/ {
        alias /srv/fatigue_detection/static/;
    }

    location /media/ {
        alias /srv/fatigue_detection/media/;
    }

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 5. HTTPS 与安全

- 使用 Let’s Encrypt 申请证书
- 强制 HTTP 跳转 HTTPS
- 为管理后台启用强密码和最小权限
- 仅对可信来源开放配置修改 API

## 6. 监控与运维建议

- 日志：
  - `logs/project.log`
  - `logs/detection.log`（按天轮转）
- 指标：
  - 接口 RT、错误率、CPU/内存
  - 帧处理速率与预警触发频率
- 定时任务：
  - 清理过旧会话日志
  - 数据库备份与恢复演练
