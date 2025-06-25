# gunicorn.conf.py
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
preload_app = True