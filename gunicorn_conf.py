workers = 4  # 根据 CPU 核心数调整 (一般: 核心数*2)
bind = "0.0.0.0:29999"
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120  # 增加超时时间，适用于大文件
loglevel = "info"