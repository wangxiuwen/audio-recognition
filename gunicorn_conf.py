workers = 1
bind = "0.0.0.0:29999"
worker_class = "uvicorn.workers.UvicornWorker"  # 使用 UvicornWorker
timeout = 1200  # 增加超时时间，适用于大文件
loglevel = "debug"  # 设置为 debug 以便调试
threads = 1
preload_app = False  # 开启 preload_app，确保模型在 master 进程中加载
# import gc
# import os

# 设置环境变量
# os.environ['PROVIDER'] = 'cuda'
# os.environ['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'info')

# 冻结主进程内存，避免垃圾回收触发 copy-on-write
# def when_ready(server):
#     gc.freeze()  

# # 在主进程中加载模型
# def on_starting(server):
#     print("loading SpeechRecognizer... " + '*' * 20)
#     from recognition import SpeechRecognizer
#     import os
#     provider = os.getenv('PROVIDER', 'cpu')  # 默认使用 'cpu'，避免警告
#     try:
#         server.recognizer = SpeechRecognizer(provider)
#         server.models_loaded = True
#         print(f"Main process: Recognizer initialized with provider {provider}")
#     except Exception as e:
#         print(f"Failed to initialize recognizer in main process: {e}")
#         raise

# # 在 worker 进程中共享模型
# def post_fork(server, worker):
#     from main import app  # 直接导入 FastAPI 实例
#     try:
#         app.state.recognizer = server.recognizer
#         app.state.models_loaded = server.models_loaded
#         print(f"Worker {worker.pid}: Recognizer assigned from server")
#     except Exception as e:
#         print(f"Worker {worker.pid}: Failed to assign recognizer - {e}")
#         raise
