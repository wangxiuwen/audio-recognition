from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from recognition import SpeechRecognizer
import argparse
import tempfile
import os
from contextlib import asynccontextmanager

# 默认的空 lifespan 事件（用于 Gunicorn）
@asynccontextmanager
async def empty_lifespan(app: FastAPI):
    yield

# 完整的 lifespan 事件（用于 Uvicorn）
@asynccontextmanager
async def lifespan_event(app: FastAPI):
    try:
        provider = os.getenv('PROVIDER')
        app.state.recognizer = SpeechRecognizer(provider)
        app.state.models_loaded = True
        print(f"Lifespan: Recognizer initialized with provider {provider}")
    except Exception as e:
        error_msg = str(e)
        print(f"Error loading models on startup: {error_msg}")
        app.state.models_loaded = False
        raise RuntimeError(f"模型加载失败: {error_msg}") from e

    yield

    if hasattr(app.state, 'recognizer'):
        del app.state.recognizer
        print("Lifespan: Recognizer cleaned up")

# 检查是否通过 Gunicorn 启动
def is_gunicorn():
    return "gunicorn" in os.environ.get("SERVER_SOFTWARE", "") or "gunicorn" in os.environ.get("PROCESS_TYPE", "")

app = FastAPI(
    title="说话人分离和语音识别服务",
    description="基于深度学习的说话人分离和语音识别服务",
    version="1.0.0",
    lifespan=empty_lifespan if is_gunicorn() else lifespan_event
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "public")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def process_audio(audio: UploadFile = File(...)):
    """音频处理接口"""
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")

    try:
        content = await audio.read()
        results = app.state.recognizer.process_audio(content)
        
        return JSONResponse(content=results)

    except Exception as e:
        print(f"Error during audio processing: {e}")
        raise HTTPException(status_code=500, detail=f"音频处理失败：{str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run the FastAPI app with uvicorn.")
    parser.add_argument("--port", type=int, default=29999, help="Port to run the server on.")
    parser.add_argument("--provider", type=str, default=None, help="cpu cuda coreml. If not specified, will auto select the best provider.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.provider:
        os.environ['PROVIDER'] = args.provider  # 设置环境变量以供 lifespan 使用
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=args.port, workers=1)
