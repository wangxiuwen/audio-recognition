from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from recognition import SpeechRecognizer
import argparse
import tempfile
import os

# 使用 lifespan 事件加载模型
async def lifespan_event(app: FastAPI):
    try:
        # 从环境变量读取 provider 选项
        provider = os.getenv('PROVIDER')
        # 初始化语音识别器
        app.state.recognizer = SpeechRecognizer(provider)
        app.state.models_loaded = True
    except Exception as e:
        error_msg = str(e)
        print(f"Error loading models on startup: {error_msg}")
        app.state.models_loaded = False
        raise RuntimeError(f"模型加载失败: {error_msg}") from e

    yield

    # 清理资源
    if hasattr(app.state, 'recognizer'):
        del app.state.recognizer

app = FastAPI(
    title="说话人分离和语音识别服务",
    description="基于深度学习的说话人分离和语音识别服务",
    version="1.0.0",
    lifespan=lifespan_event
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
        file_extension = os.path.splitext(audio.filename)[1]
        # Process audio content directly
        results = app.state.recognizer.process_audio(content, file_extension)
        
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
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=args.port, workers=1)
