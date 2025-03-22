from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from recognition import SpeechRecognizer
import argparse
import os

# 使用 lifespan 来加载模型
async def lifespan_event(app: FastAPI):
    try:
        # 读取 provider 选项
        provider = args.provider  # 从命令行参数获取 provider
        print(f"Using provider: {provider}")

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
        # 保存上传的音频文件
        file_path = f"temp_{audio.filename}"
        with open(file_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)

        # 处理音频文件
        results = app.state.recognizer.process_audio(file_path)

        # 删除临时文件
        os.remove(file_path)

        return JSONResponse(content={'results': results})

    except Exception as e:
        print(f"Error during audio processing: {e}")
        # 确保临时文件被删除
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"音频处理失败：{str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run the FastAPI app with uvicorn.")
    parser.add_argument("--port", type=int, default=29999, help="Port to run the server on.")
    parser.add_argument("--provider", type=str, default='cpu', help="cpu cuda coreml.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
