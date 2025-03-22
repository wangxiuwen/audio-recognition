# 说话人分离和语音识别服务

基于深度学习的说话人分离和语音识别服务，支持多人对话场景下的语音转文字，并能够区分不同说话人。


1. 安装依赖

地址

```
https://k2-fsa.github.io/sherpa/onnx/cuda.html
```

```bash
# 根据自己的平台自行选择
wget https://hf-mirror.com/csukuangfj/sherpa-onnx-wheels/resolve/main/cuda/1.11.2/sherpa_onnx-1.11.2+cuda-cp313-cp313-linux_x86_64.whl
pip install sherpa_onnx-1.11.2+cuda-cp313-cp313-linux_x86_64.whl
python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)"
```

2. 下载模型

```
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
```

## 使用方法

### 启动服务

```bash
python main.py --port 9999
```

服务默认运行在 http://localhost:9999

### Web界面

访问 http://localhost:9999 即可打开Web界面，支持直接上传音频文件进行处理。

### API接口

#### 音频转写接口

- **URL**: `/transcribe`
- **方法**: POST
- **Content-Type**: multipart/form-data
- **参数**:
  - audio: 音频文件（支持wav、mp3等格式）

**响应示例**:
```json
{
    "result": [
        {
            "speaker": "SPEAKER_00",
            "text": "你好，很高兴见到你。",
            "start": 0.0,
            "end": 2.5
        },
        {
            "speaker": "SPEAKER_01",
            "text": "我也很高兴见到你。",
            "start": 2.8,
            "end": 4.6
        }
    ]
}
```

3. FAQ

```
Failed to load library libonnxruntime_providers_cuda.so with error: libcublasLt.so.11

apt install nvidia-cuda-toolkit
```

## 许可证

MIT License

