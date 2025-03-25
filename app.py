import gradio as gr
import numpy as np
from loguru import logger
from src.service_context import ServiceContext
from src.config_manager.utils import Config, read_yaml, validate_config
import asyncio

config: Config = validate_config(read_yaml("config.yaml"))

default_context_cache = ServiceContext()
default_context_cache.load_from_config(config)

async def process_audio_vad(audio):
    try:
        if audio is None:
            raise ValueError("未提供音频文件")

        sample_rate, audio_array = audio  # Gradio 传入的是 (sample_rate, numpy_array)

        if len(audio_array) == 0:
            raise ValueError("无效的音频数据：音频为空")

        # 归一化音频数据到 -1 到 1
        audio_array = audio_array.astype(np.float32)
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / 32768.0  # 将16位整数转换为-1到1之间的浮点数
        logger.info(f"音频采样率: {sample_rate}, 音频数据形状: {audio_array.shape}, 音频数据类型: {audio_array.dtype}")

        # 使用 VAD 检测语音活动
        vad_results = list(default_context_cache.vad_engine.detect_speech(audio_array))
        if len(vad_results) == 0:
            logger.warning("VAD未检测到语音片段")
            return "未检测到有效的语音片段"

        transcriptions = []
        for segment in vad_results:
            logger.info(f"VAD ----segment: {segment}")
            if isinstance(segment, tuple) and len(segment) == 3:
                logger.info(f"VAD ----: {start} {end}")
                start, end, audio_bytes = segment  # 解析 (start, end, audio_bytes)
                logger.info(start, end, len(audio_bytes))

                if audio_bytes == b"<|PAUSE|>":
                    logger.info("检测到暂停信号")
                    continue
                elif audio_bytes == b"<|RESUME|>":
                    logger.info("检测到恢复信号")
                    continue
                elif len(audio_bytes) > 1024:
                    # 进行 ASR 语音转录
                    segment_audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    text = await default_context_cache.asr_engine.async_transcribe_np(segment_audio)

                    transcriptions.append({
                        'text': text,
                        'start': start,
                        'end': end
                    })

        logger.info(f"Transcription results: {transcriptions}")
        output = {
            "transcription": ' '.join([t['text'] for t in transcriptions]),
            "timestamps": [{'text': t['text'], 'start': t['start'], 'end': t['end']} for t in transcriptions]
        }
        return f"转录结果: {output['transcription']}\n时间戳: {output['timestamps']}"

    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        return f"处理失败：{str(e)}"


async def process_audio(audio):
    try:
        if audio is None:
            raise ValueError("未提供音频文件")

        sample_rate, audio_array = audio  # Gradio 传入的是 (sample_rate, numpy_array)

        if len(audio_array) == 0:
            raise ValueError("无效的音频数据：音频为空")

        # 归一化音频数据到 -1 到 1
        audio_array = audio_array.astype(np.float32)
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / 32768.0  # 将16位整数转换为-1到1之间的浮点数
        logger.info(f"音频采样率: {sample_rate}, 音频数据形状: {audio_array.shape}, 音频数据类型: {audio_array.dtype}")

        # 直接进行ASR语音识别
        text = await default_context_cache.asr_engine.async_transcribe_np(audio_array)

        return f"转录结果: {text}"

    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        return f"处理失败：{str(e)}"


def create_ui():
    """Create the Gradio interface"""
    with gr.Blocks(title="音频处理系统") as interface:
        gr.Markdown("# 音频处理系统")
        
        with gr.Row():
            audio_input = gr.Audio(
                label="上传音频文件",
                sources=["upload"]
            )
        
        with gr.Row():
            process_btn = gr.Button("开始处理", variant="primary")
        
        with gr.Row():
            output_text = gr.Textbox(
                label="处理结果",
                placeholder="处理结果将在这里显示...",
                lines=10
            )
        
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[output_text]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_ui()
    interface.launch(debug=True, server_name="0.0.0.0", server_port=29999, share=False)
