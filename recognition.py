#!/usr/bin/env python3

"""
This file combines speaker diarization and speech recognition functionality.
It can:
1. Detect different speakers and their speaking time segments
2. Transcribe the speech content for each speaker
3. Output results in chronological order with speaker ID, timestamps, and content
"""

from pathlib import Path
import datetime as dt

import sherpa_onnx
import logging
import soundfile as sf

import coloredlogs

# 配置带颜色的日志输出
coloredlogs.install(
    level='INFO',
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    field_styles={
        'asctime': {'color': 'green'},
        'levelname': {'bold': True, 'color': 'black'},
        'name': {'color': 'blue'}
    },
    level_styles={
        'debug': {'color': 'cyan'},
        'info': {'color': 'green'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red'},
        'critical': {'color': 'red', 'bold': True}
    }
)

class SpeechRecognizer:
    def __init__(self, provider: str = None, num_speakers: int = -1, cluster_threshold: float = 0.5):

        self.provider = provider or self._auto_select_provider()
        logging.info(f"Using provider: {self.provider}")

        self.num_speakers = num_speakers
        self.cluster_threshold = cluster_threshold
        self.sd = self._init_speaker_diarization()
        self.recognizer = self._init_recognizer()

    def _auto_select_provider(self) -> str:
        """
        Automatically select the best available hardware provider.
        Returns:
            The selected provider ('cuda', 'coreml', or 'cpu')
        """
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass

        try:
            import coremltools
            return 'coreml'
        except ImportError:
            pass

        return 'cpu'

    def _init_speaker_diarization(self):
        """
        Initialize the speaker diarization model.
        """
        segmentation_model = "./models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
        embedding_extractor_model = (
            "./models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
        )

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=segmentation_model
                ),
                provider=self.provider,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=embedding_extractor_model,
                provider=self.provider,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=self.num_speakers, threshold=self.cluster_threshold,
            ),
            min_duration_on=0.3,  # 最小讲话时长 0.3 秒
            min_duration_off=0.5,  # 讲话结束后至少 0.5 秒才认为讲话结束
        )
        if not config.validate():
            raise RuntimeError(
                "Please check your config and make sure all required files exist"
            )

        return sherpa_onnx.OfflineSpeakerDiarization(config)

    def _init_recognizer(self):
        """
        Initialize the speech recognition model.
        """
        model = "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx"
        tokens = "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"

        if not Path(model).is_file():
            raise ValueError(
                "Please download model files from "
                "https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models"
            )

        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model,
            tokens=tokens,
            use_itn=True,
            debug=False,
            provider=self.provider,
        )

    def _process_audio_segment(self, audio, sample_rate, start_time, end_time):
        """
        Process an audio segment and return its transcription.
        """
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment = audio[start_sample:end_sample]

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, segment)
        self.recognizer.decode_stream(stream)

        return stream.result.text

    def _progress_callback(self, num_processed_chunk: int, num_total_chunks: int) -> int:
        progress = num_processed_chunk / num_total_chunks * 100
        logging.info(f"Progress: {progress:.3f}%")
        return 0

    def process_audio(self, audio_content: bytes, file_extension: str):
        """
        Process audio content directly and return the transcription results.
        Args:
            audio_content: The audio file content in bytes
            file_extension: The file extension of the audio file
        Returns:
            List of dictionaries containing speaker ID, timestamps, and transcribed text
        """
        import io
        import soundfile as sf

        # Create in-memory file-like object
        audio_file = io.BytesIO(audio_content)
        audio_file.name = f"audio.{file_extension}"

        # Load audio file
        audio, sample_rate = sf.read(audio_file, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # only use the first channel

        if sample_rate != self.sd.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sd.sample_rate)
            sample_rate = self.sd.sample_rate

        # Process speaker diarization
        logging.info("Processing speaker diarization...")
        start_time = dt.datetime.now()
        diarization_result = self.sd.process(audio, callback=self._progress_callback).sort_by_start_time()
        diarization_time = dt.datetime.now() - start_time

        # Process speech recognition for each segment
        logging.info("Processing speech recognition...")
        sentences = []

        asr_start_time = dt.datetime.now()
        for segment in diarization_result:
            text = self._process_audio_segment(
                audio, sample_rate, segment.start, segment.end
            )
            sentences.append({
                "speaker": f"speaker_{segment.speaker:02}",
                "start": segment.start,
                "end": segment.end,
                "text": text
            })
        asr_time = dt.datetime.now() - asr_start_time

        return {
            'sentences': sentences,
            'diarization_time': diarization_time.total_seconds(),
            'asr_time': asr_time.total_seconds()
        }


def main():
    # Example usage
    recognizer = SpeechRecognizer()
    results = recognizer.process_audio("./tests/0-four-speakers-zh.wav")

    # Print results
    logging.info("Transcription Results:")
    print("-" * 80)
    for result in results:
        print(
            f"[{result['speaker']}] "
            f"{result['start']:.2f}s - {result['end']:.2f}s: "
            f"{result['text']}"
        )


if __name__ == "__main__":
    main()