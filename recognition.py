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
import soundfile as sf


class SpeechRecognizer:
    def __init__(self, provider: str = 'cpu',num_speakers: int = -1, cluster_threshold: float = 0.5):
        """
        Initialize the SpeechRecognizer with speaker diarization and speech recognition models.
        Args:
            num_speakers: Number of speakers (-1 for auto-detection)
            cluster_threshold: Threshold for speaker clustering when num_speakers is -1
        """
        self.provider = provider
        self.num_speakers = num_speakers
        self.cluster_threshold = cluster_threshold
        self.sd = self._init_speaker_diarization()
        self.recognizer = self._init_recognizer()

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
            debug=True,
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
        print(f"Progress: {progress:.3f}%")
        return 0

    def process_audio(self, audio_path: str):
        """
        Process an audio file and return the transcription results.
        Args:
            audio_path: Path to the audio file
        Returns:
            List of dictionaries containing speaker ID, timestamps, and transcribed text
        """
        if not Path(audio_path).is_file():
            raise RuntimeError(f"{audio_path} does not exist")

        # Load audio file
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # only use the first channel

        if sample_rate != self.sd.sample_rate:
            raise RuntimeError(
                f"Expected sample rate: {self.sd.sample_rate}, given: {sample_rate}"
            )

        # Process speaker diarization
        print("Processing speaker diarization...")
        diarization_result = self.sd.process(audio, callback=self._progress_callback).sort_by_start_time()

        # Process speech recognition for each segment
        print("\nProcessing speech recognition...")
        results = []
        for segment in diarization_result:
            text = self._process_audio_segment(
                audio, sample_rate, segment.start, segment.end
            )
            results.append({
                "speaker": f"speaker_{segment.speaker:02}",
                "start": segment.start,
                "end": segment.end,
                "text": text
            })

        return results


def main():
    # Example usage
    recognizer = SpeechRecognizer()

    results = recognizer.process_audio("./0-four-speakers-zh.wav")

    # Print results
    print("\nTranscription Results:")
    print("-" * 80)
    for result in results:
        print(
            f"[{result['speaker']}] "
            f"{result['start']:.2f}s - {result['end']:.2f}s: "
            f"{result['text']}"
        )


if __name__ == "__main__":
    main()