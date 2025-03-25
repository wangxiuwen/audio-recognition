from pywhispercpp.model import Model

import numpy as np
from loguru import logger
from .asr_interface import ASRInterface


class VoiceRecognition(ASRInterface):
    def __init__(
        self,
        model_name: str = "base",
        model_dir="models",
        language: str = None,
        print_realtime=True,
        print_progress=True,
        **kwargs,
    ) -> None:
        self.model = Model(
            model=model_name,
            models_dir=model_dir,
            language= self.LANG if self.LANG else None,
            print_realtime=print_realtime,
            print_progress=print_progress,
            debug=True,
            **kwargs,
        )

    def transcribe_np(self, audio: np.ndarray) -> str:
        segments = self.model.transcribe(audio, new_segment_callback=logger.info)
        full_text = ""
        for segment in segments:
            full_text += segment.text
        return full_text
