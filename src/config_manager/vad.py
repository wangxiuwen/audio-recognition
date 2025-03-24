from pydantic import ValidationInfo, Field, model_validator, BaseModel
from typing import Literal, Optional, Dict, ClassVar

class SileroVADConfig(BaseModel):
    """Configuration for Silero VAD service."""

    orig_sr: int = Field(..., alias="orig_sr")  # 16000
    target_sr: int = Field(..., alias="target_sr")  # 16000
    prob_threshold: float = Field(..., alias="prob_threshold")  # 0.4
    db_threshold: int = Field(..., alias="db_threshold")  # 60
    required_hits: int = Field(..., alias="required_hits")  # 3 * (0.032) = 0.1s
    required_misses: int = Field(..., alias="required_misses")  # 24 * (0.032) = 0.8s
    smoothing_window: int = Field(..., alias="smoothing_window")  # 5



class VADConfig(BaseModel):
    """Configuration for Automatic Speech Recognition."""

    vad_model: Literal["silero_vad",] = Field(..., alias="vad_model")
    silero_vad: Optional[SileroVADConfig] = Field(None, alias="silero_vad")


    @model_validator(mode="after")
    def check_asr_config(cls, values: "VADConfig", info: ValidationInfo):
        vad_model = values.silero_vad

        # Only validate the selected ASR model
        if vad_model == "silero_vad" and values.silero_vad is not None:
            values.silero_vad.model_validate(values.silero_vad.model_dump())

        return values
