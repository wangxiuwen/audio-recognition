from pydantic import BaseModel, Field
from .system import SystemConfig
from .asr import ASRConfig
from .vad import VADConfig

class Config(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    system_config: SystemConfig = Field(default_factory=SystemConfig, alias="system_config")
    asr_config: ASRConfig = Field(default_factory=ASRConfig, alias="asr_config")
    vad_config: VADConfig = Field(default_factory=VADConfig, alias="vad_config")
