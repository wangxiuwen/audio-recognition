from pydantic import ValidationInfo, Field, model_validator, BaseModel
from typing import Literal, Optional, Dict, ClassVar


class SystemConfig(BaseModel):
    """System configuration settings."""

    host: str = Field(..., alias="host")
    port: int = Field(..., alias="port")

    @model_validator(mode="after")
    def check_port(cls, values):
        port = values.port
        if port < 0 or port > 65535:
            raise ValueError("Port must be between 0 and 65535")
        return values
