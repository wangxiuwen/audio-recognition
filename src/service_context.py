import os
import json

from loguru import logger

from .asr.asr_interface import ASRInterface
from .vad.vad_interface import VADInterface

from .asr.asr_factory import ASRFactory
from .vad.vad_factory import VADFactory

from .config_manager import (
    Config,
    SystemConfig,
    ASRConfig,
    VADConfig,
    read_yaml,
    validate_config,
)


class ServiceContext:
    """Initializes, stores, and updates the asr, tts, and llm instances and other
    configurations for a connected client."""

    def __init__(self, config: Config | None = None):
        """Initialize the service context with optional configuration.

        Args:
            config: Optional configuration to initialize with
        """
        self.config: Config = None
        self.system_config: SystemConfig = None
        self.asr_engine: ASRInterface = None
        self.vad_engine: VADInterface | None = None
        self.system_prompt: str = None
        
        if config:
            self.load_from_config(config)

    def __str__(self):
        return (
            f"ServiceContext:\n"
            f"  System Config: {'Loaded' if self.system_config else 'Not Loaded'}\n"
            f"    Details: {json.dumps(self.system_config.model_dump(), indent=6) if self.system_config else 'None'}\n"
            f"  ASR Engine: {type(self.asr_engine).__name__ if self.asr_engine else 'Not Loaded'}\n"
            f"  VAD Engine: {type(self.vad_engine).__name__ if self.vad_engine else 'Not Loaded'}\n"
        )

    # ==== Initializers

    def load_cache(
        self,
        config: Config,
        system_config: SystemConfig,
        asr_engine: ASRInterface,
        vad_engine: VADInterface,
    ) -> None:
        """
        Load the ServiceContext with the reference of the provided instances.
        Pass by reference so no reinitialization will be done.
        """
        if not system_config:
            raise ValueError("system_config cannot be None")

        self.config = config
        self.system_config = system_config
        self.asr_engine = asr_engine
        self.vad_engine = vad_engine

    def load_from_config(self, config: Config) -> None:
        """
        Load the ServiceContext with the config.
        Reinitialize the instances if the config is different.

        Parameters:
        - config (Dict): The configuration dictionary.
        """
        if not self.config:
            self.config = config

        if not self.system_config:
            self.system_config = config.system_config

        # init asr from character config
        self.init_asr(config.asr_config)

        # init vad from character config
        self.init_vad(config.vad_config)

        # store typed config references
        self.config = config
        self.system_config = config.system_config or self.system_config

    def init_asr(self, asr_config: ASRConfig) -> None:
        if not self.asr_engine or (self.asr_config != asr_config):
            logger.info(f"Initializing ASR: {asr_config.asr_model}")
            self.asr_engine = ASRFactory.get_asr_system(
                asr_config.asr_model,
                **getattr(asr_config, asr_config.asr_model).model_dump(),
            )
            # saving config should be done after successful initialization
            self.asr_config = asr_config
        else:
            logger.info("ASR already initialized with the same config.")

    def init_vad(self, vad_config: VADConfig) -> None:
        """Initialize or update the VAD engine with the given configuration.

        Args:
            vad_config: Configuration for VAD engine
        """
        if not self.vad_engine or (self.vad_config != vad_config):
            logger.info(f"Initializing VAD: {vad_config.vad_model}")
            self.vad_engine = VADFactory.get_vad_engine(
                vad_config.vad_model,
                **getattr(vad_config, vad_config.vad_model.lower()).model_dump(),
            )
            self.vad_config = vad_config
        else:
            logger.info("VAD already initialized with the same config.")

    def initialize_services(self) -> None:
        """Initialize all services using the current configuration."""
        if not self.config:
            raise ValueError("No configuration loaded")

        self.init_vad(self.config.vad_config)
        self.init_asr(self.config.asr_config)

def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1, prioritizing values from dict2.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
