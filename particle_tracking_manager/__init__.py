"""Particle Tracking Manager."""

import logging


# Set log levels for third paries to WARNING
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("kerchunk").setLevel(logging.WARNING)
logging.getLogger("opendrift").setLevel(logging.WARNING)
logging.getLogger("numcodecs").setLevel(logging.WARNING)

from .config_the_manager import TheManagerConfig
from .models import opendrift as opendrift_models
from .models.opendrift.config_opendrift import (
    HarmfulAlgalBloomModelConfig,
    LarvalFishModelConfig,
    LeewayModelConfig,
    OceanDriftModelConfig,
    OpenDriftConfig,
    OpenOilModelConfig,
)
from .models.opendrift.opendrift import OpenDriftModel


__all__ = [
    "TheManagerConfig",
    "LarvalFishModelConfig",
    "LeewayModelConfig",
    "OceanDriftModelConfig",
    "OpenDriftConfig",
    "OpenOilModelConfig",
    "OpenDriftModel",
    "HarmfulAlgalBloomModelConfig",
    "opendrift_models",
]
