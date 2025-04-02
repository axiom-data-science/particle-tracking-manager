"""Particle Tracking Manager."""

import logging

# Set log levels for third paries to WARNING
logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.getLogger('kerchunk').setLevel(logging.WARNING)
logging.getLogger('opendrift').setLevel(logging.WARNING)
logging.getLogger('numcodecs').setLevel(logging.WARNING)

from .models.opendrift.config_opendrift import OpenDriftConfig, LarvalFishModelConfig, LeewayModelConfig, OceanDriftModelConfig, OpenOilModelConfig
from .models.opendrift.opendrift import OpenDriftModel
from .config_the_manager import TheManagerConfig