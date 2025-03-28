"""Particle Tracking Manager."""

import logging

# Set log levels for third paries to WARNING
logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.getLogger('kerchunk').setLevel(logging.WARNING)
logging.getLogger('opendrift').setLevel(logging.WARNING)
logging.getLogger('numcodecs').setLevel(logging.WARNING)

from .models.opendrift.config_opendrift import OpenDriftConfig
from .models.opendrift.opendrift import OpenDriftModel
# from .the_manager import ParticleTrackingManager
# import particle_tracking_manager.models.opendrift.config_opendrift
