"""Particle Tracking Manager."""

import cmocean

from .models.opendrift.model_opendrift import OpenDriftModel
from .the_manager import ParticleTrackingManager


cmap = cmocean.tools.crop_by_percent(cmocean.cm.amp, 20, which="max", N=None)
