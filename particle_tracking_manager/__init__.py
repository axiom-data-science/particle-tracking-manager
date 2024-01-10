"""Particle Tracking Manager."""

from .model_opendrift import OpenDriftModel
from .the_manager import ParticleTrackingManager

import cmocean
cmap = cmocean.tools.crop_by_percent(cmocean.cm.amp, 20, which='max', N=None)
