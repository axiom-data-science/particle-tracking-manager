"""Particle Tracking Manager."""

from .model_opendrift import OpenDrift

import cmocean
cmap = cmocean.tools.crop_by_percent(cmocean.cm.amp, 20, which='max', N=None)
