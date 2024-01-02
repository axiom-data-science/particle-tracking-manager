"""Test configuration methods."""

import particle_tracking_manager as ptm
import os
import pytest

def test_drift_models():
    os.system("python cli.py driftmodel=Leeway oceanmodel=None")
    os.system("python cli.py driftmodel=OceanDrift oceanmodel=None")
    os.system("python cli.py driftmodel=LarvalFish oceanmodel=None")
    os.system("python cli.py driftmodel=OpenOil oceanmodel=None")

def test_order():

    with pytest.raises(KeyError):
        manager = ptm.OpenDrift()
        manager.seed()