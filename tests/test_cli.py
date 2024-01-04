"""Test CLI methods."""

import particle_tracking_manager as ptm
import os
import pytest


def test_setup():
    """Test CLI setup
    
    No drifters are run due to oceanmodel=None
    """
    ret_value = os.system(f"python {ptm.__path__[0]}/cli.py oceanmodel=None lon=-151 lat=59 use_auto_landmask=True start_time='2000-1-1'")
    assert ret_value == 0