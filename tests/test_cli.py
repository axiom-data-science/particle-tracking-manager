"""Test CLI methods."""

import os

from datetime import datetime

import pytest

import particle_tracking_manager as ptm


def test_setup():
    """Test CLI setup

    No drifters are run due to oceanmodel=None
    """
    ret_value = os.system(
        f"ptm ocean_model=test lon=-151 lat=59 use_auto_landmask=True start_time='2000-1-1' --dry-run"
    )
    assert ret_value == 0


def test_setup_library():
    """Same test but with library"""

    m = ptm.OpenDriftModel(
        ocean_model="test",
        lon=-151,
        lat=59,
        use_auto_landmask=True,
        # steps=3,
        start_time=datetime(2000, 1, 1),
    )
    with pytest.raises(AssertionError):
        m.run_all()
