"""Test CLI methods."""

import os

import pytest

import particle_tracking_manager as ptm


@pytest.mark.slow
def test_setup():
    """Test CLI setup with dryrun

    No drifters are run due to dryrun flag
    """
    ret_value = os.system(f"ptm steps=1 --dry-run")
    assert ret_value == 0


def test_setup_library():
    """Same test but with library"""

    m = ptm.OpenDriftModel(
        steps=3,
    )
    m.config.model_dump()
