"""Test manager use in library, the default approach."""

import pytest
import particle_tracking_manager as ptm
from datetime import datetime


def test_order():
    """Have to configure before seeding."""

    with pytest.raises(KeyError):
        manager = ptm.OpenDrift()
        manager.seed()


def test_config():
    """make sure config runs with no ocean model"""
    manager = ptm.OpenDrift()
    manager.config()


def test_seed():
    """make sure seeding works with no ocean model
    
    also compare two approaches for inputting info.
    """
    manager = ptm.OpenDrift(use_auto_landmask=True, number=1)
    manager.config()
    manager.lon = -151
    manager.lat = 59
    manager.start_time = datetime(2000,1,1)
    manager.seed()
    # look at elements with manager.o.elements_scheduled
 
    seeding_kwargs = dict(lon = -151, lat = 59, start_time = datetime(2000,1,1))
    manager2 = ptm.OpenDrift(use_auto_landmask=True, number=1, **seeding_kwargs)
    manager2.config()
    manager2.seed()
    
    assert manager.o.elements_scheduled.__dict__ == manager2.o.elements_scheduled.__dict__
