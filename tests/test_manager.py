"""Test manager use in library, the default approach."""

import pytest
import particle_tracking_manager as ptm
from datetime import datetime
from unittest import mock
import numpy as np


def test_order():
    """Have to configure before seeding."""

    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
        manager.seed()

    manager.ocean_model = "test"
    manager.has_added_reader = True  # cheat to run test
    manager.seed()
    # look at elements with manager.o.elements_scheduled
 
    seeding_kwargs = dict(lon = -151, lat = 59, start_time = datetime(2000,1,1))
    manager2 = ptm.OpenDrift(use_auto_landmask=True, number=1, ocean_model="test", **seeding_kwargs)
    manager2.config()
    manager2.has_added_reader = True  # cheat to run test
    manager2.seed()
    
    assert manager.o.elements_scheduled.__dict__ == manager2.o.elements_scheduled.__dict__


@mock.patch("particle_tracking_manager.model_opendrift.OpenDrift.reader_metadata")
def test_lon_check(mock_reader_metadata):
    """Test longitude check that is run when variable and reader are set."""
    
    # Check that longitude is checked as being within (mocked) reader values
    mock_reader_metadata.return_value = np.array([-150, -140, -130])
    
    m = ptm.OpenDrift(lon=0, lat=0)

    # this causes the check
    with pytest.raises(AssertionError):
        m.has_added_reader = True


@mock.patch("particle_tracking_manager.model_opendrift.OpenDrift.reader_metadata")
def test_start_time_check(mock_reader_metadata):
    """Test start_time check that is run when variable and reader are set."""
    
    # Check that start_time is checked as being within (mocked) reader values
    mock_reader_metadata.return_value = datetime(2000,1,1)
    
    m = ptm.OpenDrift(start_time=datetime(1999,1,1))

    # this causes the check
    with pytest.raises(AssertionError):
        m.has_added_reader = True


@mock.patch("particle_tracking_manager.model_opendrift.OpenDrift.reader_metadata")
def test_ocean_model_not_None(mock_reader_metadata):
    """Test that ocean_model can't be None."""
    
    # Use this to get through steps necessary for the test
    mock_reader_metadata.return_value = datetime(2000,1,1)

    m = ptm.OpenDrift()
    m.has_run_config = True
    with pytest.raises(AssertionError):
        m.has_added_reader = True
