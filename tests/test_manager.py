"""Test manager use in library, the default approach."""

from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pytest

import particle_tracking_manager as ptm


def test_order():
    """Have to configure before seeding."""

    with pytest.raises(KeyError):
        manager = ptm.OpenDriftModel()
        manager.run()


def test_seed():
    """make sure seeding works with no ocean model

    also compare two approaches for inputting info.
    """

    manager = ptm.OpenDriftModel(use_auto_landmask=True, number=1)
    manager.lon = -151
    manager.lat = 59
    manager.start_time = datetime(2000, 1, 1)

    # with pytest.raises(ValueError):
    #     manager.seed()

    manager.ocean_model = "test"
    manager.has_added_reader = True  # cheat to run test
    manager.seed()
    # look at elements with manager.o.elements_scheduled

    seeding_kwargs = dict(lon=-151, lat=59, start_time=datetime(2000, 1, 1))
    manager2 = ptm.OpenDriftModel(
        use_auto_landmask=True, number=1, ocean_model="test", **seeding_kwargs
    )
    manager2.has_added_reader = True  # cheat to run test
    manager2.seed()

    assert (
        manager.o.elements_scheduled.__dict__ == manager2.o.elements_scheduled.__dict__
    )


@mock.patch(
    "particle_tracking_manager.models.opendrift.model_opendrift.OpenDriftModel.reader_metadata"
)
def test_lon_check(mock_reader_metadata):
    """Test longitude check that is run when variable and reader are set."""

    # Check that longitude is checked as being within (mocked) reader values
    mock_reader_metadata.return_value = np.array([-150, -140, -130])

    m = ptm.OpenDriftModel(lon=0, lat=0)

    # this causes the check
    with pytest.raises(AssertionError):
        m.has_added_reader = True


@mock.patch(
    "particle_tracking_manager.models.opendrift.model_opendrift.OpenDriftModel.reader_metadata"
)
def test_start_time_check(mock_reader_metadata):
    """Test start_time check that is run when variable and reader are set."""

    # Check that start_time is checked as being within (mocked) reader values
    mock_reader_metadata.return_value = datetime(2000, 1, 1)

    m = ptm.OpenDriftModel(start_time=datetime(1999, 1, 1))

    # this causes the check
    with pytest.raises(AssertionError):
        m.has_added_reader = True


@mock.patch(
    "particle_tracking_manager.models.opendrift.model_opendrift.OpenDriftModel.reader_metadata"
)
def test_ocean_model_not_None(mock_reader_metadata):
    """Test that ocean_model can't be None."""

    # Use this to get through steps necessary for the test
    mock_reader_metadata.return_value = datetime(2000, 1, 1)

    m = ptm.OpenDriftModel()
    with pytest.raises(AssertionError):
        m.has_added_reader = True


def test_parameter_passing():
    """make sure parameters passed into package make it to simulation runtime."""

    ts = 2 * 3600
    diffmodel = "windspeed_Sundby1983"
    use_auto_landmask = True
    vertical_mixing = True
    do3D = True

    seed_kws = dict(
        lon=4.0,
        lat=60.0,
        radius=5000,
        number=100,
        start_time=datetime(2015, 9, 22, 6, 0, 0),
    )
    m = ptm.OpenDriftModel(
        use_auto_landmask=use_auto_landmask,
        time_step=ts,
        duration=timedelta(hours=10),
        diffusivitymodel=diffmodel,
        vertical_mixing=vertical_mixing,
        do3D=do3D,
        **seed_kws
    )

    # idealized simulation, provide a fake current
    m.o.set_config("environment:fallback:y_sea_water_velocity", 1)

    # seed
    m.seed()

    # run simulation
    m.run()

    # check time_step across access points
    assert (
        m.o.time_step.seconds
        == ts
        == m.time_step
        == m.show_config_model(key="time_step")["value"]
    )

    # check diff model
    assert m.show_config(key="diffusivitymodel")["value"] == diffmodel

    # check use_auto_landmask coming through
    assert m.show_config(key="use_auto_landmask")["value"] == use_auto_landmask


def test_keyword_parameters():
    """Make sure unknown parameters are not input."""

    with pytest.raises(KeyError):
        m = ptm.OpenDriftModel(incorrect_key="test")
