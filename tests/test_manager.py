"""Test manager use in library, the default approach."""

import unittest

from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import particle_tracking_manager as ptm


def test_z_sign():
    """z should be negative"""

    with pytest.raises(ValueError):
        m = ptm.OpenDriftModel(z=1)


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
    "particle_tracking_manager.models.opendrift.opendrift.OpenDriftModel.reader_metadata"
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
    "particle_tracking_manager.models.opendrift.opendrift.OpenDriftModel.reader_metadata"
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
    "particle_tracking_manager.models.opendrift.opendrift.OpenDriftModel.reader_metadata"
)
def test_ocean_model_not_None(mock_reader_metadata):
    """Test that ocean_model can't be None."""

    # Use this to get through steps necessary for the test
    mock_reader_metadata.return_value = datetime(2000, 1, 1)

    m = ptm.OpenDriftModel()
    with pytest.raises(AssertionError):
        m.has_added_reader = True


@pytest.mark.slow
def test_parameter_passing():
    """make sure parameters passed into package make it to simulation runtime."""

    ts = 5
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
        steps=None,
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


def test_ocean_model_timing():
    """Check that error raised when timing for model wrong...

    and not other times."""

    with pytest.raises(AssertionError):
        m = ptm.OpenDriftModel(ocean_model="NWGOA", start_time="1998-1-1")

    with pytest.raises(AssertionError):
        m = ptm.OpenDriftModel(ocean_model="NWGOA", start_time="2009-2-1")

    m = ptm.OpenDriftModel(ocean_model="NWGOA", start_time="2007-2-1")

    with pytest.raises(AssertionError):
        m = ptm.OpenDriftModel(ocean_model="CIOFS", start_time="1998-1-1")

    with pytest.raises(AssertionError):
        m = ptm.OpenDriftModel(ocean_model="CIOFS", start_time="2023-2-1")

    m = ptm.OpenDriftModel(ocean_model="CIOFS", start_time="2020-2-1")

    with pytest.raises(AssertionError):
        m = ptm.OpenDriftModel(ocean_model="CIOFSOP", start_time="2020-1-1")

    m = ptm.OpenDriftModel(ocean_model="CIOFSOP", start_time="2023-1-1")


def test_lon_lat_checks():
    """Check that lon/lat check errors are raised."""

    with pytest.raises(AssertionError):
        m = ptm.OpenDriftModel(lon=-180.1)

    m = ptm.OpenDriftModel(lon=0)

    with pytest.raises(AssertionError):
        m = ptm.OpenDriftModel(lat=-95)

    m = ptm.OpenDriftModel(lat=0)


def test_setattr_oceanmodel_lon0_360():
    """Test setting oceanmodel_lon0_360 attribute."""
    manager = ptm.OpenDriftModel()
    manager.lon = -150
    manager.oceanmodel_lon0_360 = True
    assert manager.lon == 210


def test_setattr_surface_only():
    """Test setting surface_only attribute."""
    manager = ptm.OpenDriftModel(do3D=True, z=-1, vertical_mixing=True)
    manager.surface_only = True
    assert manager.do3D == False
    assert manager.z == 0
    assert manager.vertical_mixing == False


def test_input_too_many_end_of_simulation():
    with pytest.raises(AssertionError):
        ptm.OpenDriftModel(
            steps=4,
            duration=pd.Timedelta("24h"),
            end_time=pd.Timestamp("1970-01-01T02:00"),
        )


def test_changing_end_of_simulation():
    """change end_time, steps, and duration

    and make sure others are updated accordingly.
    This accounts for the default time_step of 300 seconds.

    """

    m = ptm.OpenDriftModel(start_time=pd.Timestamp("2000-1-1"))
    m.start_time = pd.Timestamp("2000-1-2")
    m.end_time = pd.Timestamp("2000-1-3")
    assert m.steps == 288
    assert m.duration == pd.Timedelta("1 days 00:00:00")

    m.steps = 48
    assert m.end_time == pd.Timestamp("2000-01-02 04:00:00")
    assert m.duration == pd.Timedelta("0 days 04:00:00")

    m.duration = pd.Timedelta("2 days 12:00:00")
    assert m.end_time == pd.Timestamp("2000-01-04 12:00:00")
    assert m.steps == 720


class TestTheManager(unittest.TestCase):
    def setUp(self):
        self.m = ptm.OpenDriftModel()
        self.m.reader_metadata = mock.MagicMock(
            side_effect=lambda x: {
                "lon": np.array([0, 180]),
                "lat": np.array([-90, 90]),
                "start_time": pd.Timestamp("2022-01-01 12:00:00"),
            }[x]
        )

    def test_has_added_reader_true_lon_lat_set(self):
        self.m.lon = 90
        self.m.lat = 45
        self.m.ocean_model = "test"
        self.m.has_added_reader = True
        self.assertEqual(self.m.has_added_reader, True)

    def test_has_added_reader_true_start_time_set(self):
        self.m.start_time = "2022-01-01 12:00:00"
        self.m.ocean_model = "test"
        self.m.has_added_reader = True
        self.assertEqual(self.m.has_added_reader, True)


class TestManager(unittest.TestCase):
    def setUp(self):
        self.m = ptm.OpenDriftModel()

    def test_start_time_str(self):
        self.m.start_time = "2022-01-01 12:00:00"
        self.assertEqual(self.m.start_time, pd.Timestamp("2022-01-01 12:00:00"))

    def test_start_time_datetime(self):
        dt = datetime(2022, 1, 1, 12, 0, 0)
        self.m.start_time = dt
        self.assertEqual(self.m.start_time, pd.Timestamp(dt))

    def test_start_time_timestamp(self):
        ts = pd.Timestamp("2022-01-01 12:00:00")
        self.m.start_time = ts
        self.assertEqual(self.m.start_time, ts)

    def test_start_time_invalid(self):
        with self.assertRaises(TypeError):
            self.m.start_time = 123

    def test_surface_only_true(self):
        self.m.surface_only = True
        self.m.do3D = True
        self.assertEqual(self.m.do3D, False)
        self.m.z = -10
        self.assertEqual(self.m.z, 0)
        self.m.vertical_mixing = True
        self.assertEqual(self.m.vertical_mixing, False)

    def test_surface_only_false_do3D_false(self):
        self.m.surface_only = False
        self.m.do3D = False
        self.m.vertical_mixing = True
        self.assertEqual(self.m.vertical_mixing, False)

    def test_surface_only_false_do3D_true(self):
        self.m.surface_only = False
        self.m.do3D = True
        self.m.vertical_mixing = True
        self.assertEqual(self.m.vertical_mixing, True)

    def test_seed_seafloor_true(self):
        self.m.seed_seafloor = True
        self.assertIsNone(self.m.z)

    def test_z_set(self):
        self.m.z = -10
        self.assertEqual(self.m.z, -10)
        self.assertFalse(self.m.seed_seafloor)

    def test_has_added_reader_true_ocean_model_set(self):
        self.m.ocean_model = "test"
        self.m.has_added_reader = True
        self.assertEqual(self.m.has_added_reader, True)

    def test_run_forward_true(self):
        self.m.run_forward = True
        self.assertEqual(self.m.timedir, 1)

    def test_run_forward_false(self):
        self.m.run_forward = False
        self.assertEqual(self.m.timedir, -1)

    def test_seed_flag_elements_lon_lat_none(self):
        self.m.seed_flag = "elements"
        self.m.lon = None
        self.m.lat = None
        with pytest.raises(KeyError):
            self.m.seed()

    def test_seed_flag_geojson_geojson_none(self):
        self.m.seed_flag = "geojson"
        self.m.geojson = None
        with pytest.raises(KeyError):
            self.m.seed()

    def test_seed_seafloor_false_z_none(self):
        self.m.seed_seafloor = False
        self.m.lon = 0
        self.m.lat = 0
        self.m.z = None
        with pytest.raises(AssertionError):
            self.m.seed()

    def test_start_time_none(self):
        self.m.start_time = None
        with pytest.raises(KeyError):
            self.m.seed()


if __name__ == "__main__":
    unittest.main()
