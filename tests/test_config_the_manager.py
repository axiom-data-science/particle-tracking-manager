from datetime import datetime

import pandas as pd
import pytest

from pydantic import ValidationError

from particle_tracking_manager.config_the_manager import TheManagerConfig


def test_start_time_type():
    """Check start time type."""
    m = TheManagerConfig(steps=1, start_time="2022-01-01 12:00:00")
    assert m.start_time == pd.Timestamp("2022-01-01 12:00:00")

    m = TheManagerConfig(steps=1, start_time=pd.Timestamp("2022-01-01 12:00:00"))
    assert m.start_time == pd.Timestamp("2022-01-01 12:00:00")

    m = TheManagerConfig(steps=1, start_time=datetime(2022, 1, 1, 12, 0, 0))
    assert m.start_time == pd.Timestamp("2022-01-01 12:00:00")


def test_time_calculations():
    with pytest.raises(ValidationError):
        m = TheManagerConfig(steps=1, start_time=None)

    with pytest.raises(ValidationError):
        m = TheManagerConfig(steps=1, duration="1d", start_time=None)

    with pytest.raises(ValidationError):
        # all times defined but not consistently
        m = TheManagerConfig(
            steps=1,
            duration="1d",
            start_time="2022-01-01 12:00:00",
            end_time="2022-01-03 12:00:00",
        )

    # all times defined but consistently
    # duration is 1 time step
    ts, duration = 5 * 60, "P0DT0H5M0S"
    start_time = "2022-01-01 12:00:00"
    m = TheManagerConfig(
        time_step=ts,
        steps=1,
        duration=duration,
        start_time=start_time,
        end_time=pd.Timestamp(start_time) + pd.Timedelta(seconds=ts),
    )

    m = TheManagerConfig(steps=1, end_time="2022-01-01 12:00:00", start_time=None)
    assert m.duration == duration
    assert m.start_time == m.end_time - pd.Timedelta(m.duration)

    with pytest.raises(ValidationError):
        m = TheManagerConfig(
            steps=1,
            end_time="2000-01-02",
            start_time=pd.Timestamp("2000-1-1"),
            ocean_model="CIOFS",
        )

    m = TheManagerConfig(
        end_time="2000-01-02", start_time=pd.Timestamp("2000-1-1"), ocean_model="CIOFS"
    )
    assert m.steps == 288
    assert m.duration == pd.Timedelta("1 days 00:00:00").isoformat()

    m = TheManagerConfig(
        end_time="2023-01-02", start_time=pd.Timestamp("2023-1-1"), run_forward=True
    )
    assert m.timedir == 1

    m = TheManagerConfig(
        end_time="2023-01-02", start_time=pd.Timestamp("2023-1-1"), run_forward=False
    )
    assert m.timedir == -1


def test_lon_lat():
    """Check for valid lon and lat values

    In the general sense from TheManagerConfig.
    Checked for individual ocean_models in test_config_ocean_model.py.
    """

    with pytest.raises(ValidationError):
        m = TheManagerConfig(steps=1, start_time="2022-01-01", lon=-180.1)

    with pytest.raises(ValidationError):
        m = TheManagerConfig(steps=1, start_time="2022-01-01", lat=95)

    m = TheManagerConfig(steps=1, lon=-152, lat=58, start_time="2022-01-01")
    assert m.lon == -152
    assert m.lat == 58


def test_unknown_parameter():
    """Make sure unknown parameters are not input."""

    with pytest.raises(ValidationError):
        m = TheManagerConfig(unknown="test", steps=1, start_time="2022-01-01")


def test_horizontal_diffusivity_logic():
    """Check logic for using default horizontal diff values for known models."""

    m = TheManagerConfig(ocean_model="NWGOA", steps=1, start_time="2007-01-01")
    assert m.horizontal_diffusivity == 150.0  # known grid values

    m = TheManagerConfig(ocean_model="CIOFS", steps=1, start_time="2020-01-01")
    assert m.horizontal_diffusivity == 10.0  # known grid values

    m = TheManagerConfig(ocean_model="CIOFSOP", horizontal_diffusivity=11, steps=1)
    assert m.horizontal_diffusivity == 11.0  # user-selected value

    m = TheManagerConfig(ocean_model="CIOFSOP", steps=1)
    assert m.horizontal_diffusivity == 10.0  # known grid values


def test_z():
    m = TheManagerConfig(steps=1, start_time="2022-01-01", z=-10)

    with pytest.raises(ValueError):
        m = TheManagerConfig(steps=1, start_time="2022-01-01", z=10)


def test_seed_location_inputs():
    """Check lonlat vs geojson."""
    geojson = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Point", "coordinates": [0, 0]},
    }

    m = TheManagerConfig(steps=1, start_time="2022-01-01", geojson=None)
    # these are set in a validator if geojson is None and lon/lat are None
    assert m.lon == -151.0
    assert m.lat == 58.0

    with pytest.raises(ValidationError):
        m = TheManagerConfig(
            steps=1,
            start_time="2022-01-01",
            geojson=geojson,
            lon=50,
            lat=50,
        )

    m = TheManagerConfig(
        steps=1,
        start_time="2022-01-01",
        geojson=geojson,
    )

    m = TheManagerConfig(steps=1, lon=-154, lat=58)


def test_misc_parameters():
    """Test values of parameters being input."""

    m = TheManagerConfig(
        steps=1,
        start_time="2022-01-01",
        horizontal_diffusivity=1,
        number=100,
        time_step=50,
        stokes_drift=False,
    )

    assert m.horizontal_diffusivity == 1
    assert m.number == 100
    assert m.time_step == 50
    assert m.stokes_drift == False


def test_ocean_model_not_known():
    with pytest.raises(ValidationError):
        TheManagerConfig(ocean_model="wrong_name", steps=1)
