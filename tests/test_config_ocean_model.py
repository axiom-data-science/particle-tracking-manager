import importlib
import os
import tempfile

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pydantic import ValidationError

import particle_tracking_manager

from particle_tracking_manager.config_ocean_model import ocean_model_simulation_mapper
from particle_tracking_manager.ocean_model_registry import ocean_model_registry


# Valid values
# end_time calculated as 1 5-minute step
tests_valid = {
    "NWGOA": {
        "lon": -147,
        "lat": 59,
        "start_time": "2007-01-01T00:00",
        "end_time": "2007-01-01T05:00",
    },
    "CIOFS": {
        "lon": -153,
        "lat": 59,
        "start_time": "2021-01-01T00:00",
        "end_time": "2021-01-01T05:00",
    },
    "CIOFSOP": {
        "lon": -153,
        "lat": 59,
        "start_time": "2022-01-01T00:00",
        "end_time": "2022-01-01T05:00",
    },
    "CIOFSFRESH": {
        "lon": -153,
        "lat": 59,
        "start_time": "2014-01-01T00:00",
        "end_time": "2014-01-01T05:00",
    },
}

# Invalid values (except start_times are valid since not testing those here)
tests_invalid = {
    "NWGOA": {
        "lon": 185 - 360,
        "lat": 50,
        "start_time": "2022-01-01",
        "end_time": "2022-01-01T05:00",
        "ocean_model_config": ocean_model_registry.get("NWGOA"),
    },
    "CIOFS": {
        "lon": -145,
        "lat": 40,
        "start_time": "2024-01-01",
        "end_time": "2024-01-01T05:00",
        "ocean_model_config": ocean_model_registry.get("CIOFS"),
    },
    "CIOFSOP": {
        "lon": -145,
        "lat": 40,
        "start_time": "2020-01-01",
        "end_time": "2020-01-01T05:00",
        "ocean_model_config": ocean_model_registry.get("CIOFSOP"),
    },
    "CIOFSFRESH": {
        "lon": -145,
        "lat": 40,
        "start_time": "2022-01-01",
        "end_time": "2022-01-01T05:00",
        "ocean_model_config": ocean_model_registry.get("CIOFSFRESH"),
    },
}


def test_lon_lat():
    """Check for valid lon and lat values

    ...for specific ocean models of type OceanModelSimulation.
    Checked for general ranges in test_config_the_manager.py.
    """

    for ocean_model, test in tests_valid.items():
        m = ocean_model_simulation_mapper[ocean_model](
            start_time=test["start_time"],
            end_time=test["end_time"],
            lon=test["lon"],
            lat=test["lat"],
            ocean_model_local=True,
        )

    for ocean_model, test in tests_invalid.items():
        with pytest.raises(ValidationError):
            m = ocean_model_simulation_mapper[ocean_model](
                start_time=tests_valid[ocean_model]["start_time"],
                end_time=tests_valid[ocean_model]["end_time"],
                lon=test["lon"],
                lat=tests_valid[ocean_model]["lat"],
                ocean_model_local=True,
            )
        with pytest.raises(ValidationError):
            m = ocean_model_simulation_mapper[ocean_model](
                start_time=tests_valid[ocean_model]["start_time"],
                end_time=tests_valid[ocean_model]["end_time"],
                lon=tests_valid[ocean_model]["lon"],
                lat=test["lat"],
                ocean_model_local=True,
            )


def test_oceanmodel_lon0_360():
    """Check for correct value of oceanmodel_lon0_360

    based on ocean model and lon input."""

    lon_in = -153

    m = ocean_model_simulation_mapper["CIOFSOP"](
        start_time="2022-01-01",
        lon=lon_in,
        lat=57,
        end_time="2022-01-01T05:00",
        ocean_model_local=True,
    )
    assert m.ocean_model_config.oceanmodel_lon0_360 == False
    assert m.lon == lon_in

    m = ocean_model_simulation_mapper["CIOFS"](
        start_time="2022-01-01",
        lon=lon_in,
        lat=57,
        end_time="2022-01-01T05:00",
        ocean_model_local=True,
    )
    assert m.ocean_model_config.oceanmodel_lon0_360 == False
    assert m.lon == lon_in

    m = ocean_model_simulation_mapper["CIOFSFRESH"](
        start_time="2004-01-01",
        lon=lon_in,
        lat=57,
        end_time="2004-01-01T05:00",
        ocean_model_local=True,
    )
    assert m.ocean_model_config.oceanmodel_lon0_360 == False
    assert m.lon == lon_in

    m = ocean_model_simulation_mapper["NWGOA"](
        start_time="2007-01-01",
        lon=lon_in,
        lat=57,
        end_time="2007-01-01T05:00",
        ocean_model_local=True,
    )
    assert m.ocean_model_config.oceanmodel_lon0_360 == True
    assert m.lon == lon_in + 360


def test_start_end_times():
    """Check for valid start_time and end_time values

    ...for specific ocean models of type OceanModelSimulation.
    Checked for general ranges in test_config_the_manager.py.
    """

    for ocean_model, test in tests_invalid.items():
        with pytest.raises(ValidationError):
            m = ocean_model_simulation_mapper[ocean_model](
                start_time=test["start_time"],
                end_time=tests_valid[ocean_model]["end_time"],
                lon=tests_valid[ocean_model]["lon"],
                lat=tests_valid[ocean_model]["lat"],
                ocean_model_local=True,
            )
        with pytest.raises(ValidationError):
            m = ocean_model_simulation_mapper[ocean_model](
                start_time=tests_valid[ocean_model]["start_time"],
                end_time=test["end_time"],
                lon=tests_valid[ocean_model]["lon"],
                lat=tests_valid[ocean_model]["lat"],
                ocean_model_local=True,
            )


def test_user_registry():

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the content of the YAML file
        data = {
            "USER": {
                "name": "USER",
                "loc_remote": None,
                "temporal_resolution_str": "PT1H",
                "lon_min": -180,
                "lon_max": 180,
                "lat_min": -90,
                "lat_max": 90,
                "start_time_model": "2009-01-01T00:00",
                "end_time_fixed": "2010-01-01T00:00:00",
                "oceanmodel_lon0_360": False,
                "standard_name_mapping": {},
                "model_drop_vars": [],
                "dx": 1000,
                "kerchunk_func_str": None,
            }
        }

        # Define the path for the YAML file
        yaml_file_path = Path(temp_dir) / f"USER.yaml"

        # Write the YAML data to the file
        with open(yaml_file_path, "w") as file:
            yaml.dump(data, file)

        # Set an environment variable for testing purposes
        # this defines where to find user templates with *.yaml
        os.environ["PTM_CONFIG_DIR"] = str(temp_dir)

        # Check if it's defined
        env_var = os.getenv("PTM_CONFIG_DIR")

        # reload particle_tracking_manager since we changed the environment variable
        importlib.reload(particle_tracking_manager.ocean_model_registry)

        # now USER is in the registry
        assert (
            "USER"
            in particle_tracking_manager.ocean_model_registry.ocean_model_registry.all()
        )


def test_onthefly_registry():

    ds_info = dict(
        temporal_resolution_str="PT1H",
        lon_min=1,
        lon_max=3,
        lat_min=1,
        lat_max=2,
        start_time_model=0,
        end_time_fixed=1,
    )
    particle_tracking_manager.config_ocean_model.register_on_the_fly(ds_info)

    # now new config is in the registry
    assert ocean_model_registry.get("ONTHEFLY").lon_min == 1


@patch("particle_tracking_manager.ocean_model_registry.calculate_CIOFSOP_max")
def test_CIOFSOP_max_update(model_CIOFSOP_max):
    """Make sure that CIOFSOP end_time_model is updated

    ...when the model is updated. Mock input from calculate_CIOFSOP_max.
    """

    # check the value once
    return_value1 = datetime(2025, 1, 16, 23, 0)
    model_CIOFSOP_max.return_value = return_value1
    m = particle_tracking_manager.OpenDriftModel(drift_model="Leeway", steps=1)
    assert m.config.ocean_model_config.end_time_model == return_value1

    # check the value again to see if it is updated, like it would be in real life
    return_value2 = datetime(2025, 2, 16, 23, 0)
    model_CIOFSOP_max.return_value = return_value2
    m = particle_tracking_manager.OpenDriftModel(drift_model="Leeway", steps=1)
    assert m.config.ocean_model_config.end_time_model == return_value2
