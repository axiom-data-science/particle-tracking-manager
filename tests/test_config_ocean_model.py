
from particle_tracking_manager.config_ocean_model import OceanModelConfig, NWGOA, CIOFS, CIOFSOP, CIOFSFRESH, \
    ocean_model_simulation_mapper
import pytest
from pydantic import ValidationError

# Valid values
# end_time calculated as 1 5-minute step
tests_valid = {"NWGOA": {"lon": -147, "lat": 59, "start_time": "2007-01-01T00:00", "end_time": "2007-01-01T05:00"},
            "CIOFS": {"lon": -153, "lat": 59, "start_time": "2021-01-01T00:00", "end_time": "2021-01-01T05:00"},
            "CIOFSOP": {"lon": -153, "lat": 59, "start_time": "2022-01-01T00:00", "end_time": "2022-01-01T05:00"},
            "CIOFSFRESH": {"lon": -153, "lat": 59, "start_time": "2014-01-01T00:00", "end_time": "2014-01-01T05:00"}}

# Invalid values (except start_times are valid since not testing those here)
tests_invalid = {"NWGOA": {"lon": 185-360, "lat": 50, "start_time": "2022-01-01", "end_time": "2022-01-01T05:00", "ocean_model_config": NWGOA},
            "CIOFS": {"lon": -145, "lat": 40, "start_time": "2024-01-01", "end_time": "2024-01-01T05:00", "ocean_model_config": CIOFS},
            "CIOFSOP": {"lon": -145, "lat": 40, "start_time": "2020-01-01", "end_time": "2020-01-01T05:00", "ocean_model_config": CIOFSOP},
            "CIOFSFRESH": {"lon": -145, "lat": 40, "start_time": "2022-01-01", "end_time": "2022-01-01T05:00", "ocean_model_config": CIOFSFRESH}}


def test_lon_lat():
    """Check for valid lon and lat values
    
    ...for specific ocean models of type OceanModelSimulation.
    Checked for general ranges in test_config_the_manager.py.
    """

    for ocean_model, test in tests_valid.items():
        m = ocean_model_simulation_mapper[ocean_model](steps=1, start_time=test["start_time"], end_time=test["end_time"],
                                    lon=test["lon"], lat=test["lat"], ocean_model_local=True)

    for ocean_model, test in tests_invalid.items():
        with pytest.raises(ValidationError):
            m = ocean_model_simulation_mapper[ocean_model](steps=1, start_time=tests_valid[ocean_model]["start_time"], end_time=tests_valid[ocean_model]["end_time"],
                                        lon=test["lon"], lat=tests_valid[ocean_model]["lat"], ocean_model_local=True)
        with pytest.raises(ValidationError):
            m = ocean_model_simulation_mapper[ocean_model](steps=1, start_time=tests_valid[ocean_model]["start_time"], end_time=tests_valid[ocean_model]["end_time"],
                                        lon=tests_valid[ocean_model]["lon"], lat=test["lat"], ocean_model_local=True)


def test_oceanmodel_lon0_360():
    """Check for correct value of oceanmodel_lon0_360 
    
    based on ocean model and lon input."""
    
    lon_in = -153
    
    m = ocean_model_simulation_mapper["CIOFSOP"](steps=1, start_time="2022-01-01", lon=lon_in, lat=57, end_time="2022-01-01T05:00", 
                                    ocean_model_local=True)
    assert m.ocean_model_config.oceanmodel_lon0_360 == False
    assert m.lon == lon_in

    m = ocean_model_simulation_mapper["CIOFS"](steps=1, start_time="2022-01-01", lon=lon_in, lat=57, end_time="2022-01-01T05:00", 
                                    ocean_model_local=True)
    assert m.ocean_model_config.oceanmodel_lon0_360 == False
    assert m.lon == lon_in

    m = ocean_model_simulation_mapper["CIOFSFRESH"](steps=1, start_time="2004-01-01", lon=lon_in, lat=57, end_time="2004-01-01T05:00", 
                                    ocean_model_local=True)
    assert m.ocean_model_config.oceanmodel_lon0_360 == False
    assert m.lon == lon_in
    
    m = ocean_model_simulation_mapper["NWGOA"](steps=1, start_time="2007-01-01", lon=lon_in, lat=57, end_time="2007-01-01T05:00",
                                    ocean_model_local=True)
    assert m.ocean_model_config.oceanmodel_lon0_360 == True
    assert m.lon == lon_in + 360


def test_start_end_times():
    """Check for valid start_time and end_time values
    
    ...for specific ocean models of type OceanModelSimulation.
    Checked for general ranges in test_config_the_manager.py.
    """

    for ocean_model, test in tests_invalid.items():
        with pytest.raises(ValidationError):
            m = ocean_model_simulation_mapper[ocean_model](steps=1, start_time=test["start_time"], end_time=tests_valid[ocean_model]["end_time"],
                                        lon=tests_valid[ocean_model]["lon"], lat=tests_valid[ocean_model]["lat"], ocean_model_local=True)
        with pytest.raises(ValidationError):
            m = ocean_model_simulation_mapper[ocean_model](steps=1, start_time=tests_valid[ocean_model]["start_time"], end_time=test["end_time"],
                                        lon=tests_valid[ocean_model]["lon"], lat=tests_valid[ocean_model]["lat"], ocean_model_local=True)
