"""Test manager use in library, the default approach."""


from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# import particle_tracking_manager as ptm
from particle_tracking_manager.the_manager import ParticleTrackingManager
from pydantic import ValidationError


# the following few tests might not work for the manager because need to know
# about the model too
# def test_order():
#     """Have to configure before seeding."""

#     m = ParticleTrackingManager(steps=1, start_time="2022-01-01")
#     with pytest.raises(ValueError):
#         m.run()


# def test_seed():
#     m = ParticleTrackingManager(steps=1, start_time="2022-01-01")
#     m.has_added_reader = True
#     m.seed()
    
#     m = ParticleTrackingManager(steps=1, start_time="2022-01-01")
#     with pytest.raises(ValueError):
#         m.seed()

# @mock.patch(
#     "particle_tracking_manager.models.opendrift.opendrift.OpenDriftModel.reader_metadata"
# )
# def test_start_time_check(mock_reader_metadata):
#     """Test start_time check that is run when variable and reader are set."""

#     # Check that start_time is checked as being within (mocked) reader values
#     mock_reader_metadata.return_value = datetime(2000, 1, 1)

#     m = ptm.OpenDriftModel(start_time=datetime(1999, 1, 1))

#     # this causes the check
#     with pytest.raises(ValueError):
#         m.has_added_reader = True

# @pytest.mark.slow
# def test_parameter_passing():
#     """make sure parameters passed into package make it to simulation runtime."""

#     ts = 5
#     diffmodel = "windspeed_Sundby1983"
#     use_auto_landmask = True
#     vertical_mixing = True
#     do3D = True

#     seed_kws = dict(
#         lon=4.0,
#         lat=60.0,
#         radius=5000,
#         number=100,
#         start_time=datetime(2015, 9, 22, 6, 0, 0),
#     )
#     m = ptm.OpenDriftModel(
#         use_auto_landmask=use_auto_landmask,
#         time_step=ts,
#         duration=timedelta(hours=10),
#         steps=None,
#         diffusivitymodel=diffmodel,
#         vertical_mixing=vertical_mixing,
#         do3D=do3D,
#         **seed_kws
#     )

#     # idealized simulation, provide a fake current
#     m.o.set_config("environment:fallback:y_sea_water_velocity", 1)

#     # seed
#     m.seed()

#     # run simulation
#     m.run()

#     # check time_step across access points
#     assert (
#         m.o.time_step.seconds
#         == ts
#         == m.time_step
#         == m.show_config_model(key="time_step")["value"]
#     )

#     # check diff model
#     assert m.show_config(key="diffusivitymodel")["value"] == diffmodel

#     # check use_auto_landmask coming through
#     assert m.show_config(key="use_auto_landmask")["value"] == use_auto_landmask



# class TestTheManager(unittest.TestCase):
#     def setUp(self):
#         self.m = ptm.OpenDriftModel()
#         self.m.reader_metadata = mock.MagicMock(
#             side_effect=lambda x: {
#                 "lon": np.array([0, 180]),
#                 "lat": np.array([-90, 90]),
#                 "start_time": pd.Timestamp("2022-01-01 12:00:00"),
#             }[x]
#         )

#     def test_has_added_reader_true_lon_lat_set(self):
#         self.m.lon = 90
#         self.m.lat = 45
#         self.m.ocean_model = "test"
#         self.m.has_added_reader = True
#         self.assertEqual(self.m.has_added_reader, True)

#     def test_has_added_reader_true_start_time_set(self):
#         self.m.start_time = "2022-01-01 12:00:00"
#         self.m.ocean_model = "test"
#         self.m.has_added_reader = True
#         self.assertEqual(self.m.has_added_reader, True)

    # def test_has_added_reader_true_ocean_model_set(self):
    #     self.m.ocean_model = "test"
    #     self.m.has_added_reader = True
    #     self.assertEqual(self.m.has_added_reader, True)






def test_keyword_parameters():
    """Make sure unknown parameters are not input."""

    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(unknown="test", steps=1, start_time="2022-01-01")


def test_oceanmodel_lon0_360():
    """Check for correct value of oceanmodel_lon0_360 
    
    based on ocean model and lon input."""
    
    lon_in = -153

    m = ParticleTrackingManager(steps=1, start_time="2022-01-01", lon=lon_in)
    assert m.config.oceanmodel_lon0_360 == False
    assert m.config.lon == lon_in

    m = ParticleTrackingManager(steps=1, start_time="2022-01-01", ocean_model="CIOFS")
    assert m.config.oceanmodel_lon0_360 == False

    m = ParticleTrackingManager(steps=1, start_time="2007-01-01", ocean_model="NWGOA", lon=lon_in)
    assert m.config.oceanmodel_lon0_360 == True
    assert m.config.lon == lon_in + 360



def test_lon_lat():
    """Check for valid lon and lat values."""

    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", lon=-180.1)

    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", lat=95)

    m = ParticleTrackingManager(steps=1, lon=-152, lat=58, start_time="2022-01-01")

    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", ocean_model="NWGOA",
                                    lon=185-360)

    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", ocean_model="CIOFS",
                                    lon=-145)


def test_seed_flag_elements():
    """Check seed flag elements."""
    
    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="elements", lon=None, lat=None)
    
    m = ParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="elements")


def test_seed_flag_geojson():
    """Check seed flag geojson."""
    geojson = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Point",
            "coordinates": [0, 0]
        }
    }
    
    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="geojson", geojson=None)

    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="geojson", geojson=geojson,
                                    lon=50, lat=50)
        
    m = ParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="geojson", geojson=geojson, lon=None, lat=None)


def test_start_time_type():
    """Check start time type."""
    m = ParticleTrackingManager(steps=1, start_time="2022-01-01 12:00:00")
    assert m.config.start_time == pd.Timestamp("2022-01-01 12:00:00")
    
    m = ParticleTrackingManager(steps=1, start_time=pd.Timestamp("2022-01-01 12:00:00"))
    assert m.config.start_time == pd.Timestamp("2022-01-01 12:00:00")
    
    m = ParticleTrackingManager(steps=1, start_time=datetime(2022, 1, 1, 12, 0, 0))
    assert m.config.start_time == pd.Timestamp("2022-01-01 12:00:00")

    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time=123)


def test_time_calculations():
    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time=None)
    
    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, duration=timedelta(days=1), start_time=None)

    m = ParticleTrackingManager(steps=1, end_time="2022-01-01 12:00:00", start_time=None)
    assert m.config.duration == timedelta(seconds=m.config.time_step*m.config.steps)
    assert m.config.start_time == m.config.end_time - m.config.duration

    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, end_time="2000-01-02", start_time=pd.Timestamp("2000-1-1"), ocean_model="CIOFS")

    m = ParticleTrackingManager(end_time="2000-01-02", start_time=pd.Timestamp("2000-1-1"), ocean_model="CIOFS")
    assert m.config.steps == 288
    assert m.config.duration == pd.Timedelta("1 days 00:00:00")

    m = ParticleTrackingManager(end_time="2023-01-02", start_time=pd.Timestamp("2023-1-1"), run_forward=True)
    assert m.config.timedir == 1

    m = ParticleTrackingManager(end_time="2023-01-02", start_time=pd.Timestamp("2023-1-1"), run_forward=False)
    assert m.config.timedir == -1


def test_start_time_NWGOA():
    with pytest.raises(ValueError):
        m = ParticleTrackingManager(steps=1, start_time="2009-01-02 00:00:00", ocean_model="NWGOA")
    with pytest.raises(ValueError):
        m = ParticleTrackingManager(steps=1, start_time="1998-01-01 12:00:00", ocean_model="NWGOA")
    m = ParticleTrackingManager(steps=1, start_time="2000-01-01 12:00:00", ocean_model="NWGOA")


def test_start_time_CIOFS():
    with pytest.raises(ValueError):
        m = ParticleTrackingManager(steps=1, start_time="1998-01-01 12:00:00", ocean_model="CIOFS")
    with pytest.raises(ValueError):
        m = ParticleTrackingManager(steps=1, start_time="2023-01-01 12:00:00", ocean_model="CIOFS")
    m = ParticleTrackingManager(steps=1, start_time="2020-01-01 12:00:00", ocean_model="CIOFS")


def test_start_time_CIOFSOP():
    with pytest.raises(ValueError):
        m = ParticleTrackingManager(steps=1, start_time="2020-01-01 12:00:00", ocean_model="CIOFSOP")
    with pytest.raises(ValueError):
        future_date = pd.Timestamp.now() + pd.Timedelta(days=10)
        m = ParticleTrackingManager(steps=1, start_time=future_date, ocean_model="CIOFSOP")
    m = ParticleTrackingManager(steps=1, start_time="2023-01-01 12:00:00", ocean_model="CIOFSOP")
    
    assert m.config.start_time == pd.Timestamp("2023-01-01 12:00:00")

def test_do3D():
    m = ParticleTrackingManager(steps=1, do3D=True, start_time="2022-01-01", vertical_mixing=True)
    m = ParticleTrackingManager(steps=1, do3D=True, start_time="2022-01-01", vertical_mixing=False)
    with pytest.raises(ValueError):
        m = ParticleTrackingManager(steps=1, do3D=False, start_time="2022-01-01", vertical_mixing=True)


def test_z():
    m = ParticleTrackingManager(steps=1, start_time="2022-01-01", z=-10)
    assert m.config.seed_seafloor == False
    
    with pytest.raises(ValueError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", z=10)
    
    m = ParticleTrackingManager(steps=1, start_time="2022-01-01", z=None, seed_seafloor=True)
    
    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", z=None, seed_seafloor=False)
    
    with pytest.raises(ValidationError):
        m = ParticleTrackingManager(steps=1, start_time="2022-01-01", z=-10, seed_seafloor=True)


def test_interpolator_filename():
    with pytest.raises(ValueError):
        m = ParticleTrackingManager(interpolator_filename="test", steps=1, use_cache=False)

    m = ParticleTrackingManager(interpolator_filename="test", steps=1)
    assert m.config.interpolator_filename == "test.pickle"

    m = ParticleTrackingManager(interpolator_filename=None, use_cache=False, steps=1)


def test_log_name():
    m = ParticleTrackingManager(output_file="newtest", steps=1)
    assert m.logfile_name == "newtest.log"

    m = ParticleTrackingManager(output_file="newtest.nc", steps=1)
    assert m.logfile_name == "newtest.log"

    m = ParticleTrackingManager(output_file="newtest.parq", steps=1)
    assert m.logfile_name == "newtest.log"

    m = ParticleTrackingManager(output_file="newtest.parquet", steps=1)
    assert m.logfile_name == "newtest.log"


def test_misc_parameters():
    """Test values of parameters being input."""

    m = ParticleTrackingManager(steps=1, start_time="2022-01-01",
                                horizontal_diffusivity=1,
                                radius=100, number=100, time_step=5,
                                use_auto_landmask=True, mixed_layer_depth=10,
                                diffusivitymodel="windspeed_Sundby1983",
                                radius_type="uniform", wind_drift_factor=0.04,
                                stokes_drift=False, coastline_action="previous",
                                seafloor_action="previous", current_uncertainty=0.1,
                                wind_uncertainty=0.1, wind_drift_depth=10,
                                vertical_mixing_timestep=10, log="high")
    
    assert m.config.horizontal_diffusivity == 1
    assert m.config.radius == 100
    assert m.config.number == 100
    assert m.config.time_step == 5
    assert m.config.use_auto_landmask == True
    assert m.config.mixed_layer_depth == 10
    assert m.config.diffusivitymodel == "windspeed_Sundby1983"
    assert m.config.radius_type == "uniform"
    assert m.config.wind_drift_factor == 0.04
    assert m.config.stokes_drift == False
    assert m.config.coastline_action == "previous"
    assert m.config.seafloor_action == "previous"
    assert m.config.current_uncertainty == 0.1
    assert m.config.wind_uncertainty == 0.1
    assert m.config.wind_drift_depth == 10
    assert m.config.vertical_mixing_timestep == 10