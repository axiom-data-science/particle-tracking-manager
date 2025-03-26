"""Test manager use in library, the default approach."""


from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# import particle_tracking_manager as ptm
from particle_tracking_manager.the_manager import ParticleTrackingManager
from pydantic import ValidationError
from particle_tracking_manager.config_the_manager import TheManagerConfig


class TestConfig(TheManagerConfig):
    pass


# Set up a subclass for testing. This is meant to be a simple version of the
# OpenDriftModel.
class TestParticleTrackingManager(ParticleTrackingManager):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config = TestConfig(**kwargs)
        
    def _add_reader(self):
        pass

    def _seed(self):
        pass
        
    def _run(self):
        pass



def test_order():
    """Have to configure before seeding."""

    m = TestParticleTrackingManager(steps=1, start_time="2022-01-01")
    with pytest.raises(ValueError):
        m.run()


def test_seed():
    m = TestParticleTrackingManager(steps=1, start_time="2022-01-01")
    m.state.has_added_reader = True
    m.seed()
    
    m = TestParticleTrackingManager(steps=1, start_time="2022-01-01")
    with pytest.raises(ValueError):
        m.seed()



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





# # THIS ONE SHOULD BE IN OPENDRIFT SINCE THAT ONE EXCLUDES UNKNOWN INPUTS
# def test_keyword_parameters():
#     """Make sure unknown parameters are not input."""

#     with pytest.raises(ValidationError):
#         m = TestParticleTrackingManager(unknown="test", steps=1, start_time="2022-01-01")
#         # m = ParticleTrackingManager(unknown="test", steps=1, start_time="2022-01-01")


# def test_oceanmodel_lon0_360():
#     """Check for correct value of oceanmodel_lon0_360 
    
#     based on ocean model and lon input."""
    
#     lon_in = -153

#     m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", lon=lon_in)
#     assert m.config.oceanmodel_lon0_360 == False
#     assert m.config.lon == lon_in

#     m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", ocean_model="CIOFS")
#     assert m.config.oceanmodel_lon0_360 == False

#     m = TestParticleTrackingManager(steps=1, start_time="2007-01-01", ocean_model="NWGOA", lon=lon_in)
#     assert m.config.oceanmodel_lon0_360 == True
#     assert m.config.lon == lon_in + 360



def test_lon_lat():
    """Check for valid lon and lat values."""

    with pytest.raises(ValidationError):
        m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", lon=-180.1)

    with pytest.raises(ValidationError):
        m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", lat=95)

    m = TestParticleTrackingManager(steps=1, lon=-152, lat=58, start_time="2022-01-01")

    # with pytest.raises(ValidationError):
    #     m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", ocean_model="NWGOA",
    #                                 lon=185-360)

    # with pytest.raises(ValidationError):
    #     m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", ocean_model="CIOFS",
    #                                 lon=-145)


def test_seed_flag_elements():
    """Check seed flag elements."""
    
    with pytest.raises(ValidationError):
        m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="elements", lon=None, lat=None)
    
    m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="elements")


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
        m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="geojson", geojson=None)

    with pytest.raises(ValidationError):
        m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="geojson", geojson=geojson,
                                    lon=50, lat=50)
        
    m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", seed_flag="geojson", geojson=geojson, lon=None, lat=None)


# def test_start_time_NWGOA():
#     with pytest.raises(ValueError):
#         m = TestParticleTrackingManager(steps=1, start_time="2009-01-02 00:00:00", ocean_model="NWGOA")
#     with pytest.raises(ValueError):
#         m = TestParticleTrackingManager(steps=1, start_time="1998-01-01 12:00:00", ocean_model="NWGOA")
#     m = TestParticleTrackingManager(steps=1, start_time="2000-01-01 12:00:00", ocean_model="NWGOA")


# def test_start_time_CIOFS():
#     with pytest.raises(ValueError):
#         m = TestParticleTrackingManager(steps=1, start_time="1998-01-01 12:00:00", ocean_model="CIOFS")
#     with pytest.raises(ValueError):
#         m = TestParticleTrackingManager(steps=1, start_time="2023-01-01 12:00:00", ocean_model="CIOFS")
#     m = TestParticleTrackingManager(steps=1, start_time="2020-01-01 12:00:00", ocean_model="CIOFS")


# def test_start_time_CIOFSOP():
#     with pytest.raises(ValueError):
#         m = TestParticleTrackingManager(steps=1, start_time="2020-01-01 12:00:00", ocean_model="CIOFSOP")
#     with pytest.raises(ValueError):
#         future_date = pd.Timestamp.now() + pd.Timedelta(days=10)
#         m = TestParticleTrackingManager(steps=1, start_time=future_date, ocean_model="CIOFSOP")
#     m = TestParticleTrackingManager(steps=1, start_time="2023-01-01 12:00:00", ocean_model="CIOFSOP")
    
#     assert m.config.start_time == pd.Timestamp("2023-01-01 12:00:00")

# def test_do3D():
#     m = TestParticleTrackingManager(steps=1, do3D=True, start_time="2022-01-01", vertical_mixing=True)
#     m = TestParticleTrackingManager(steps=1, do3D=True, start_time="2022-01-01", vertical_mixing=False)
#     with pytest.raises(ValueError):
#         m = TestParticleTrackingManager(steps=1, do3D=False, start_time="2022-01-01", vertical_mixing=True)


# def test_z():
#     # m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", z=-10)
#     # assert m.config.seed_seafloor == False
    
#     with pytest.raises(ValueError):
#         m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", z=10)
    
#     m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", z=None, seed_seafloor=True)
    
#     with pytest.raises(ValidationError):
#         m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", z=None, seed_seafloor=False)
    
#     with pytest.raises(ValidationError):
#         m = TestParticleTrackingManager(steps=1, start_time="2022-01-01", z=-10, seed_seafloor=True)


# THIS IS SET IN OPENDRIFT NOW SO CHANGE TO TEST IN THAT FILE
# def test_interpolator_filename():
#     with pytest.raises(ValueError):
#         m = TestParticleTrackingManager(interpolator_filename="test", steps=1, use_cache=False)

#     m = TestParticleTrackingManager(interpolator_filename="test", steps=1)
#     assert m.config.interpolator_filename == "test.pickle"

#     m = TestParticleTrackingManager(interpolator_filename=None, use_cache=False, steps=1)


def test_log_name():
    m = TestParticleTrackingManager(output_file="newtest", steps=1)
    assert m.files.logfile_name == "newtest.log"

    m = TestParticleTrackingManager(output_file="newtest.nc", steps=1)
    assert m.files.logfile_name == "newtest.log"

    m = TestParticleTrackingManager(output_file="newtest.parq", steps=1)
    assert m.files.logfile_name == "newtest.log"

    m = TestParticleTrackingManager(output_file="newtest.parquet", steps=1)
    assert m.files.logfile_name == "newtest.log"


def test_misc_parameters():
    """Test values of parameters being input."""

    m = TestParticleTrackingManager(steps=1, start_time="2022-01-01",
                                # horizontal_diffusivity=1,
                                # number=100, 
                                time_step=5,
                                # wind_drift_factor=0.04,
                                stokes_drift=False, log="DEBUG",)
    
    # assert m.config.horizontal_diffusivity == 1
    # assert m.config.number == 100
    assert m.config.time_step == 5
    # assert m.config.wind_drift_factor == 0.04


# def test_horizontal_diffusivity_logic():
#     """Check logic for using default horizontal diff values for known models."""

#     m = TestParticleTrackingManager(ocean_model="NWGOA", steps=1, start_time="2007-01-01")
#     assert m.config.horizontal_diffusivity == 150.0  # known grid values

#     m = TestParticleTrackingManager(ocean_model="CIOFS", steps=1, start_time="2020-01-01")
#     assert m.config.horizontal_diffusivity == 10.0  # known grid values

#     m = TestParticleTrackingManager(ocean_model="CIOFSOP", horizontal_diffusivity=11, steps=1)
#     assert m.config.horizontal_diffusivity == 11.0  # user-selected value

#     m = TestParticleTrackingManager(ocean_model="CIOFSOP", steps=1)
#     assert m.config.horizontal_diffusivity == 10.0  # known grid values




def test_output_file():
    """make sure output file is parquet if output_format is parquet"""

    m = TestParticleTrackingManager(output_format="parquet", steps=1)
    assert m.files.output_file.endswith(".parquet")

    m = TestParticleTrackingManager(output_format="netcdf", steps=1)
    assert m.files.output_file.endswith(".nc")


def test_ocean_model_not_known():
    with pytest.raises(ValidationError):
        TestParticleTrackingManager(ocean_model="wrong_name", steps=1)
