"""From Copilot"""

import unittest

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from particle_tracking_manager.models.opendrift.opendrift import (
    OpenDriftModel,
)
from particle_tracking_manager.models.opendrift.config_opendrift import (
    OpenDriftConfig,
)



ds = xr.Dataset(
    data_vars={
        "u": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "v": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "w": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "salt": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "temp": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "wetdry_mask_rho": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "mask_rho": (("Y", "X"), np.zeros((2, 3))),
        "Uwind": (("ocean_time", "Y", "X"), np.zeros((2, 2, 3))),
        "Vwind": (("ocean_time", "Y", "X"), np.zeros((2, 2, 3))),
        "Cs_r": (("Z"), np.linspace(-1, 0, 3)),
        "hc": 16,
    },
    coords={
        "ocean_time": ("ocean_time", [0, 1], {"units": "seconds since 1970-01-01"}),
        "s_rho": (("Z"), np.linspace(-1, 0, 3)),
        "lon_rho": (("Y", "X"), np.array([[1, 2, 3], [1, 2, 3]])),
        "lat_rho": (("Y", "X"), np.array([[1, 1, 1], [2, 2, 2]])),
    },
)


# # WORK ON THIS ONE ONCE HAVE DONE USER INPUT
# class TestOpenDriftModel_OceanDrift_static_mask(unittest.TestCase):
#     def setUp(self):
#         self.model = OpenDriftModel(drift_model="OceanDrift", use_static_masks=True, steps=4)

#     def test_ocean_model_not_known_ds_None(self):
#         self.model.config.ocean_model = "wrong_name"
#         self.model.ds = None  # this is the default
#         # need to input steps, duration, or end_time but don't here
#         with pytest.raises(ValueError):
#             self.model.add_reader(ds=ds)

#     def test_drop_vars_do3D_true(self):
#         self.model.config.do3D = True
#         self.model.config.steps = 4
#         self.model.add_reader(ds=ds)
#         assert self.model.reader.variables == [
#             "x_sea_water_velocity",
#             "y_sea_water_velocity",
#             "upward_sea_water_velocity",
#             "land_binary_mask",
#             "x_wind",
#             "y_wind",
#             "wind_speed",
#             "sea_water_speed",
#         ]

#     def test_drop_vars_use_static_masks(self):
#         self.model.config.do3D = False
#         self.model.config.duration = pd.Timedelta("24h")
#         self.model.add_reader(ds=ds)
#         assert self.model.reader.variables == [
#             "x_sea_water_velocity",
#             "y_sea_water_velocity",
#             "land_binary_mask",
#             "x_wind",
#             "y_wind",
#             "wind_speed",
#             "sea_water_speed",
#         ]
#         assert "mask_rho" in self.model.reader.Dataset.data_vars
#         assert "wetdry_mask_rho" not in self.model.reader.Dataset.data_vars

#     def test_drop_vars_no_wind(self):
#         self.model.config.stokes_drift = False
#         self.model.config.wind_drift_factor = 0
#         self.model.config.wind_uncertainty = 0
#         self.model.config.vertical_mixing = False
#         self.model.config.end_time = pd.Timestamp("1970-01-01T02:00")
#         self.model.add_reader(ds=ds)
#         assert self.model.reader.variables == [
#             "x_sea_water_velocity",
#             "y_sea_water_velocity",
#             "land_binary_mask",
#             "sea_water_speed",
#         ]


# class TestOpenDriftModel_OceanDrift_wetdry_mask(unittest.TestCase):
#     def setUp(self):
#         self.model = OpenDriftModel(drift_model="OceanDrift", use_static_masks=False, steps=1)

#     def test_error_no_end_of_simulation(self):
#         self.model.config.do3D = False
#         # need to input steps, duration, or end_time but don't here
#         with pytest.raises(ValueError):
#             self.model.add_reader(ds=ds)

#     def test_drop_vars_do3D_false(self):
#         self.model.config.do3D = False
#         self.model.config.steps = 4
#         self.model.add_reader(ds=ds)
#         assert self.model.reader.variables == [
#             "x_sea_water_velocity",
#             "y_sea_water_velocity",
#             "land_binary_mask",
#             "x_wind",
#             "y_wind",
#             "wind_speed",
#             "sea_water_speed",
#         ]
#         assert "wetdry_mask_rho" in self.model.reader.Dataset.data_vars
#         assert "mask_rho" not in self.model.reader.Dataset.data_vars





def test_drift_model():
    with pytest.raises(ValueError):
        m = OpenDriftModel(drift_model="not_a_real_model")


def test_Leeway():
    
    m = OpenDriftModel(
        drift_model="Leeway", object_type=">PIW, scuba suit (face up)",
        stokes_drift=False, steps=1, wind_drift_factor=None, wind_drift_depth=None,
    )
    assert m.manager_config.wind_drift_factor is None
    assert "seed:wind_drift_factor" not in m.show_all_config()
    
    with pytest.raises(ValidationError): 
        m = OpenDriftModel(drift_model="Leeway", stokes_drift=True, steps=1)

    with pytest.raises(ValidationError): 
        m = OpenDriftModel(drift_model="Leeway", wind_drift_factor=10, wind_drift_depth=10, steps=1)


def test_LarvalFish_disallowed_settings():
    """LarvalFish is incompatible with some settings.

    LarvalFish has to always be 3D.
    """

    with pytest.raises(ValueError):
        m = OpenDriftModel(drift_model="LarvalFish", vertical_mixing=False, steps=1)

    with pytest.raises(ValueError):
        m = OpenDriftModel(drift_model="LarvalFish", do3D=False, steps=1)



# class TestOpenDriftModel_LarvalFish(unittest.TestCase):
#     def setUp(self):
#         self.model = OpenDriftModel(drift_model="LarvalFish", do3D=True)

#     def test_drop_vars_wind(self):
#         self.model.duration = pd.Timedelta("1h")
#         self.model.add_reader(ds=ds)
#         assert self.model.reader.variables == [
#             "x_sea_water_velocity",
#             "y_sea_water_velocity",
#             "upward_sea_water_velocity",
#             "sea_water_salinity",
#             "sea_water_temperature",
#             "land_binary_mask",
#             "x_wind",
#             "y_wind",
#             "wind_speed",
#             "sea_water_speed",
#         ]


def test_LarvalFish_init():
    m = OpenDriftModel(drift_model="LarvalFish", 
                       do3D=True,
                        vertical_mixing=True,
                        wind_drift_factor=None,
                        wind_drift_depth=None,
                        steps=1
                        length=10
                       )

    

def test_LarvalFish_seeding():
    """Make sure special seed parameter comes through"""

    m = OpenDriftModel(
        drift_model="LarvalFish",
        lon=-151,
        lat=60,
        do3D=True,
        hatched=1,
        start_time="2022-01-01T00:00:00",
        use_auto_landmask=True,
        steps=1,
        vertical_mixing=True,
        wind_drift_factor=None,
        wind_drift_depth=None
    )
    # m.add_reader()
    # m.seed()
    # assert m.o.elements_scheduled.hatched == 1
    assert m.o._config["seed:hatched"]["value"] == 1


def test_OpenOil_seeding():
    """Make sure special seed parameters comes through"""

    m = OpenDriftModel(
        drift_model="OpenOil",
        lon=-151,
        lat=60,
        do3D=True,
        start_time="2023-01-01T00:00:00",
        use_auto_landmask=True,
        m3_per_hour=5,
        droplet_diameter_max_subsea=0.1,
        droplet_diameter_min_subsea=0.01,
        droplet_diameter_mu=0.01,
        droplet_size_distribution="normal",
        droplet_diameter_sigma=10,
        oil_film_thickness=5,
        oil_type="GENERIC DIESEL",
        steps=1
    )

    # m.o.set_config("environment:constant:x_wind", -1)
    # m.o.set_config("environment:constant:y_wind", -1)
    # m.o.set_config("environment:constant:x_sea_water_velocity", -1)
    # m.o.set_config("environment:constant:y_sea_water_velocity", -1)
    # m.o.set_config("environment:constant:sea_water_temperature", 15)
    # m.seed()

    # to check impact of m3_per_hour: mass_oil for m3_per_hour of 1 * 5
    # assert np.allclose(m.o.elements_scheduled.mass_oil, 0.855 * 5)  # i'm getting different answers local vs github actiosn
    assert m.o._config["seed:m3_per_hour"]["value"] == 5
    assert m.o._config["seed:droplet_diameter_max_subsea"]["value"] == 0.1
    assert m.o._config["seed:droplet_diameter_min_subsea"]["value"] == 0.01
    assert m.o._config["seed:droplet_diameter_mu"]["value"] == 0.01
    assert m.o._config["seed:droplet_size_distribution"]["value"] == "normal"
    assert m.o._config["seed:droplet_diameter_sigma"]["value"] == 10
    # assert m.o.elements_scheduled.oil_film_thickness == 5
    assert m.o._config["seed:oil_type"]["value"] == "GENERIC DIESEL"


def test_wind_drift():
    """Make sure changed wind drift numbers comes through"""

    m = OpenDriftModel(
        drift_model="OceanDrift",
        lon=-151,
        lat=60,
        do3D=True,
        wind_drift_factor=1,
        wind_drift_depth=10,
        start_time="2023-01-01T00:00:00",
        use_auto_landmask=True,
        steps=1
    )
    # m.add_reader()
    # m.seed()
    # assert m.o.elements_scheduled.wind_drift_factor == 1
    assert m.o._config["seed:wind_drift_factor"]["value"] == 1
    assert m.o._config["drift:wind_drift_depth"]["value"] == 10


# def test_plots_linecolor():
#     # this should error if user inputs some export_variables, which
#     # changes the default from returning all variables to just those
#     # selected plus a short list of required variables
#     with pytest.raises(ValueError):
#         m = OpenDriftModel(
#             drift_model="OceanDrift",
#             plots={"spaghetti": {"linecolor": "x_wind"}},
#             export_variables=[],
#         )

#     m = OpenDriftModel(
#         drift_model="OceanDrift",
#         plots={"spaghetti": {"linecolor": "x_wind"}},
#         export_variables=None,
#     )

#     # this should work bc "z" should already be included
#     m = OpenDriftModel(
#         drift_model="OceanDrift", plots={"spaghetti": {"linecolor": "z"}}
#     )


# def test_plots_background():
#     # this should error if user inputs some export_variables, which
#     # changes the default from returning all variables to just those
#     # selected plus a short list of required variables
#     with pytest.raises(ValueError):
#         m = OpenDriftModel(
#             drift_model="OceanDrift",
#             plots={"animation": {"background": "sea_surface_height"}},
#             export_variables=[],
#         )

#     m = OpenDriftModel(
#         drift_model="OceanDrift",
#         plots={"animation": {"background": "sea_surface_height"}},
#     )


# def test_plots_oil():
#     # this should error if user inputs some export_variables, which
#     # changes the default from returning all variables to just those
#     # selected plus a short list of required variables
#     with pytest.raises(ValueError):
#         m = OpenDriftModel(
#             drift_model="OpenOil",
#             plots={"oil": {"show_wind_and_current": True}},
#             export_variables=[],
#         )

#     m = OpenDriftModel(
#         drift_model="OpenOil", plots={"oil": {"show_wind_and_current": True}}
#     )

#     with pytest.raises(ValueError):
#         m = OpenDriftModel(drift_model="OceanDrift", plots={"oil": {}})


# def test_plots_property():
#     # this should error if user inputs some export_variables, which
#     # changes the default from returning all variables to just those
#     # selected plus a short list of required variables
#     with pytest.raises(ValueError):
#         m = OpenDriftModel(
#             drift_model="LarvalFish",
#             do3D=True,
#             plots={"property": {"prop": "survival"}},
#             export_variables=["x_wind"],
#         )

#     m = OpenDriftModel(
#         drift_model="LarvalFish",
#         do3D=True,
#         plots={"property": {"prop": "survival"}},
#     )


# def test_plots_all():

#     with pytest.raises(ValueError):
#         m = OpenDriftModel(
#             drift_model="OceanDrift",
#             plots={
#                 "all": {},
#                 "spaghetti": {"line_color": "x_wind"},
#                 "animation": {"background": "sea_surface_height"},
#             },
#         )



# TODO: Add tests, such as from test_manager, that test the known models




if __name__ == "__main__":
    unittest.main()
