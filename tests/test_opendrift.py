"""From Copilot"""

import unittest

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from particle_tracking_manager.models.opendrift.opendrift import (
    OpenDriftModel,
    config_model,
)


class TestOpenDriftModel(unittest.TestCase):
    def setUp(self):
        self.odm = OpenDriftModel()

    def test_init(self):
        self.assertEqual(self.odm.drift_model, config_model["drift_model"]["default"])
        self.assertEqual(self.odm.radius, config_model["radius"]["default"])
        self.assertEqual(self.odm.radius_type, config_model["radius_type"]["default"])
        # self.assertEqual(self.odm.horizontal_diffusivity, config_model["horizontal_diffusivity"]["default"])
        self.assertEqual(
            self.odm.use_auto_landmask, config_model["use_auto_landmask"]["default"]
        )
        self.assertEqual(
            self.odm.diffusivitymodel, config_model["diffusivitymodel"]["default"]
        )
        self.assertEqual(self.odm.stokes_drift, config_model["stokes_drift"]["default"])
        self.assertEqual(
            self.odm.mixed_layer_depth, config_model["mixed_layer_depth"]["default"]
        )
        self.assertEqual(
            self.odm.coastline_action, config_model["coastline_action"]["default"]
        )
        self.assertEqual(self.odm.max_speed, config_model["max_speed"]["default"])
        self.assertEqual(
            self.odm.wind_drift_factor, config_model["wind_drift_factor"]["default"]
        )
        self.assertEqual(
            self.odm.wind_drift_depth, config_model["wind_drift_depth"]["default"]
        )
        self.assertEqual(
            self.odm.vertical_mixing_timestep,
            config_model["vertical_mixing_timestep"]["default"],
        )
        self.assertEqual(self.odm.object_type, config_model["object_type"]["default"])
        self.assertEqual(self.odm.diameter, config_model["diameter"]["default"])
        self.assertEqual(
            self.odm.neutral_buoyancy_salinity,
            config_model["neutral_buoyancy_salinity"]["default"],
        )
        self.assertEqual(
            self.odm.stage_fraction, config_model["stage_fraction"]["default"]
        )
        self.assertEqual(self.odm.hatched, config_model["hatched"]["default"])
        self.assertEqual(self.odm.length, config_model["length"]["default"])
        self.assertEqual(self.odm.weight, config_model["weight"]["default"])
        self.assertEqual(self.odm.oil_type, config_model["oil_type"]["default"])
        self.assertEqual(self.odm.m3_per_hour, config_model["m3_per_hour"]["default"])
        self.assertEqual(
            self.odm.oil_film_thickness, config_model["oil_film_thickness"]["default"]
        )
        self.assertEqual(
            self.odm.droplet_size_distribution,
            config_model["droplet_size_distribution"]["default"],
        )
        self.assertEqual(
            self.odm.droplet_diameter_mu, config_model["droplet_diameter_mu"]["default"]
        )
        self.assertEqual(
            self.odm.droplet_diameter_sigma,
            config_model["droplet_diameter_sigma"]["default"],
        )
        self.assertEqual(
            self.odm.droplet_diameter_min_subsea,
            config_model["droplet_diameter_min_subsea"]["default"],
        )
        self.assertEqual(
            self.odm.droplet_diameter_max_subsea,
            config_model["droplet_diameter_max_subsea"]["default"],
        )
        self.assertEqual(
            self.odm.emulsification, config_model["emulsification"]["default"]
        )
        self.assertEqual(self.odm.dispersion, config_model["dispersion"]["default"])
        self.assertEqual(self.odm.evaporation, config_model["evaporation"]["default"])
        self.assertEqual(
            self.odm.update_oilfilm_thickness,
            config_model["update_oilfilm_thickness"]["default"],
        )
        self.assertEqual(
            self.odm.biodegradation, config_model["biodegradation"]["default"]
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


class TestOpenDriftModel_OceanDrift(unittest.TestCase):
    def setUp(self):
        self.model = OpenDriftModel(drift_model="OceanDrift")

    def test_error_no_end_of_simulation(self):
        self.model.do3D = False
        self.use_static_masks = False
        # need to input steps, duration, or end_time but don't here
        with pytest.raises(ValueError):
            self.model.add_reader(ds=ds)

    def test_ocean_model_not_known_ds_None(self):
        self.model.ocean_model = "wrong_name"
        self.model.ds = None  # this is the default
        # need to input steps, duration, or end_time but don't here
        with pytest.raises(ValueError):
            self.model.add_reader(ds=ds)

    def test_drop_vars_do3D_false(self):
        self.model.do3D = False
        self.use_static_masks = False
        self.model.steps = 4
        self.model.add_reader(ds=ds)
        assert self.model.reader.variables == [
            "x_sea_water_velocity",
            "y_sea_water_velocity",
            "land_binary_mask",
            "x_wind",
            "y_wind",
            "wind_speed",
            "sea_water_speed",
        ]
        assert "wetdry_mask_rho" in self.model.reader.Dataset.data_vars
        assert "mask_rho" not in self.model.reader.Dataset.data_vars

    def test_drop_vars_do3D_true(self):
        self.model.do3D = True
        self.model.steps = 4
        self.model.add_reader(ds=ds)
        assert self.model.reader.variables == [
            "x_sea_water_velocity",
            "y_sea_water_velocity",
            "upward_sea_water_velocity",
            "land_binary_mask",
            "x_wind",
            "y_wind",
            "wind_speed",
            "sea_water_speed",
        ]

    def test_drop_vars_use_static_masks(self):
        self.model.do3D = False
        self.model.use_static_masks = True
        self.model.duration = pd.Timedelta("24h")
        self.model.add_reader(ds=ds)
        assert self.model.reader.variables == [
            "x_sea_water_velocity",
            "y_sea_water_velocity",
            "land_binary_mask",
            "x_wind",
            "y_wind",
            "wind_speed",
            "sea_water_speed",
        ]
        assert "mask_rho" in self.model.reader.Dataset.data_vars
        assert "wetdry_mask_rho" not in self.model.reader.Dataset.data_vars

    def test_drop_vars_no_wind(self):
        self.model.stokes_drift = False
        self.model.wind_drift_factor = 0
        self.model.wind_uncertainty = 0
        self.model.vertical_mixing = False
        self.model.end_time = pd.Timestamp("1970-01-01T02:00")
        self.model.add_reader(ds=ds)
        assert self.model.reader.variables == [
            "x_sea_water_velocity",
            "y_sea_water_velocity",
            "land_binary_mask",
            "sea_water_speed",
        ]


class TestOpenDriftModel_LarvalFish(unittest.TestCase):
    def setUp(self):
        self.model = OpenDriftModel(drift_model="LarvalFish")

    def test_drop_vars_do3D_false(self):
        self.model.do3D = False
        self.model.end_time = pd.Timestamp("1970-01-01T02:00")
        self.model.add_reader(ds=ds)
        assert self.model.reader.variables == [
            "x_sea_water_velocity",
            "y_sea_water_velocity",
            "sea_water_salinity",
            "sea_water_temperature",
            "land_binary_mask",
            "x_wind",
            "y_wind",
            "wind_speed",
            "sea_water_speed",
        ]

    def test_drop_vars_do3D_true(self):
        self.model.do3D = True
        self.model.duration = pd.Timedelta("1h")
        self.model.add_reader(ds=ds)
        assert self.model.reader.variables == [
            "x_sea_water_velocity",
            "y_sea_water_velocity",
            "upward_sea_water_velocity",
            "sea_water_salinity",
            "sea_water_temperature",
            "land_binary_mask",
            "x_wind",
            "y_wind",
            "wind_speed",
            "sea_water_speed",
        ]

    def test_drop_vars_no_wind(self):
        self.model.stokes_drift = False
        self.model.wind_drift_factor = 0
        self.model.wind_uncertainty = 0
        self.model.vertical_mixing = False
        self.model.steps = 3
        self.model.add_reader(ds=ds)
        # import pdb; pdb.set_trace()
        assert self.model.reader.variables == [
            "x_sea_water_velocity",
            "y_sea_water_velocity",
            "sea_water_salinity",
            "sea_water_temperature",
            "land_binary_mask",
            "sea_water_speed",
        ]


def test_drift_model():
    """can't change the drift_model after class initialization"""
    with pytest.raises(ValueError):
        m = OpenDriftModel(drift_model="not_a_real_model")


class TestTheManager(unittest.TestCase):
    def setUp(self):
        self.m = OpenDriftModel()
        # self.m.config_model = {"test_key": {"value": "old_value"}}

    def test_set_drift_model(self):
        with self.assertRaises(KeyError):
            self.m.drift_model = "new_model"

    def test_set_o(self):
        with self.assertRaises(KeyError):
            self.m.o = "new_o"

    def test_stokes_drift_true_not_leeway(self):
        self.m.stokes_drift = True
        self.m.drift_model = "OceanDrift"
        assert self.m.show_config(key="drift:use_tabularised_stokes_drift")

    def test_surface_only_true_not_leeway(self):
        self.m.surface_only = True
        self.m.drift_model = "OceanDrift"
        assert (
            self.m.show_config(key="drift:truncate_ocean_model_below_m")["value"] == 0.5
        )

    def test_surface_only_false_not_leeway(self):
        self.m.surface_only = False
        self.m.drift_model = "OceanDrift"
        assert (
            self.m.show_config(key="drift:truncate_ocean_model_below_m")["value"]
            is None
        )

    def test_do3D_false_not_leeway(self):
        self.m.do3D = False
        self.m.drift_model = "OceanDrift"
        assert not self.m.show_config(key="drift:vertical_advection")["value"]
        assert not self.m.show_config(key="drift:vertical_mixing")["value"]

    def test_do3D_true(self):
        self.m.do3D = True
        # assert self.m.show_config(key="drift:vertical_advection")["value"]

    def test_vertical_mixing_false_vertical_mixing_timestep_not_default(self):
        self.m.vertical_mixing = False
        self.m.vertical_mixing_timestep = 10
        d = self.m.show_config(key="vertical_mixing_timestep")
        assert d["value"] == d["default"]

    def test_vertical_mixing_false_diffusivitymodel_not_default(self):
        self.m.vertical_mixing = False
        self.m.diffusivitymodel = "not_default"
        d = self.m.show_config(key="diffusivitymodel")
        assert d["value"] == d["default"]

    def test_vertical_mixing_false_mixed_layer_depth_not_default(self):
        self.m.vertical_mixing = False
        self.m.mixed_layer_depth = 10
        d = self.m.show_config(key="mixed_layer_depth")
        assert d["value"] == d["default"]


class TestOpenDriftModel_Leeway(unittest.TestCase):
    def setUp(self):
        self.m = OpenDriftModel(drift_model="Leeway")

    def test_leeway_model_wind_drift_factor_not_default(self):
        self.m.wind_drift_factor = 10
        d = self.m.show_config(key="wind_drift_factor")
        assert d["value"] == d["default"]

    def test_leeway_model_wind_drift_depth_not_default(self):
        self.m.wind_drift_depth = 10
        d = self.m.show_config(key="wind_drift_depth")
        assert d["value"] == d["default"]

    def test_leeway_model_stokes_drift_true(self):
        self.m.stokes_drift = True
        assert not self.m.stokes_drift
        # assert not self.m.show_config(key="stokes_drift")["value"]


def test_horizontal_diffusivity_logic():
    """Check logic for using default horizontal diff values for known models."""

    m = OpenDriftModel()
    m.ocean_model = "NWGOA"
    assert m.horizontal_diffusivity == 150.0  # known grid values
    m.ocean_model = "CIOFS"
    assert m.horizontal_diffusivity == 10.0  # known grid values

    # or can overwrite it in this order
    m.horizontal_diffusivity = 11
    assert m.horizontal_diffusivity == 11.0  # user-selected value

    m.ocean_model = "CIOFSOP"
    assert m.horizontal_diffusivity == 10.0  # known grid values

    m = OpenDriftModel(ocean_model="NWGOA")
    assert m.horizontal_diffusivity == 150.0  # known grid values


if __name__ == "__main__":
    unittest.main()
