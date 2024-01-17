"""From Copilot"""

import unittest

from particle_tracking_manager.models.opendrift.model_opendrift import (
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


if __name__ == "__main__":
    unittest.main()
