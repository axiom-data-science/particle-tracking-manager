"""From Co-pilot"""

import unittest

from datetime import datetime, timedelta

import particle_tracking_manager as ptm


# from ptm.the_manager import ParticleTrackingManager


class TestParticleTrackingManager(unittest.TestCase):
    def setUp(self):
        self.ptm = ptm.ParticleTrackingManager(model="opendrift")

    def test_init(self):
        self.assertEqual(self.ptm.model, "opendrift")
        self.assertIsNone(self.ptm.lon)
        self.assertIsNone(self.ptm.lat)
        self.assertIsNone(self.ptm.start_time)
        self.assertIsNone(self.ptm.ocean_model)
        self.assertIsNone(self.ptm.surface_only)
        self.assertEqual(self.ptm.log, "low")

    def test_set_lon(self):
        self.ptm.lon = 50
        self.assertEqual(self.ptm.lon, 50)

    def test_set_lat(self):
        self.ptm.lat = 30
        self.assertEqual(self.ptm.lat, 30)

    def test_set_start_time(self):
        start_time = datetime.now()
        self.ptm.start_time = start_time
        self.assertEqual(self.ptm.start_time, start_time)

    def test_set_ocean_model(self):
        self.ptm.ocean_model = "CIOFS"
        self.assertEqual(self.ptm.ocean_model, "CIOFS")

    def test_set_surface_only(self):
        self.ptm.surface_only = True
        self.assertEqual(self.ptm.surface_only, True)
        self.assertEqual(self.ptm.do3D, False)
        self.assertEqual(self.ptm.z, 0)
        self.assertEqual(self.ptm.vertical_mixing, False)

    def test_set_do3D(self):
        self.ptm.do3D = True
        self.assertEqual(self.ptm.do3D, True)
        self.assertEqual(self.ptm.vertical_mixing, False)

    def test_set_seed_seafloor(self):
        self.ptm.seed_seafloor = True
        self.assertEqual(self.ptm.seed_seafloor, True)
        self.assertEqual(self.ptm.z, None)

    def test_set_time_step(self):
        self.ptm.time_step = 3600
        self.assertEqual(self.ptm.time_step, 3600)

    def test_set_time_step_output(self):
        self.ptm.time_step_output = 3600
        self.assertEqual(self.ptm.time_step_output, 3600)

    def test_set_steps(self):
        self.ptm.steps = 10
        self.assertEqual(self.ptm.steps, 10)

    def test_set_duration(self):
        duration = timedelta(hours=48)
        self.ptm.duration = duration
        self.assertEqual(self.ptm.duration, duration)

    def test_set_end_time(self):
        end_time = datetime.now() + timedelta(hours=48)
        self.ptm.end_time = end_time
        self.assertEqual(self.ptm.end_time, end_time)


if __name__ == "__main__":
    unittest.main()
