"""Test manager use in library, the default approach."""


from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pydantic import ValidationError

from particle_tracking_manager.config_the_manager import TheManagerConfig
from particle_tracking_manager.the_manager import ParticleTrackingManager


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

    def _setup_for_simulation(self):
        pass


def test_order():
    """Have to configure before seeding."""

    m = TestParticleTrackingManager(steps=1, start_time="2022-01-01")
    with pytest.raises(ValueError):
        m.run()


def test_seed_order():
    m = TestParticleTrackingManager(steps=1, start_time="2022-01-01")
    m.state.has_added_reader = True
    m.seed()

    m = TestParticleTrackingManager(steps=1, start_time="2022-01-01")
    with pytest.raises(ValueError):
        m.seed()


def test_log_name():
    m = TestParticleTrackingManager(output_file="newtest", steps=1)
    assert m.files.logfile_name == "newtest.log"

    m = TestParticleTrackingManager(output_file="newtest.nc", steps=1)
    assert m.files.logfile_name == "newtest.log"

    m = TestParticleTrackingManager(output_file="newtest.parq", steps=1)
    assert m.files.logfile_name == "newtest.log"

    m = TestParticleTrackingManager(output_file="newtest.parquet", steps=1)
    assert m.files.logfile_name == "newtest.log"


def test_output_file():
    """make sure output file is parquet if output_format is parquet"""

    m = TestParticleTrackingManager(output_format="parquet", steps=1)
    assert m.files.output_file.suffix == ".parquet"

    m = TestParticleTrackingManager(output_format="netcdf", steps=1)
    assert m.files.output_file.suffix == ".nc"


def test_ocean_model_not_known():
    with pytest.raises(ValidationError):
        TestParticleTrackingManager(ocean_model="wrong_name", steps=1)
