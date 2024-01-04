"""Test realistic scenarios, which are slower."""

import xroms
from opendrift.readers import reader_ROMS_native
import particle_tracking_manager as ptm
import pytest


@pytest.mark.slow
def test_add_new_reader():
    """Add a separate reader from the defaults."""
    
    manager = ptm.OpenDrift()
    manager.config()
    
    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    reader_kwargs = dict(loc=url, kwargs_xarray={})
    manager.add_reader(**reader_kwargs)


@pytest.mark.slow
def test_run():
    """Set up and run."""

    seeding_kwargs = dict(lon = -90, lat = 28.7, number=1)
    manager = ptm.OpenDrift(**seeding_kwargs)
    manager.config()
    
    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    reader_kwargs = dict(loc=url, kwargs_xarray={})
    manager.add_reader(**reader_kwargs)
    # can find reader at manager.o.env.readers['roms native']

    manager.start_time = manager.o.env.readers['roms native'].start_time
    manager.seed()
    manager.run()