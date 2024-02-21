"""Test realistic scenarios, which are slower."""

import pytest
import xarray as xr

import particle_tracking_manager as ptm


@pytest.mark.slow
def test_add_new_reader():
    """Add a separate reader from the defaults."""

    import xroms

    manager = ptm.OpenDriftModel(use_static_masks=True, steps=2)

    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    ds = xr.open_dataset(url, decode_times=False)
    manager.add_reader(ds=ds)


@pytest.mark.slow
def test_run():
    """Set up and run."""

    import xroms

    seeding_kwargs = dict(lon=-90, lat=28.7, number=1)
    manager = ptm.OpenDriftModel(**seeding_kwargs, use_static_masks=True, steps=2)
    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    ds = xr.open_dataset(url, decode_times=False)
    manager.add_reader(ds=ds, name="txla")
    manager.seed()
    manager.run()
