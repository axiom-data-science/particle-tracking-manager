"""Test realistic scenarios, which are slower."""

import pytest

import particle_tracking_manager as ptm


@pytest.mark.slow
def test_add_new_reader():
    """Add a separate reader from the defaults."""

    import xroms

    manager = ptm.OpenDriftModel()

    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    reader_kwargs = dict(loc=url, kwargs_xarray={})
    manager.add_reader(**reader_kwargs)


@pytest.mark.slow
def test_run():
    """Set up and run."""

    import xroms

    seeding_kwargs = dict(lon=-90, lat=28.7, number=1)
    manager = ptm.OpenDriftModel(**seeding_kwargs)

    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    reader_kwargs = dict(loc=url, kwargs_xarray={})
    manager.add_reader(**reader_kwargs)
    # can find reader at manager.o.env.readers['roms native']

    manager.start_time = manager.o.env.readers["roms native"].start_time
    manager.seed()
    manager.run()
