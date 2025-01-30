"""Test realistic scenarios, which are slower."""

import pickle

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
def test_run_parquet():
    """Set up and run."""

    import xroms

    seeding_kwargs = dict(lon=-90, lat=28.7, number=1)
    manager = ptm.OpenDriftModel(
        **seeding_kwargs, use_static_masks=True, steps=2, output_format="parquet"
    )
    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    ds = xr.open_dataset(url, decode_times=False)
    manager.add_reader(ds=ds, name="txla")
    manager.seed()
    manager.run()

    assert "parq" in manager.o.outfile_name


@pytest.mark.slow
def test_run_netcdf():
    """Set up and run."""

    import tempfile

    import xroms

    seeding_kwargs = dict(lon=-90, lat=28.7, number=1)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        manager = ptm.OpenDriftModel(
            **seeding_kwargs,
            use_static_masks=True,
            steps=2,
            output_format="netcdf",
            use_cache=True,
            interpolator_filename=temp_file.name
        )
    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    ds = xr.open_dataset(url, decode_times=False)
    manager.add_reader(ds=ds, name="txla")
    manager.seed()
    manager.run()

    assert "nc" in manager.o.outfile_name
    assert manager.interpolator_filename == temp_file.name + ".pickle"

    # Replace 'path_to_pickle_file.pkl' with the actual path to your pickle file
    with open(manager.interpolator_filename, "rb") as file:
        data = pickle.load(file)
    assert "spl_x" in data
    assert "spl_y" in data
