from datetime import datetime
from unittest import mock

import numpy as np
import xarray as xr

import particle_tracking_manager as ptm

from particle_tracking_manager.models.opendrift.reader_ROMS_native import Reader


# @mock.patch("xarray.open_mfdataset")
@mock.patch("xarray.open_dataset")
def test_init(mock_dataset):
    """Test initialization of ROMS reader."""

    ds = xr.Dataset()
    ds["ocean_time"] = (
        ("ocean_time"),
        [1.258632e09, 1.258646e09, 1.258660e09],
        {"units": "seconds since 1970-01-01"},
    )
    ds["lat_rho"] = (("eta_rho", "xi_rho"), np.array([[54, 55, 56], [55, 56, 57]]))
    ds["lon_rho"] = (
        ("eta_rho", "xi_rho"),
        np.array([[-151, -151, -151], [-150, -150, -150]]),
    )
    ds["salt"] = (("eta_rho", "xi_rho"), np.ones((2, 3)))
    ds["temp"] = (("eta_rho", "xi_rho"), np.ones((2, 3)))

    mock_dataset.return_value = ds

    reader = Reader(filename="test")


# @pytest.mark.slow
# def test_add_new_reader():
#     manager = ptm.OpenDriftModel()

#     url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
#     reader_kwargs = dict(loc=url, kwargs_xarray={})
#     manager.add_reader(**reader_kwargs)

#     assert len(manager.o.env.readers) == 1
#     assert "roms native" in manager.o.env.readers


# @pytest.mark.slow
# def test_get_variables():
#     manager = ptm.OpenDriftModel()

#     url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
#     reader_kwargs = dict(loc=url, kwargs_xarray={})
#     manager.add_reader(**reader_kwargs)

#     # Set up test data
#     requested_variables = ["sea_floor_depth_below_sea_level", "x_sea_water_velocity"]
#     time = datetime.now()
#     x = 10.0
#     y = 20.0
#     z = None

#     variables = manager.o.env.readers["roms native"].get_variables(
#         requested_variables, time, x, y, z
#     )

#     # Assert the returned variables
#     assert "sea_floor_depth_below_sea_level" in variables
#     assert "x_sea_water_velocity" in variables

#     # Assert the shape of the variables
#     assert variables["sea_floor_depth_below_sea_level"].shape == (1,)
#     assert variables["x_sea_water_velocity"].shape == (1,)

#     # Assert the values of the variables
#     assert variables["sea_floor_depth_below_sea_level"][0] == pytest.approx(10.0, abs=1e-6)
#     assert variables["x_sea_water_velocity"][0] == pytest.approx(20.0, abs=1e-6)
