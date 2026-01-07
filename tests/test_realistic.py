"""Test realistic scenarios, which are slower."""

import pickle

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import particle_tracking_manager as ptm


def is_netcdf(path):
    with open(path, "rb") as f:
        sig = f.read(8)
    return sig.startswith(b"CDF") or sig.startswith(b"\x89HDF\r\n\x1a\n")


def is_parquet(path):
    with open(path, "rb") as f:
        start = f.read(4)
        f.seek(-4, 2)
        end = f.read(4)
    return start == b"PAR1" and end == b"PAR1"


# set up an alternate dataset on-the-fly
ds = xr.Dataset(
    data_vars={
        "u": (("ocean_time", "Z", "Y", "X"), np.ones((2, 3, 4, 5))),
        "v": (("ocean_time", "Z", "Y", "X"), np.ones((2, 3, 4, 5))),
        "w": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 4, 5))),
        "salt": (("ocean_time", "Z", "Y", "X"), np.ones((2, 3, 4, 5)) * 31),
        "temp": (("ocean_time", "Z", "Y", "X"), np.ones((2, 3, 4, 5)) * 18),
        "wetdry_mask_rho": (("ocean_time", "Y", "X"), np.ones((2, 4, 5))),
        "mask_rho": (("Y", "X"), np.ones((4, 5))),
        "h": (("Y", "X"), np.ones((4, 5)) * 10),
        "angle": (("Y", "X"), np.zeros((4, 5))),
        "Uwind": (("ocean_time", "Y", "X"), np.zeros((2, 4, 5))),
        "Vwind": (("ocean_time", "Y", "X"), np.zeros((2, 4, 5))),
        "Cs_r": (("Z"), np.linspace(-1, 0, 3)),
        "hc": 16,
    },
    coords={
        # "ocean_time": ("ocean_time", ["1970-01-01T00:00:00", "1970-01-01T00:10:00"], {"units": "seconds since 1970-01-01"}),
        "ocean_time": (
            "ocean_time",
            [0, 60 * 10],
            {"units": "seconds since 1970-01-01"},
        ),
        "s_rho": (("Z"), np.linspace(-1, 0, 3)),
        "lon_rho": (
            ("Y", "X"),
            np.array(
                [
                    [1, 1.5, 2, 2.5, 3],
                    [1, 1.5, 2, 2.5, 3],
                    [1, 1.5, 2, 2.5, 3],
                    [1, 1.5, 2, 2.5, 3],
                ]
            ),
        ),
        "lat_rho": (
            ("Y", "X"),
            np.array(
                [
                    [1, 1.25, 1.5, 1.75, 2],
                    [1, 1.25, 1.5, 1.75, 2],
                    [1, 1.25, 1.5, 1.75, 2],
                    [1, 1.25, 1.5, 1.75, 2],
                ]
            ),
        ),
    },
)
ds_info = dict(
    lon_min=1,
    lon_max=3,
    lat_min=1,
    lat_max=2,
    start_time_model=0,
    end_time_fixed=60 * 10,
)

ptm.config_ocean_model.register_on_the_fly(ds_info)


# also to use the user-defined template of the TXLA model, need to input where pooch is downloading
# the file
ptm.config_ocean_model.update_TXLA_with_download_location()


@pytest.mark.slow
def test_add_new_reader():
    """Add a separate reader from the defaults using ds."""

    manager = ptm.OpenDriftModel(
        steps=1, ocean_model="ONTHEFLY", lon=1.2, lat=1.2, start_time=0, time_step=0.01
    )
    manager.add_reader(ds=ds)


@pytest.mark.slow
def test_run_netcdf_and_plot():
    """Set up and run."""

    import tempfile

    ts = 6 * 60  # 6 minutes in seconds

    seeding_kwargs = dict(lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        manager = ptm.OpenDriftModel(
            **seeding_kwargs,
            use_static_masks=True,
            steps=2,
            output_format="netcdf",
            use_cache=True,
            interpolator_filename=temp_file.name,
            ocean_model="TXLA",
            ocean_model_local=False,
            plots={
                "all": {},
            },
            time_step=ts,
        )
        manager.run_all()

        assert "nc" in manager.o.outfile_name
        assert manager.config.interpolator_filename == Path(temp_file.name).with_suffix(
            ".pickle"
        )

        # Replace 'path_to_pickle_file.pkl' with the actual path to your pickle file
        with open(manager.config.interpolator_filename, "rb") as file:
            data = pickle.load(file)
        assert "spl_x" in data
        assert "spl_y" in data

    # check time_step across access points
    assert (
        # m.o._config["general:time_step_minutes"]["value"]  # this is not correct, don't know why
        manager.o.time_step.total_seconds()
        == ts
        == manager.config.time_step
        # == m.o.get_configspec()["general:time_step_minutes"]["value"]  # this is not correct, don't know why
    )


@pytest.mark.slow
def test_run_HarmfulAlgalBloom_biomass_change():
    """Set up and run HarmfulAlgalBloom and match biomass change."""

    seeding_kwargs = dict(lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00")
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="HarmfulAlgalBloom",
    )
    m.add_reader()

    # change model values for test
    inds = m.ds.temp.notnull()
    m.ds["temp"].values[inds] = 15.0
    m.ds["salt"].values[inds] = 32.0
    m.run_all()

    # check that biomass decreased due to temperature-induced mortality
    # calculated as: biomass = initial_biomass * exp(growth_rate-mortality_rate_high * time)
    assert np.allclose(
        float(m.o.elements.biomass[0]),
        np.exp(
            (m.config.growth_rate_high - m.config.mortality_rate_low) * 3600 / 86400
        ),
    )


@pytest.mark.slow
def test_run_HarmfulAlgalBloom_vertical_behavior_band():
    """Set up and run HarmfulAlgalBloom and match vertical change."""

    seeding_kwargs = dict(
        lon=-90,
        lat=28.7,
        number=1,
        start_time="2009-11-19T12:00:00",
        swim_speed=0.001,
        z=-30,
    )
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="HarmfulAlgalBloom",
        vertical_behavior="band",
        band_center_depth=-10.0,
        band_half_width=0.0,
        vertical_mixing=False,
        do3D=False,
    )
    m.add_reader()
    m.run_all()

    dz = seeding_kwargs["swim_speed"] * 3600  # swim speed * time in seconds
    assert np.allclose(float(m.o.elements.z[0]), seeding_kwargs["z"] + dz)


@pytest.mark.slow
def test_run_HarmfulAlgalBloom_vertical_behavior_diel_band():
    """Set up and run HarmfulAlgalBloom with diel_band and match vertical change.

    Particle starts deeper than both diel_day_depth and diel_night_depth, so
    it should swim upward at constant swim_speed for the duration, independent
    of whether this hour is classified as day or night.
    """

    seeding_kwargs = dict(
        lon=-90.0,
        lat=28.7,
        number=1,
        start_time="2009-11-19T12:00:00",  # same as band test
        swim_speed=0.001,
        z=-50.0,
    )

    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="HarmfulAlgalBloom",
        vertical_behavior="diel_band",
        diel_day_depth=-20.0,
        diel_night_depth=-10.0,
        vertical_mixing=False,
        do3D=False,
    )
    m.add_reader()
    m.run_all()

    # Analytical expectation: no vertical advection/mixing, so only active
    # swimming contributes. 1 hour = 3600 s.
    dz = seeding_kwargs["swim_speed"] * 3600  # swim_speed * time (s)
    expected_z = seeding_kwargs["z"] + dz

    assert np.allclose(float(m.o.elements.z[0]), expected_z)


# reinstitute this test once OpenDrift PR is accepted that outputs parquet files directly
# @pytest.mark.slow
# def test_run_parquet():
#     """Set up and run."""

#     seeding_kwargs = dict(lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00")
#     manager = ptm.OpenDriftModel(
#         **seeding_kwargs,
#         use_static_masks=True,
#         steps=2,
#         output_format="parquet",
#         ocean_model="TXLA",
#         ocean_model_local=False,
#     )
#     manager.run_all()

#     assert "parquet" in manager.o.outfile_name


@pytest.mark.slow
def test_run_parquet_and_netcdf():
    """Set up and run."""

    seeding_kwargs = dict(lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00")
    manager = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        steps=2,
        output_format="both",
        ocean_model="TXLA",
        ocean_model_local=False,
    )
    manager.run_all()

    assert "nc" in manager.o.outfile_name

    # 2. parquet file with same stem exists
    out_parquet = Path(manager.o.outfile_name).with_suffix(".parquet")
    assert out_parquet.exists()

    # Check actual file format signatures
    assert is_netcdf(manager.o.outfile_name), "NC file is not valid netCDF"
    assert not is_parquet(
        manager.o.outfile_name
    ), "NC file is incorrectly a parquet file"

    assert is_parquet(out_parquet), "Parquet file is not valid parquet"
    assert not is_netcdf(out_parquet), "Parquet file is incorrectly netCDF"


@pytest.mark.slow
def test_run_HarmfulAlgalBloom_PN():
    """Set up and run HarmfulAlgalBloom for Pseudo Nitzschia."""

    seeding_kwargs = dict(
        lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00", z=-20
    )
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="HarmfulAlgalBloom",
        species_type="PN",
        vertical_mixing=False,
        do3D=False,
    )
    m.add_reader()

    # change model values for test
    inds = m.ds.temp.notnull()
    m.ds["temp"].values[inds] = m.config.temperature_pref_min
    m.ds["salt"].values[inds] = (
        m.config.salinity_death_min + 0.5
    )  # this causes medium mortality/growth
    m.run_all()

    # check that biomass decreased due to temperature-induced mortality
    # calculated as: biomass = initial_biomass * exp(growth_rate-mortality_rate_high * time)
    assert np.allclose(
        float(m.o.elements.biomass[0]),
        np.exp(
            (m.config.growth_rate_medium - m.config.mortality_rate_medium)
            * 3600
            / 86400
        ),
    )

    # Analytical expectation: no vertical advection/mixing, so only active
    # swimming contributes. 1 hour = 3600 s.
    # particles are swimming downward from near the surface
    dz = m.config.swim_speed * 3600  # swim_speed * time (s)
    expected_z = seeding_kwargs["z"] - dz
    assert np.allclose(float(m.o.elements.z[0]), expected_z)


@pytest.mark.slow
def test_run_HarmfulAlgalBloom_AX():
    """Set up and run HarmfulAlgalBloom for Alexandrium."""

    seeding_kwargs = dict(
        lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00", z=-40
    )
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="HarmfulAlgalBloom",
        species_type="AX",
        vertical_mixing=False,
        do3D=False,
    )
    m.add_reader()

    # change model values for test
    inds = m.ds.temp.notnull()
    m.ds["temp"].values[inds] = (
        m.config.temperature_death_max + 0.5
    )  # this causes high mortality/low growth
    m.ds["salt"].values[inds] = m.config.salinity_pref_max
    m.run_all()

    # check that biomass decreased due to temperature-induced mortality
    # calculated as: biomass = initial_biomass * exp(growth_rate-mortality_rate_high * time)
    assert np.allclose(
        float(m.o.elements.biomass[0]),
        np.exp(
            (m.config.growth_rate_low - m.config.mortality_rate_high) * 3600 / 86400
        ),
    )

    # Analytical expectation: no vertical advection/mixing, so only active
    # swimming contributes. 1 hour = 3600 s.
    dz = m.config.swim_speed * 3600  # swim_speed * time (s)
    expected_z = seeding_kwargs["z"] + dz
    assert np.allclose(float(m.o.elements.z[0]), expected_z)


@pytest.mark.slow
def test_run_HarmfulAlgalBloom_DP():
    """Set up and run HarmfulAlgalBloom for Dinophysis."""

    seeding_kwargs = dict(
        lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00", z=-40
    )
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="HarmfulAlgalBloom",
        species_type="DP",
        vertical_mixing=False,
        do3D=False,
    )
    m.add_reader()

    # change model values for test
    inds = m.ds.temp.notnull()
    m.ds["temp"].values[inds] = m.config.temperature_pref_max
    m.ds["salt"].values[inds] = m.config.salinity_pref_max
    m.run_all()

    # check that biomass decreased due to temperature-induced mortality
    # calculated as: biomass = initial_biomass * exp(growth_rate-mortality_rate_high * time)
    assert np.allclose(
        float(m.o.elements.biomass[0]),
        np.exp(
            (m.config.growth_rate_high - m.config.mortality_rate_low) * 3600 / 86400
        ),
    )

    # Analytical expectation: no vertical advection/mixing, so only active
    # swimming contributes. 1 hour = 3600 s.
    dz = m.config.swim_speed * 3600  # swim_speed * time (s)
    expected_z = seeding_kwargs["z"] + dz
    assert np.allclose(float(m.o.elements.z[0]), expected_z)
