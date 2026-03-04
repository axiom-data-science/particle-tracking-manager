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
def test_run_Phytoplankton_basic():
    """Set up and run Phytoplankton and verify simulation completes successfully."""

    seeding_kwargs = dict(lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00")
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="Phytoplankton",
        vertical_behavior_mode="depth",
        z_pref=-10.0,
    )
    m.add_reader()
    m.run_all()

    # Verify simulation completed and particles exist
    assert len(m.o.elements) > 0


@pytest.mark.slow
def test_run_Phytoplankton_vertical_behavior_depth():
    """Set up and run Phytoplankton with depth mode and match vertical change."""

    seeding_kwargs = dict(
        lon=-90,
        lat=28.7,
        number=1,
        start_time="2009-11-19T12:00:00",
        w_active=0.001,
        z=-30,
    )
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="Phytoplankton",
        vertical_behavior_mode="depth",
        z_pref=-10.0,
    )
    m.add_reader()
    m.run_all()

    # Phytoplankton swims toward preferred depth with vertical mixing active
    # The particle should move upward from z=-30 toward z_pref=-10
    final_z = float(m.o.elements.z[0])
    assert final_z > seeding_kwargs["z"], "Particle should swim upward toward z_pref"


@pytest.mark.slow
def test_run_Phytoplankton_vertical_behavior_dvm():
    """Set up and run Phytoplankton with DVM mode and match vertical change.

    Particle starts deeper than both z_day and z_night, so
    it should swim upward toward the target depth for the time of day.
    """

    seeding_kwargs = dict(
        lon=-90.0,
        lat=28.7,
        number=1,
        start_time="2009-11-19T12:00:00",  # noon local time
        w_active=0.001,
        z=-50.0,
    )

    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="Phytoplankton",
        vertical_behavior_mode="dvm",
        z_day=-20.0,
        z_night=-10.0,
    )
    m.add_reader()
    m.run_all()

    # Particle should swim upward from z=-50 toward daytime depth (z_day=-20)
    # With vertical mixing active, the exact final position varies,
    # but it should be shallower than starting depth
    final_z = float(m.o.elements.z[0])
    assert final_z > seeding_kwargs["z"], "Particle should swim upward toward target depth"


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
def test_run_LarvalFish_vertical_behavior_depth():
    """Set up and run LarvalFish with depth mode vertical behavior."""

    seeding_kwargs = dict(
        lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00", z=-20
    )
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="LarvalFish",
        vertical_behavior_mode="depth",
        z_pref=-15.0,
        w_active=0.003,
        hatched=1,
        do3D=True,
    )
    m.add_reader()
    m.run_all()

    # Larvae start at z=-20 and swim toward preferred depth z_pref=-15
    # With vertical mixing active, the particle should move upward
    final_z = float(m.o.elements.z[0])
    assert final_z > seeding_kwargs["z"], "Larvae should swim upward toward z_pref"


@pytest.mark.slow
def test_run_LarvalFish_vertical_behavior_dvm():
    """Set up and run LarvalFish with DVM mode vertical behavior."""

    seeding_kwargs = dict(
        lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00", z=-40
    )
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="LarvalFish",
        vertical_behavior_mode="dvm",
        z_day=-20.0,
        z_night=-8.0,
        w_active=0.003,
        hatched=1,
        do3D=True,
    )
    m.add_reader()
    m.run_all()

    # Larvae start at z=-40 and swim toward daytime depth z_day=-20
    # With vertical mixing active, the particle should move upward
    final_z = float(m.o.elements.z[0])
    assert final_z > seeding_kwargs["z"], "Larvae should swim upward toward daytime depth"


@pytest.mark.slow
def test_run_LarvalFish_hatching_fixed_time():
    """Set up and run LarvalFish with fixed-time hatching method."""

    seeding_kwargs = dict(
        lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00", z=-15
    )
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="LarvalFish",
        hatching_method="fixed_time",
        hatch_time_hours=0.99,  # Set to 1 hour for quick test but isn't exact
        hatched=0,  # Start as egg
        do3D=True,
    )
    m.add_reader()
    m.run_all()

    # After 1 hour, the egg should have hatched (hatched=1)
    # The particle should still exist and have moved
    assert len(m.o.elements) > 0
    final_hatched = float(m.o.elements.hatched[0])
    assert final_hatched == 1, "Egg should have hatched after 1 hour"


@pytest.mark.slow
def test_run_LarvalFish_legacy_mode():
    """Set up and run LarvalFish with legacy vertical behavior mode.
    
    Legacy mode preserves the original LarvalFish time-based swimming behavior.
    """

    seeding_kwargs = dict(
        lon=-90, lat=28.7, number=1, start_time="2009-11-19T12:00:00", z=-25
    )
    m = ptm.OpenDriftModel(
        **seeding_kwargs,
        use_static_masks=True,
        duration="1h",
        ocean_model="TXLA",
        ocean_model_local=False,
        drift_model="LarvalFish",
        vertical_behavior_mode="legacy",
        hatched=1,
        stage_fraction=1.0,
        do3D=True,
    )
    m.add_reader()
    m.run_all()

    # Verify simulation completed successfully with legacy mode
    assert len(m.o.elements) > 0
    # With legacy mode, vertical position is governed by time-based swimming
    # rather than target depths

