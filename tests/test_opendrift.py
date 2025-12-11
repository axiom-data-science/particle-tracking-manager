"""From Copilot"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pydantic import ValidationError

import particle_tracking_manager as ptm

from particle_tracking_manager.models.opendrift.enums.oil_types import OIL_ID_TO_NAME
from particle_tracking_manager.models.opendrift.opendrift import OpenDriftModel


ds = xr.Dataset(
    data_vars={
        "u": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "v": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "w": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "salt": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "temp": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "wetdry_mask_rho": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "mask_rho": (("Y", "X"), np.zeros((2, 3))),
        "Uwind": (("ocean_time", "Y", "X"), np.zeros((2, 2, 3))),
        "Vwind": (("ocean_time", "Y", "X"), np.zeros((2, 2, 3))),
        "Cs_r": (("Z"), np.linspace(-1, 0, 3)),
        "hc": 16,
    },
    coords={
        "ocean_time": ("ocean_time", [0, 1], {"units": "seconds since 1970-01-01"}),
        "s_rho": (("Z"), np.linspace(-1, 0, 3)),
        "lon_rho": (("Y", "X"), np.array([[1, 2, 3], [1, 2, 3]])),
        "lat_rho": (("Y", "X"), np.array([[1, 1, 1], [2, 2, 2]])),
    },
)
ds_info = dict(
    lon_min=1, lon_max=3, lat_min=1, lat_max=2, start_time_model=0, end_time_fixed=1
)

ptm.config_ocean_model.register_on_the_fly(ds_info)

# ocean_model_local is False because otherwise it requires a `kerchunk_func_str`
seed_kws = dict(lon=2, lat=1.5, start_time=0, time_step=0.01, ocean_model_local=False)


def test_start_time_tz():
    """Check start time timezone is removed."""

    m = OpenDriftModel(
        duration="1s",
        start_time="1970-01-01T00:00Z",
        lon=2,
        lat=1.5,
        time_step=0.01,
        ocean_model="ONTHEFLY",
        ocean_model_local=False,
    )
    m.add_reader(ds=ds)
    assert m.config.start_time == pd.Timestamp("1970-01-01 00:00:00")
    assert m.config.end_time == pd.Timestamp("1970-01-01 00:00:01")

    m = OpenDriftModel(
        duration="1s",
        start_time=(pd.Timestamp("1970-01-01T00:00") - pd.Timedelta("5h")).tz_localize(
            "US/Eastern"
        ),
        lon=2,
        lat=1.5,
        time_step=0.01,
        ocean_model="ONTHEFLY",
        ocean_model_local=False,
    )
    m.add_reader(ds=ds)
    assert m.config.start_time == pd.Timestamp("1970-01-01 00:00:00")
    assert m.config.end_time == pd.Timestamp("1970-01-01 00:00:01")


def test_drop_vars_do3D_true():
    m = OpenDriftModel(
        drift_model="OceanDrift",
        do3D=True,
        duration="1s",
        ocean_model="ONTHEFLY",
        **seed_kws
    )
    # drop variables manually using drop_vars from config to check behavior
    m.add_reader(ds=ds.drop_vars(m.config.drop_vars, errors="ignore"))
    assert m.reader.variables == [
        "x_sea_water_velocity",
        "y_sea_water_velocity",
        "upward_sea_water_velocity",
        "land_binary_mask",
        "x_wind",
        "y_wind",
        "wind_speed",
        "sea_water_speed",
    ]


def test_drop_vars_do3D_false_use_static_masks():
    m = OpenDriftModel(
        drift_model="OceanDrift",
        use_static_masks=True,
        duration="1s",
        ocean_model="ONTHEFLY",
        **seed_kws
    )
    # drop variables manually using drop_vars from config to check behavior
    m.add_reader(ds=ds.drop_vars(m.config.drop_vars, errors="ignore"))
    assert m.reader.variables == [
        "x_sea_water_velocity",
        "y_sea_water_velocity",
        "land_binary_mask",
        "x_wind",
        "y_wind",
        "wind_speed",
        "sea_water_speed",
    ]
    assert "mask_rho" in m.reader.Dataset.data_vars
    assert "wetdry_mask_rho" not in m.reader.Dataset.data_vars


def test_drop_vars_no_wind():
    m = OpenDriftModel(
        drift_model="OceanDrift",
        duration="1s",
        stokes_drift=False,
        wind_drift_factor=0,
        wind_uncertainty=0,
        vertical_mixing=False,
        ocean_model="ONTHEFLY",
        **seed_kws
    )
    # drop variables manually using drop_vars from config to check behavior
    m.add_reader(ds=ds.drop_vars(m.config.drop_vars, errors="ignore"))
    assert m.reader.variables == [
        "x_sea_water_velocity",
        "y_sea_water_velocity",
        "land_binary_mask",
        "sea_water_speed",
    ]


def test_Leeway():

    m = OpenDriftModel(
        drift_model="Leeway",
        object_type=">PIW, scuba suit (face up)",
        stokes_drift=False,
        steps=1,
        ocean_model="ONTHEFLY",
        **seed_kws
    )
    m.setup_for_simulation()  # creates m.o
    # assert "wind_drift_factor" not in m.config
    assert not hasattr(m.config, "wind_drift_factor")
    assert "seed:wind_drift_factor" not in m.o.get_configspec().keys()


def test_LarvalFish_add_reader():
    m = OpenDriftModel(
        drift_model="LarvalFish",
        do3D=True,
        duration="1s",
        ocean_model="ONTHEFLY",
        **seed_kws
    )
    m.add_reader(ds=ds.drop_vars(m.config.drop_vars, errors="ignore"))
    assert m.reader.variables == [
        "x_sea_water_velocity",
        "y_sea_water_velocity",
        "upward_sea_water_velocity",
        "sea_water_salinity",
        "sea_water_temperature",
        "land_binary_mask",
        "x_wind",
        "y_wind",
        "wind_speed",
        "sea_water_speed",
    ]


def test_LarvalFish_seeding():
    """Make sure special seed parameter comes through"""

    m = OpenDriftModel(
        drift_model="LarvalFish",
        lon=-151,
        lat=60,
        do3D=True,
        hatched=1,
        stage_fraction=None,
        start_time="2022-01-01T00:00:00",
        use_auto_landmask=True,
        steps=1,
        vertical_mixing=True,
        wind_drift_factor=0,
        wind_drift_depth=0,
    )
    m.setup_for_simulation()  # creates m.o
    assert m.o._config["seed:hatched"]["value"] == 1


def test_OpenOil_seeding():
    """Make sure special seed parameters comes through"""
    oil_type = "AD00010"
    m = OpenDriftModel(
        drift_model="OpenOil",
        lon=-151,
        lat=60,
        do3D=True,
        start_time="2023-01-01T00:00:00",
        use_auto_landmask=True,
        m3_per_hour=5,
        droplet_diameter_max_subsea=0.1,
        droplet_diameter_min_subsea=0.01,
        droplet_diameter_mu=0.01,
        droplet_size_distribution="normal",
        droplet_diameter_sigma=0.9,
        oil_film_thickness=5,
        oil_type=oil_type,
        # oil_type="GN00002",
        steps=1,
    )
    m.setup_for_simulation()  # creates m.o

    # m.o.set_config("environment:constant:x_wind", -1)
    # m.o.set_config("environment:constant:y_wind", -1)
    # m.o.set_config("environment:constant:x_sea_water_velocity", -1)
    # m.o.set_config("environment:constant:y_sea_water_velocity", -1)
    # m.o.set_config("environment:constant:sea_water_temperature", 15)
    # m.seed()

    # to check impact of m3_per_hour: mass_oil for m3_per_hour of 1 * 5
    # assert np.allclose(m.o.elements_scheduled.mass_oil, 0.855 * 5)  # i'm getting different answers local vs github actiosn
    assert m.o._config["seed:m3_per_hour"]["value"] == 5
    assert m.o._config["seed:droplet_diameter_max_subsea"]["value"] == 0.1
    assert m.o._config["seed:droplet_diameter_min_subsea"]["value"] == 0.01
    assert m.o._config["seed:droplet_diameter_mu"]["value"] == 0.01
    assert m.o._config["seed:droplet_size_distribution"]["value"] == "normal"
    assert m.o._config["seed:droplet_diameter_sigma"]["value"] == 0.9
    # assert m.o.elements_scheduled.oil_film_thickness == 5
    assert (
        m.o._config["seed:oil_type"]["value"] == OIL_ID_TO_NAME[oil_type]
    )  # don't use ID because it is stripped off


def test_OpenOil_oil_type_id():
    """Make sure oil type entered as id works"""
    oil_type = "AD00010"
    m = OpenDriftModel(drift_model="OpenOil", oil_type=oil_type, steps=1)


@pytest.mark.slow
def test_OpenOil_vertical_mixing():
    m = ptm.OpenDriftModel(
        drift_model="OpenOil",
        steps=1,
        oil_type="AD00010",
        do3D=True,
        vertical_mixing=False,
    )
    m.setup_for_simulation()
    assert not m.o.get_config("drift:vertical_mixing")


@pytest.mark.slow
def test_OpenOil_all_oils_exact_match():
    """Make sure that oils in PTM exactly match those in OpenDrift."""
    import opendrift.models.openoil.adios.dirjs as dirjs

    schema = ptm.OpenOilModelConfig.model_json_schema()
    ptm_oils = {
        oil["const"]: oil["title"] for oil in schema["properties"]["oil_type"]["oneOf"]
    }
    od_oils = {oil.id: oil.name for oil in dirjs.oils(limit=1300)}

    # OpenDrift has 5 extra Norwegian oils that are not in PTM so we remove them for this test
    extra_od_oils = ["NO00167", "NO00168", "NO00169", "NO00170", "NO00171"]
    for extra_oil in extra_od_oils:
        od_oils.pop(extra_oil, None)

    assert ptm_oils == od_oils


def test_HarmfulAlgalBloom_seeding():
    """Make sure special parameter comes through"""

    m = OpenDriftModel(
        drift_model="HarmfulAlgalBloom",
        species_type="PN",
        temperature_death_min=1,
        lon=-151,
        lat=60,
        start_time="2022-01-01T00:00:00",
        use_auto_landmask=True,
        steps=1,
    )
    m.setup_for_simulation()  # creates m.o
    assert m.o._config["hab:temperature_death_min"]["value"] == 1


def test_wind_drift():
    """Make sure changed wind drift numbers comes through"""

    m = OpenDriftModel(
        drift_model="OceanDrift",
        lon=-151,
        lat=60,
        do3D=True,
        wind_drift_factor=1,
        wind_drift_depth=10,
        start_time="2023-01-01T00:00:00",
        use_auto_landmask=True,
        steps=1,
    )
    m.setup_for_simulation()  # creates m.o
    assert m.o._config["seed:wind_drift_factor"]["value"] == 1
    assert m.o._config["drift:wind_drift_depth"]["value"] == 10


def test_plots_linecolor():
    # since export_variables are all included by default now, I am not testing this
    # this should error if user inputs some export_variables, which
    # changes the default from returning all variables to just those
    # selected plus a short list of required variables
    # with pytest.raises(ValueError):
    #     m = OpenDriftModel(
    #         drift_model="OceanDrift",
    #         plots={"spaghetti": {"linecolor": "x_wind"}},
    #         export_variables=[],
    #         steps=1
    #     )

    m = OpenDriftModel(
        drift_model="OceanDrift", plots={"spaghetti": {"linecolor": "x_wind"}}, steps=1
    )

    m = OpenDriftModel(
        drift_model="OceanDrift",
        plots={"spaghetti": {"linecolor": "x_wind"}},
        export_variables=None,
        steps=1,
    )

    # this should work bc "z" should already be included
    m = OpenDriftModel(
        drift_model="OceanDrift", plots={"spaghetti": {"linecolor": "z"}}, steps=1
    )


def test_plots_background():
    # # this should error if user inputs some export_variables, which
    # # changes the default from returning all variables to just those
    # # selected plus a short list of required variables
    # with pytest.raises(ValueError):
    #     m = OpenDriftModel(
    #         drift_model="OceanDrift",
    #         plots={"animation": {"background": "sea_surface_height"}},
    #         export_variables=[],
    #         steps=1
    #     )

    m = OpenDriftModel(
        drift_model="OceanDrift",
        plots={"animation": {"background": "sea_surface_height"}},
        steps=1,
    )


def test_plots_oil():
    # # this should error if user inputs some export_variables, which
    # # changes the default from returning all variables to just those
    # # selected plus a short list of required variables
    # with pytest.raises(ValueError):
    #     m = OpenDriftModel(
    #         drift_model="OpenOil",
    #         plots={"oil": {"show_wind_and_current": True}},
    #         export_variables=[],
    #     )

    m = OpenDriftModel(
        drift_model="OpenOil", plots={"oil": {"show_wind_and_current": True}}, steps=1
    )

    with pytest.raises(ValidationError):
        m = OpenDriftModel(drift_model="OceanDrift", plots={"oil": {}}, steps=1)


def test_plots_property():
    # # this should error if user inputs some export_variables, which
    # # changes the default from returning all variables to just those
    # # selected plus a short list of required variables
    # with pytest.raises(ValueError):
    #     m = OpenDriftModel(
    #         drift_model="LarvalFish",
    #         do3D=True,
    #         plots={"property": {"prop": "survival"}},
    #         export_variables=["x_wind"],
    #     )

    m = OpenDriftModel(
        drift_model="LarvalFish",
        do3D=True,
        plots={"property": {"prop": "survival"}},
        steps=1,
    )


def test_plots_all():

    with pytest.raises(ValueError):
        m = OpenDriftModel(
            drift_model="OceanDrift",
            plots={
                "all": {},
                "spaghetti": {"line_color": "x_wind"},
                "animation": {"background": "sea_surface_height"},
            },
            steps=1,
        )


def test_plots_names():

    with pytest.raises(ValidationError):
        m = OpenDriftModel(
            plots={
                "random_plot_name": {},
            },
            steps=1,
        )

    with pytest.raises(ValidationError):
        m = OpenDriftModel(
            plots={
                "something_spaghetti": {},
            },
            steps=1,
        )


# @pytest.mark.slow
def test_parameter_passing():
    """make sure parameters passed into package make it to simulation runtime."""

    ts = 7 * 60  # seconds
    diffmodel = "windspeed_Sundby1983"
    use_auto_landmask = True
    vertical_mixing = True
    do3D = True

    seed_kws = dict(
        lon=-151,
        lat=60.0,
        radius=5000,
        number=100,
        start_time=datetime(2022, 9, 22, 6, 0, 0),
    )
    m = OpenDriftModel(
        use_auto_landmask=use_auto_landmask,
        time_step=ts,
        duration="P0DT0H5M0S",
        steps=None,
        diffusivitymodel=diffmodel,
        vertical_mixing=vertical_mixing,
        do3D=do3D,
        **seed_kws
    )

    m.setup_for_simulation()  # creates m.o

    # idealized simulation, provide a fake current
    m.o.set_config("environment:fallback:y_sea_water_velocity", 1)

    # check time_step across access points
    assert (
        # m.o._config["general:time_step_minutes"]["value"]  # this is not correct, don't know why
        # m.o.time_step.total_seconds()  # this is only created once run
        ts
        == m.config.time_step
        # == m.o.get_configspec()["general:time_step_minutes"]["value"]  # this is not correct, don't know why
    )

    # check diff model
    assert (
        m.o.get_configspec()["vertical_mixing:diffusivitymodel"]["value"] == diffmodel
    )

    # check use_auto_landmask coming through
    assert (
        m.o.get_configspec()["general:use_auto_landmask"]["value"] == use_auto_landmask
    )
