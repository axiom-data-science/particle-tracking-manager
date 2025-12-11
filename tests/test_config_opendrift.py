from pathlib import Path

import pytest

from pydantic import ValidationError

from particle_tracking_manager.models.opendrift.config_opendrift import (
    HarmfulAlgalBloomModelConfig,
    LarvalFishModelConfig,
    LeewayModelConfig,
    OceanDriftModelConfig,
    OpenDriftConfig,
    OpenOilModelConfig,
)
from particle_tracking_manager.models.opendrift.enums import ObjectTypeEnum
from particle_tracking_manager.models.opendrift.enums.species_types import (
    SPECIES_HAB_DEFAULTS,
)
from particle_tracking_manager.models.opendrift.opendrift import OpenDriftModel


def test_drift_model():
    # this test could be done on any of the drift model classes with the same result
    # i.e. LarvalFishModelConfig, LeewayModelConfig, OpenOilModelConfig, OceanDriftModelConfig
    with pytest.raises(ValidationError):
        m = OpenDriftConfig(drift_model="not_a_real_model")


## LarvalFish ##


def test_LarvalFish_init():
    m = LarvalFishModelConfig(
        drift_model="LarvalFish",
        do3D=True,
        vertical_mixing=True,
        wind_drift_factor=0,
        wind_drift_depth=0,
        steps=1,
        length=10,
    )


def test_LarvalFish_parameters():
    """Make sure LarvalFish-specific parameters are present."""
    m = LarvalFishModelConfig(drift_model="LarvalFish", steps=1)
    params = [
        "diameter",
        "neutral_buoyancy_salinity",
        "stage_fraction",
        "hatched",
        "length",
        "weight",
    ]
    for param in params:
        assert hasattr(m, param)


def test_LarvalFish_disallowed_settings():
    """LarvalFish is incompatible with some settings.

    LarvalFish has to always be 3D with vertical_mixing on.
    """

    with pytest.raises(ValidationError):
        m = LarvalFishModelConfig(
            drift_model="LarvalFish", vertical_mixing=False, steps=1
        )

    with pytest.raises(ValidationError):
        m = LarvalFishModelConfig(drift_model="LarvalFish", do3D=False, steps=1)


def test_LarvalFish_hatched_stage_fraction():
    """If hatched==1, stage_fraction must be None."""

    with pytest.raises(ValidationError):
        m = LarvalFishModelConfig(
            drift_model="LarvalFish", steps=1, hatched=1, stage_fraction=0.5
        )

    m = LarvalFishModelConfig(
        drift_model="LarvalFish", steps=1, hatched=1, stage_fraction=None
    )

    m = LarvalFishModelConfig(
        drift_model="LarvalFish", steps=1, hatched=0, stage_fraction=0.5
    )


## Leeway ##


def test_Leeway_init():
    m = LeewayModelConfig(
        drift_model="Leeway",
        do3D=False,
        steps=1,
    )


def test_Leeway_parameters():
    """Make sure Leeway-specific parameters are present."""
    m = LeewayModelConfig(drift_model="Leeway", steps=1)
    params = ["object_type"]
    for param in params:
        assert hasattr(m, param)


def test_Leeway_disallowed_settings():
    """Leeway is incompatible with some settings.

    Leeway can't have stokes drift or wind drift factor/depth or be 3D
    """

    with pytest.raises(ValidationError):
        m = LeewayModelConfig(drift_model="Leeway", stokes_drift=True, steps=1)

    with pytest.raises(ValidationError):
        m = LeewayModelConfig(
            drift_model="Leeway", wind_drift_factor=10, wind_drift_depth=10, steps=1
        )

    with pytest.raises(ValidationError):
        m = LeewayModelConfig(drift_model="Leeway", do3D=True, steps=1)


@pytest.mark.slow
def test_object_type_list():
    """Make sure options are exactly the same as in OpenDrift."""

    m = OpenDriftModel(drift_model="Leeway", steps=1)
    m.setup_for_simulation()
    od_objects = m.o.get_configspec("seed:object_type")["seed:object_type"]["enum"]

    ptm_objects = [v.value for v in ObjectTypeEnum.__members__.values()]

    assert od_objects == ptm_objects


## OceanDrift ##


def test_OceanDrift_init():
    m = OceanDriftModelConfig(
        drift_model="OceanDrift",
        steps=1,
    )


def test_OceanDrift_parameters():
    """Make sure OceanDrift-specific parameters are present."""
    m = OceanDriftModelConfig(drift_model="OceanDrift", steps=1)
    params = [
        "seed_seafloor",
        "diffusivitymodel",
        "mixed_layer_depth",
        "seafloor_action",
        "wind_drift_depth",
        "vertical_mixing_timestep",
        "wind_drift_factor",
        "vertical_mixing",
    ]
    for param in params:
        assert hasattr(m, param)


## OpenOil ##


def test_OpenOil_init():
    m = OpenOilModelConfig(
        drift_model="OpenOil",
        do3D=False,
        steps=1,
    )


def test_OpenOil_parameters():
    """Make sure OpenOil-specific parameters are present."""
    m = OpenOilModelConfig(drift_model="OpenOil", steps=1)
    params = [
        "oil_type",
        "m3_per_hour",
        "oil_film_thickness",
        "droplet_size_distribution",
        "droplet_diameter_mu",
        "droplet_diameter_sigma",
        "droplet_diameter_min_subsea",
        "droplet_diameter_max_subsea",
        "emulsification",
        "dispersion",
        "evaporation",
        "update_oilfilm_thickness",
        "biodegradation",
    ]
    for param in params:
        assert hasattr(m, param)


def test_OpenOil_json_schema():
    schema = OpenOilModelConfig.model_json_schema()
    assert "{'const': 'AD00010', 'title': 'ABU SAFAH, ARAMCO'}" in map(
        str, schema["properties"]["oil_type"]["oneOf"]
    )


## OceanDrift
def test_unknown_parameter():
    """Make sure unknown parameters are not input."""

    with pytest.raises(ValidationError):
        m = OpenDriftConfig(unknown="test", steps=1, start_time="2022-01-01")


def test_do3D_OceanDrift():
    m = OceanDriftModelConfig(
        steps=1, do3D=True, start_time="2022-01-01", vertical_mixing=True
    )
    m = OceanDriftModelConfig(
        steps=1, do3D=True, start_time="2022-01-01", vertical_mixing=False
    )
    with pytest.raises(ValidationError):
        m = OpenDriftConfig(
            steps=1, do3D=False, start_time="2022-01-01", vertical_mixing=True
        )


def test_z_OceanDrift():
    m = OceanDriftModelConfig(steps=1, start_time="2022-01-01", z=-10)
    assert m.seed_seafloor == False

    m = OceanDriftModelConfig(
        steps=1, start_time="2022-01-01", z=None, seed_seafloor=True
    )

    with pytest.raises(ValidationError):
        m = OceanDriftModelConfig(
            steps=1, start_time="2022-01-01", z=None, seed_seafloor=False
        )

    with pytest.raises(ValidationError):
        m = OceanDriftModelConfig(
            steps=1, start_time="2022-01-01", z=-10, seed_seafloor=True
        )


def test_interpolator_filename():
    with pytest.raises(ValidationError):
        m = OpenDriftConfig(interpolator_filename="test", steps=1, use_cache=False)

    m = OpenDriftConfig(interpolator_filename=None, use_cache=False, steps=1)

    m = OpenDriftConfig(use_cache=True, interpolator_filename="test", steps=1)
    assert m.interpolator_filename == Path("test.pickle")

    m = OpenDriftConfig(use_cache=True, interpolator_filename=None, steps=1)
    assert m.interpolator_filename.name == "CIOFSOP_interpolator.pickle"


## HarmfulAlgalBloom ##


def test_HarmfulAlgalBloom_init():
    m = HarmfulAlgalBloomModelConfig(
        drift_model="HarmfulAlgalBloom",
        steps=1,
    )


def test_HarmfulAlgalBloom_parameters():
    """Make sure HarmfulAlgalBloom-specific parameters are present."""
    m = HarmfulAlgalBloomModelConfig(drift_model="HarmfulAlgalBloom", steps=1)
    params = [
        "species_type",
        "temperature_death_min",
        "temperature_death_max",
        "salinity_death_min",
        "salinity_death_max",
    ]
    for param in params:
        assert hasattr(m, param)


def test_HarmfulAlgalBloom_species_type():
    """Make sure species_type parameter works as expected."""
    m = HarmfulAlgalBloomModelConfig(
        drift_model="HarmfulAlgalBloom",
        species_type="PN",
        steps=1,
    )
    assert m.species_type == "PN"
    assert m.temperature_death_min == SPECIES_HAB_DEFAULTS["PN"].temperature_death_min
    assert m.temperature_death_max == SPECIES_HAB_DEFAULTS["PN"].temperature_death_max
    assert m.salinity_death_min == SPECIES_HAB_DEFAULTS["PN"].salinity_death_min
    assert m.salinity_death_max == SPECIES_HAB_DEFAULTS["PN"].salinity_death_max


def test_HarmfulAlgalBloom_disallowed_settings():
    """HarmfulAlgalBloom is incompatible with some settings depending on the species.

    HarmfulAlgalBloom has to always be 3D with vertical_mixing on.
    """

    with pytest.raises(ValidationError):
        m = HarmfulAlgalBloomModelConfig(
            drift_model="HarmfulAlgalBloom",
            steps=1,
            species_type="custom",
        )

    # with pytest.raises(ValidationError):
    #     m = HarmfulAlgalBloomModelConfig(drift_model="HarmfulAlgalBloom", do3D=True, steps=1, species_type="Pseudo_nitzschia")
