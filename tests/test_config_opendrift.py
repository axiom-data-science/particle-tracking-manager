from pathlib import Path

import pytest

from pydantic import ValidationError

from particle_tracking_manager.models.opendrift.config_opendrift import (
    LarvalFishModelConfig,
    LeewayModelConfig,
    OceanDriftModelConfig,
    OpenDriftConfig,
    OpenOilModelConfig,
    PhytoplanktonModelConfig,
)
from particle_tracking_manager.models.opendrift.enums import (
    DriftModelEnum,
    HatchingMethodEnum,
    ObjectTypeEnum,
    ParticleTypeEnum,
    VerticalBehaviorModeEnum,
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


# def test_LarvalFish_disallowed_settings():
#     """LarvalFish is incompatible with some settings.

#     LarvalFish has to always be 3D with vertical_mixing on.
#     """

#     with pytest.raises(ValidationError):
#         m = LarvalFishModelConfig(
#             drift_model="LarvalFish", vertical_mixing=False, steps=1
#         )

#     with pytest.raises(ValidationError):
#         m = LarvalFishModelConfig(drift_model="LarvalFish", do3D=False, steps=1)


def test_LarvalFish_hatched_stage_fraction():
    """If hatched==1, stage_fraction must be a number but will be ignored."""

    with pytest.raises(ValidationError):
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


def test_OceanDrift_wind_drift():
    m = OceanDriftModelConfig(
        drift_model="OceanDrift",
        steps=1,
        wind_drift=False,
    )

    assert m.wind_drift_factor == 0

    m = OceanDriftModelConfig(
        drift_model="OceanDrift",
        steps=1,
        wind_drift=True,
    )

    assert m.wind_drift_factor == 0.02


def test_do3D_vertical_mixing_False():
    """If do3D is False, vertical_mixing should be set to False."""

    # OceanDrift
    m = OceanDriftModelConfig(
        steps=1, do3D=False, start_time="2022-01-01", vertical_mixing=True
    )
    assert m.vertical_mixing == False

    # OpenOil
    m = OpenOilModelConfig(
        steps=1, do3D=False, start_time="2022-01-01", vertical_mixing=True
    )
    assert m.vertical_mixing == False

    # Phytoplankton
    m = PhytoplanktonModelConfig(
        steps=1, do3D=False, start_time="2022-01-01", vertical_mixing=True
    )
    assert m.vertical_mixing == False


def test_vertical_advection_surface():
    """If do3D is False, vertical_advection_at_surface should be set to False

    and if True, vertical_advection_at_surface should be set to True."""

    # OceanDrift
    m = OceanDriftModelConfig(
        steps=1,
        do3D=False,
        start_time="2022-01-01",
        vertical_advection_at_surface=True,
    )
    assert m.vertical_advection_at_surface == False

    m = OceanDriftModelConfig(
        steps=1,
        do3D=True,
        start_time="2022-01-01",
        vertical_advection_at_surface=False,
    )
    assert m.vertical_advection_at_surface == True

    # OpenOil
    m = OpenOilModelConfig(
        steps=1,
        do3D=False,
        start_time="2022-01-01",
        vertical_advection_at_surface=True,
    )
    assert m.vertical_advection_at_surface == False

    m = OpenOilModelConfig(
        steps=1,
        do3D=True,
        start_time="2022-01-01",
        vertical_advection_at_surface=False,
    )
    assert m.vertical_advection_at_surface == True

    # Phytoplankton
    m = PhytoplanktonModelConfig(
        steps=1,
        do3D=False,
        start_time="2022-01-01",
        vertical_advection_at_surface=True,
    )
    assert m.vertical_advection_at_surface == False

    m = PhytoplanktonModelConfig(
        steps=1,
        do3D=True,
        start_time="2022-01-01",
        vertical_advection_at_surface=False,
    )
    assert m.vertical_advection_at_surface == True

    m = LarvalFishModelConfig(
        steps=1,
        do3D=True,
        start_time="2022-01-01",
        vertical_advection_at_surface=False,
    )
    assert m.vertical_advection_at_surface == True


def test_vertical_mixing_surface():
    """If vertical_mixing is True, vertical_mixing_at_surface should be True

    and if False, vertical_mixing_at_surface should be False."""

    # OceanDrift
    m = OceanDriftModelConfig(
        steps=1,
        vertical_mixing=True,
        start_time="2022-01-01",
        vertical_mixing_at_surface=False,
        do3D=True,
    )
    assert m.vertical_mixing_at_surface == True

    m = OceanDriftModelConfig(
        steps=1,
        vertical_mixing=False,
        start_time="2022-01-01",
        vertical_mixing_at_surface=True,
    )
    assert m.vertical_mixing_at_surface == False

    # OpenOil
    m = OpenOilModelConfig(
        steps=1,
        vertical_mixing=True,
        start_time="2022-01-01",
        vertical_mixing_at_surface=False,
        do3D=True,
    )
    assert m.vertical_mixing_at_surface == True

    m = OpenOilModelConfig(
        steps=1,
        vertical_mixing=False,
        start_time="2022-01-01",
        vertical_mixing_at_surface=True,
    )
    assert m.vertical_mixing_at_surface == False

    # Phytoplankton
    m = PhytoplanktonModelConfig(
        steps=1,
        vertical_mixing=True,
        start_time="2022-01-01",
        vertical_mixing_at_surface=False,
        do3D=True,
    )
    assert m.vertical_mixing_at_surface == True

    m = PhytoplanktonModelConfig(
        steps=1,
        vertical_mixing=False,
        start_time="2022-01-01",
        vertical_mixing_at_surface=True,
    )
    assert m.vertical_mixing_at_surface == False

    # LarvalFish
    m = LarvalFishModelConfig(
        steps=1,
        vertical_mixing=True,
        start_time="2022-01-01",
        vertical_mixing_at_surface=False,
        do3D=True,
    )
    assert m.vertical_mixing_at_surface == True

    m = LarvalFishModelConfig(
        steps=1,
        vertical_mixing=False,
        start_time="2022-01-01",
        vertical_mixing_at_surface=True,
    )
    assert m.vertical_mixing_at_surface == False


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

    with pytest.raises(ValidationError):
        m = OceanDriftModelConfig(
            steps=1, start_time="2022-01-01", z=None, seed_seafloor=True
        )

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


## Phytoplankton ##


def test_Phytoplankton_init():
    """Test PhytoplanktonModelConfig instantiation."""
    m = PhytoplanktonModelConfig(
        drift_model="Phytoplankton",
        steps=1,
    )
    assert m.drift_model == DriftModelEnum.Phytoplankton


def test_Phytoplankton_defaults():
    """Test PhytoplanktonModelConfig default values."""
    m = PhytoplanktonModelConfig(
        drift_model="Phytoplankton",
        steps=1,
    )
    assert m.vertical_behavior_mode == VerticalBehaviorModeEnum.dvm
    assert m.w_active == 0.001
    assert m.z_pref == -10.0
    assert m.do3D == True
    assert m.vertical_mixing == True


def test_Phytoplankton_dvm_mode():
    """Test PhytoplanktonModelConfig with DVM mode."""
    m = PhytoplanktonModelConfig(
        drift_model="Phytoplankton",
        steps=1,
        vertical_behavior_mode=VerticalBehaviorModeEnum.dvm,
        z_day=-25.0,
        z_night=-5.0,
        w_active=0.002,
    )
    assert m.vertical_behavior_mode == VerticalBehaviorModeEnum.dvm
    assert m.z_day == -25.0
    assert m.z_night == -5.0


# def test_Phytoplankton_disallowed_settings():
#     """Phytoplankton requires 3D with vertical_mixing."""
#     with pytest.raises(ValidationError):
#         m = PhytoplanktonModelConfig(
#             drift_model="Phytoplankton",
#             vertical_mixing=False,
#             steps=1,
#         )

#     with pytest.raises(ValidationError):
#         m = PhytoplanktonModelConfig(
#             drift_model="Phytoplankton",
#             do3D=False,
#             steps=1,
#         )


## Enums ##


def test_vertical_behavior_mode_enum():
    """Test VerticalBehaviorModeEnum values."""
    assert VerticalBehaviorModeEnum.none == "none"
    assert VerticalBehaviorModeEnum.depth == "depth"
    assert VerticalBehaviorModeEnum.dvm == "dvm"
    assert VerticalBehaviorModeEnum.legacy == "legacy"


def test_hatching_method_enum():
    """Test HatchingMethodEnum values."""
    assert HatchingMethodEnum.temperature == "temperature"
    assert HatchingMethodEnum.fixed_time == "fixed_time"


def test_particle_type_enum():
    """Test ParticleTypeEnum values."""
    assert ParticleTypeEnum.larva == "larva"
    assert ParticleTypeEnum.phytoplankton == "phytoplankton"


## LarvalFish with new features ##


def test_LarvalFish_vertical_behavior_legacy():
    """Test LarvalFishModelConfig with legacy vertical behavior mode."""
    m = LarvalFishModelConfig(
        drift_model="LarvalFish",
        steps=1,
        vertical_behavior_mode=VerticalBehaviorModeEnum.legacy,
        hatching_method=HatchingMethodEnum.temperature,
    )
    assert m.vertical_behavior_mode == VerticalBehaviorModeEnum.legacy
    assert m.hatching_method == HatchingMethodEnum.temperature


def test_LarvalFish_vertical_behavior_depth():
    """Test LarvalFishModelConfig with depth vertical behavior mode."""
    m = LarvalFishModelConfig(
        drift_model="LarvalFish",
        steps=1,
        vertical_behavior_mode=VerticalBehaviorModeEnum.depth,
        z_pref=-15.0,
        w_active=0.003,
    )
    assert m.vertical_behavior_mode == VerticalBehaviorModeEnum.depth
    assert m.z_pref == -15.0


def test_LarvalFish_fixed_time_hatching():
    """Test LarvalFishModelConfig with fixed-time hatching."""
    m = LarvalFishModelConfig(
        drift_model="LarvalFish",
        steps=1,
        hatching_method=HatchingMethodEnum.fixed_time,
        hatch_time_days=3.0,
    )
    assert m.hatching_method == HatchingMethodEnum.fixed_time
    assert m.hatch_time_days == 3.0


def test_LarvalFish_dvm_mode():
    """Test LarvalFishModelConfig with DVM mode."""
    m = LarvalFishModelConfig(
        drift_model="LarvalFish",
        steps=1,
        vertical_behavior_mode=VerticalBehaviorModeEnum.dvm,
        z_day=-20.0,
        z_night=-8.0,
    )
    assert m.vertical_behavior_mode == VerticalBehaviorModeEnum.dvm
    assert m.z_day == -20.0
    assert m.z_night == -8.0


def test_od_mapping_present():
    """Test that new parameters have od_mapping configured."""
    m = PhytoplanktonModelConfig(
        drift_model="Phytoplankton",
        steps=1,
    )

    fields_with_mapping = [
        "vertical_behavior_mode",
        "w_active",
        "z_pref",
        "z_day",
        "z_night",
        "dz_min",
        "dz_rel",
        "dz_max",
    ]

    for field_name in fields_with_mapping:
        field = m.model_fields[field_name]
        assert (
            "od_mapping" in field.json_schema_extra
        ), f"{field_name} missing od_mapping"
