"""OpenDrift HAB species types enum definition."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class HABSpeciesTypeEnum(str, Enum):
    """Harmful Algal Bloom species types supported by OpenDrift."""

    PN = "PN"
    AX = "AX"
    DP = "DP"
    custom = "custom"


HAB_SPECIES_LABELS = {
    HABSpeciesTypeEnum.PN: "Pseudo nitzschia",
    HABSpeciesTypeEnum.AX: "Alexandrium",
    HABSpeciesTypeEnum.DP: "Dinophysis",
    HABSpeciesTypeEnum.custom: "Custom species (manual parameters)",
}


class VerticalBehaviorEnum(str, Enum):
    """Enum for vertical behavior types."""

    none = "none"
    band = "band"
    diel_band = "diel_band"


class HABParameters(BaseModel):
    """Harmful Algal Bloom species parameters for OpenDrift."""

    model_config = ConfigDict(extra="forbid")

    temperature_death_min: float = Field(
        description="Minimum temperature for living. Below this temperature, cells have high mortality rate. Between this and temperature_viable_min, cells have no growth.",
        title="Cell death below this temperature",
        ge=-5.0,
        le=40.0,
        default=3.0,
        json_schema_extra={
            "units": "degrees",
            "od_mapping": "hab:temperature_death_min",
            "ptm_level": 2,
        },
    )

    temperature_death_max: float = Field(
        description="Maximum temperature for living. Above this temperature, cells have high mortality rate. Between temperature_viable_max and this parameter, cells have no growth.",
        title="Cell death above this temperature",
        ge=-5.0,
        le=40.0,
        default=22.0,
        json_schema_extra={
            "units": "degrees",
            "od_mapping": "hab:temperature_death_max",
            "ptm_level": 2,
        },
    )

    temperature_pref_min: float = Field(
        description="Minimum temperature for preferred temperature range; cells have regular growth.",
        title="Minimum preferred temperature",
        ge=-5.0,
        le=40.0,
        default=10.0,
        json_schema_extra={
            "units": "degrees",
            "od_mapping": "hab:temperature_pref_min",
            "ptm_level": 2,
        },
    )

    temperature_pref_max: float = Field(
        description="Maximum temperature for preferred temperature range; cells have regular growth.",
        title="Maximum preferred temperature",
        ge=-5.0,
        le=40.0,
        default=20.0,
        json_schema_extra={
            "units": "degrees",
            "od_mapping": "hab:temperature_pref_max",
            "ptm_level": 2,
        },
    )

    salinity_death_min: float = Field(
        description="Minimum salinity for living. Below this salinity, cells have high mortality rate. Between this and salinity_viable_min, cells have no growth.",
        title="Cell death below this salinity",
        ge=0.0,
        le=50.0,
        default=25.0,
        json_schema_extra={
            "units": "psu",
            "od_mapping": "hab:salinity_death_min",
            "ptm_level": 2,
        },
    )

    salinity_death_max: float = Field(
        description="Maximum salinity for living. Above this salinity, cells have high mortality rate. Between salinity_viable_max and this parameter, cells have no growth.",
        title="Cell death above this salinity",
        ge=0.0,
        le=50.0,
        default=36.0,
        json_schema_extra={
            "units": "psu",
            "od_mapping": "hab:salinity_death_max",
            "ptm_level": 2,
        },
    )

    salinity_pref_min: float = Field(
        description="Minimum salinity for preferred salinity range; cells have regular growth.",
        title="Minimum preferred salinity",
        ge=0.0,
        le=50.0,
        default=30.0,
        json_schema_extra={
            "units": "psu",
            "od_mapping": "hab:salinity_pref_min",
            "ptm_level": 2,
        },
    )

    salinity_pref_max: float = Field(
        description="Maximum salinity for preferred salinity range; cells have regular growth.",
        title="Maximum preferred salinity",
        ge=0.0,
        le=50.0,
        default=40.0,
        json_schema_extra={
            "units": "psu",
            "od_mapping": "hab:salinity_pref_max",
            "ptm_level": 2,
        },
    )

    mortality_rate_high: float = Field(
        description="Rate of mortality applied below temperature_death_min and above temperature_death_max.",
        title="High mortality rate",
        ge=0.0,
        le=10.0,
        default=1.0,
        json_schema_extra={
            "units": "days^-1",
            "od_mapping": "hab:mortality_rate_high",
            "ptm_level": 2,
        },
    )

    mortality_rate_medium: float = Field(
        description="Rate of mortality applied in edge conditions defined by temperature_death_min, temperature_death_max, salinity_death_min, salinity_death_max, temperature_pref_min, temperature_pref_max, salinity_pref_min, and salinity_pref_max.",
        title="Medium mortality rate",
        ge=0.0,
        le=10.0,
        default=0.5,
        json_schema_extra={
            "units": "days^-1",
            "od_mapping": "hab:mortality_rate_medium",
            "ptm_level": 2,
        },
    )

    mortality_rate_low: float = Field(
        description="Rate of mortality applied in preferred conditions defined by temperature_pref_min, temperature_pref_max, salinity_pref_min, and salinity_pref_max.",
        title="Low mortality rate",
        ge=0.0,
        le=10.0,
        default=0.1,
        json_schema_extra={
            "units": "days^-1",
            "od_mapping": "hab:mortality_rate_low",
            "ptm_level": 2,
        },
    )

    growth_rate_high: float = Field(
        description="Rate of growth applied in preferred conditions defined by temperature_pref_min, temperature_pref_max, salinity_pref_min, and salinity_pref_max.",
        title="High growth rate",
        ge=0.0,
        le=10.0,
        default=0.7,
        json_schema_extra={
            "units": "days^-1",
            "od_mapping": "hab:growth_rate_high",
            "ptm_level": 2,
        },
    )

    growth_rate_medium: float = Field(
        description="Rate of growth applied in edge conditions defined by temperature_death_min, temperature_death_max, salinity_death_min, salinity_death_max, temperature_pref_min, temperature_pref_max, salinity_pref_min, and salinity_pref_max.",
        title="Medium growth rate",
        ge=0.0,
        le=10.0,
        default=0.2,
        json_schema_extra={
            "units": "days^-1",
            "od_mapping": "hab:growth_rate_medium",
            "ptm_level": 2,
        },
    )

    growth_rate_low: float = Field(
        description="Rate of growth applied below viable conditions defined by temperature_death_min, temperature_death_max, salinity_death_min, and salinity_death_max.",
        title="Low growth rate",
        ge=0.0,
        le=10.0,
        default=0.0,
        json_schema_extra={
            "units": "days^-1",
            "od_mapping": "hab:growth_rate_low",
            "ptm_level": 2,
        },
    )

    vertical_behavior: VerticalBehaviorEnum = Field(
        description="Vertical behavior: no active movement, fixed band, or diel band migration.",
        title="Vertical behavior",
        default="none",
        json_schema_extra={
            "od_mapping": "hab:vertical_behavior",
            "ptm_level": 2,
        },
    )

    swim_speed: float = Field(
        description="Maximum active vertical swimming speed (m/s).",
        title="Swim speed",
        ge=0.0,
        le=100.0,
        default=0.001,
        json_schema_extra={
            "units": "m/s",
            "od_mapping": "hab:swim_speed",
            "ptm_level": 2,
        },
    )

    band_center_depth: float = Field(
        description="Target center of preferred depth band (m, negative down).",
        title="Band center depth",
        ge=-10000.0,
        le=0.0,
        default=-10.0,
        json_schema_extra={
            "units": "m",
            "od_mapping": "hab:band_center_depth",
            "ptm_level": 2,
        },
    )

    band_half_width: float = Field(
        description="Half-width of preferred depth band (m).",
        title="Band half-width",
        ge=0.0,
        le=5000.0,
        default=5.0,
        json_schema_extra={
            "units": "m",
            "od_mapping": "hab:band_half_width",
            "ptm_level": 2,
        },
    )

    diel_day_depth: float = Field(
        description="Target depth during daytime (m, negative).",
        title="Diel day depth",
        ge=-10000.0,
        le=0.0,
        default=-20.0,
        json_schema_extra={
            "units": "m",
            "od_mapping": "hab:diel_day_depth",
            "ptm_level": 2,
        },
    )

    diel_night_depth: float = Field(
        description="Target depth during nighttime (m, negative).",
        title="Diel night depth",
        ge=-10000.0,
        le=0.0,
        default=-5.0,
        json_schema_extra={
            "units": "m",
            "od_mapping": "hab:diel_night_depth",
            "ptm_level": 2,
        },
    )


# Species default for HarmfulAlgalBloom model
SPECIES_HAB_DEFAULTS: dict[HABSpeciesTypeEnum, HABParameters] = {
    HABSpeciesTypeEnum.PN: HABParameters(
        temperature_death_min=2.0,
        temperature_death_max=22.0,
        temperature_pref_min=10.0,
        temperature_pref_max=18.0,
        salinity_death_min=22.0,
        salinity_death_max=36.0,
        salinity_pref_min=29.0,
        salinity_pref_max=34.0,
        mortality_rate_high=1.0,
        mortality_rate_medium=0.5,
        mortality_rate_low=0.1,
        growth_rate_high=0.8,  # day⁻¹  (within 0.4–1.04 observed)
        growth_rate_medium=0.3,  # day⁻¹
        growth_rate_low=0.0,
        vertical_behavior="none",
        band_center_depth=-5.0,  # m
        band_half_width=10.0,  # m   (very broad, not tight band)
        swim_speed=0.0,
    ),
    HABSpeciesTypeEnum.AX: HABParameters(
        temperature_death_min=5.0,
        temperature_death_max=22.0,
        temperature_pref_min=8.0,
        temperature_pref_max=14.0,
        salinity_death_min=15.0,
        salinity_death_max=35.0,
        salinity_pref_min=25.0,
        salinity_pref_max=33.0,
        mortality_rate_high=1.0,
        mortality_rate_medium=0.5,
        mortality_rate_low=0.1,
        growth_rate_high=0.65,  # day⁻¹  (near 0.63 observed)
        growth_rate_medium=0.25,  # day⁻¹
        growth_rate_low=0.0,
        # Alexandrium defaults
        vertical_behavior="diel_band",
        swim_speed=0.0005,  # m/s
        # MLD 20 like in Lower Cook Inlet
        diel_day_depth=-6.0,  # m
        diel_night_depth=-26.0,  # m, this should be below the MLD
        # # MLD 12 like in KB
        # diel_day_depth=-4.0,       # m
        # diel_night_depth=-18.0,      # m
        # # If you ever use pure band: (not used for AX)
        # band_center_depth=-10.0,
        band_half_width=5.0,
    ),
    HABSpeciesTypeEnum.DP: HABParameters(
        temperature_death_min=6.0,
        temperature_death_max=22.0,
        temperature_pref_min=10.0,
        temperature_pref_max=16.0,
        salinity_death_min=20.0,
        salinity_death_max=36.0,
        salinity_pref_min=28.0,
        salinity_pref_max=34.0,
        mortality_rate_high=1.0,
        mortality_rate_medium=0.5,
        mortality_rate_low=0.1,
        growth_rate_high=0.4,  # day⁻¹  (near 0.36–0.39 observed)
        growth_rate_medium=0.15,  # day⁻¹
        growth_rate_low=0.0,
        vertical_behavior="band",
        swim_speed=0.0003,  # m/s (≈ 26 m/day max vertical travel)
        # # Day: somewhat deeper, but not as deep as Alexandrium
        # diel_day_depth=-15.0,
        # # Night: a bit shallower, but not right at the surface
        # diel_night_depth=-5.0,
        # Band settings:
        band_center_depth=-12.5,
        band_half_width=7.5,
    ),
    # HABSpeciesTypeEnum.custom intentionally has no entry
}


# Other config defaults per species (z, do3D, etc.)
SPECIES_HAB_MANAGER_DEFAULTS: dict[HABSpeciesTypeEnum, dict[str, object]] = {
    HABSpeciesTypeEnum.PN: {"z": -1.0, "do3D": True, "vertical_mixing": True},
    HABSpeciesTypeEnum.AX: {"z": -1.0, "do3D": True, "vertical_mixing": True},
    HABSpeciesTypeEnum.DP: {"do3D": True, "vertical_mixing": True},
}

# this is for the schema
_species_descriptions = {
    species.value: "Defaults: "
    + ", ".join(
        [
            f"{key}={value}"
            for key, value in SPECIES_HAB_DEFAULTS[species].model_dump().items()
        ]
    )
    for species in HABSpeciesTypeEnum
    if species != "custom"
}
_species_descriptions["custom"] = "Custom species with user-defined parameters."
