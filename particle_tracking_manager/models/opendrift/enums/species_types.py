"""OpenDrift HAB species types enum definition."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class HABSpeciesTypeEnum(str, Enum):
    """Harmful Algal Bloom species types supported by OpenDrift."""

    PN = "PN"
    # AX = "AX"
    # DP = "DP"
    custom = "custom"


HAB_SPECIES_LABELS = {
    HABSpeciesTypeEnum.PN: "Pseudo nitzschia",
    # HABSpeciesTypeEnum.AX: "Alexandrium",
    # HABSpeciesTypeEnum.DP: "Dinophysis",
    HABSpeciesTypeEnum.custom: "Custom species (manual parameters)",
}


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


# Species default for HarmfulAlgalBloom model
SPECIES_HAB_DEFAULTS: dict[HABSpeciesTypeEnum, HABParameters] = {
    HABSpeciesTypeEnum.PN: HABParameters(
        temperature_death_min=3.0,
        temperature_death_max=22.0,
        mortality_rate_high=1.0,
        salinity_death_min=25.0,
        salinity_death_max=36.0,
    ),
    # HABSpeciesTypeEnum.custom intentionally has no entry
}


# Other config defaults per species (z, do3D, etc.)
SPECIES_HAB_MANAGER_DEFAULTS: dict[HABSpeciesTypeEnum, dict[str, object]] = {
    HABSpeciesTypeEnum.PN: {
        "z": -1.0,
        "do3D": False,
    },
    # HABSpeciesTypeEnum.Alexandrium: {"do3D": True, ...},
    # HABSpeciesTypeEnum.Dinophysis: {...},
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
