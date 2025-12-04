"""Other smaller Enums for OpenDrift"""

from enum import Enum


# Enum for drift_model
class DriftModelEnum(str, Enum):
    """Enum for drift models used in OpenDrift."""

    OceanDrift = "OceanDrift"
    LarvalFish = "LarvalFish"
    OpenOil = "OpenOil"
    Leeway = "Leeway"
    HarmfulAlgalBloom = "HarmfulAlgalBloom"


# Enum for radius_type
class RadiusTypeEnum(str, Enum):
    """Enum for radius types used in OpenDrift."""

    gaussian = "gaussian"
    uniform = "uniform"


# Define Pydantic Enum classes
class DiffusivityModelEnum(str, Enum):
    """Enum for diffusivity models used in OpenDrift."""

    environment = "environment"
    stepfunction = "stepfunction"
    windspeed_Sundby1983 = "windspeed_Sundby1983"
    windspeed_Large1994 = "windspeed_Large1994"
    gls_tke = "gls_tke"
    constant = "constant"


class CoastlineActionEnum(str, Enum):
    """Enum for coastline actions used in OpenDrift."""

    none = "none"
    stranding = "stranding"
    previous = "previous"


class SeafloorActionEnum(str, Enum):
    """Enum for seafloor actions used in OpenDrift."""

    none = "none"
    lift_to_seafloor = "lift_to_seafloor"
    deactivate = "deactivate"
    previous = "previous"


class PlotTypeEnum(str, Enum):
    """Enum for plot types used in OpenDrift."""

    spaghetti = "spaghetti"
    animation = "animation"
    animation_profile = "animation_profile"
    oil = "oil"
    property = "property"
    all = "all"


class DropletSizeDistributionEnum(str, Enum):
    """Enum for droplet size distribution types."""

    uniform = "uniform"
    normal = "normal"
    lognormal = "lognormal"
