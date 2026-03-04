"""Enums for OpenDrift."""

from .object_types import ObjectTypeEnum
from .oil_types import NAME_TO_OIL_ID, OIL_ID_TO_NAME, OilTypeEnum
from .others import (
    CoastlineActionEnum,
    DiffusivityModelEnum,
    DriftModelEnum,
    DropletSizeDistributionEnum,
    HatchingMethodEnum,
    ParticleTypeEnum,
    PlotTypeEnum,
    RadiusTypeEnum,
    SeafloorActionEnum,
    VerticalBehaviorModeEnum,
)


__all__ = [
    "OilTypeEnum",
    "OIL_ID_TO_NAME",
    "NAME_TO_OIL_ID",
    "ObjectTypeEnum",
    "DriftModelEnum",
    "RadiusTypeEnum",
    "DiffusivityModelEnum",
    "CoastlineActionEnum",
    "SeafloorActionEnum",
    "PlotTypeEnum",
    "DropletSizeDistributionEnum",
    "VerticalBehaviorModeEnum",
    "HatchingMethodEnum",
    "ParticleTypeEnum",
]
