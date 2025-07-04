"""Enums for OpenDrift."""

from .object_types import ObjectTypeEnum
from .oil_types import ModifyOilTypeJsonSchema, OilTypeEnum
from .others import (
    CoastlineActionEnum,
    DiffusivityModelEnum,
    DriftModelEnum,
    DropletSizeDistributionEnum,
    PlotTypeEnum,
    RadiusTypeEnum,
    SeafloorActionEnum,
)


__all__ = [
    "OilTypeEnum",
    "ModifyOilTypeJsonSchema",
    "ObjectTypeEnum",
    "DriftModelEnum",
    "RadiusTypeEnum",
    "DiffusivityModelEnum",
    "CoastlineActionEnum",
    "SeafloorActionEnum",
    "PlotTypeEnum",
    "DropletSizeDistributionEnum",
]
