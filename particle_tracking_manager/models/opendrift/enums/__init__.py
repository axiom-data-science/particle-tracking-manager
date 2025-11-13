"""Enums for OpenDrift."""

from .object_types import ObjectTypeEnum
from .oil_types import OIL_ID_TO_NAME, OilTypeEnum
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
    "OIL_ID_TO_NAME",
    "ObjectTypeEnum",
    "DriftModelEnum",
    "RadiusTypeEnum",
    "DiffusivityModelEnum",
    "CoastlineActionEnum",
    "SeafloorActionEnum",
    "PlotTypeEnum",
    "DropletSizeDistributionEnum",
]
