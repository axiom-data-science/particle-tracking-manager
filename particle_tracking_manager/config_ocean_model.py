"""Defines OceanModelSimulation and ocean_model_simulation_mapper, a dict with key, value pairs of ocean model name and OceanModelSimulation class. Each OceanModelSimulation is built using information from an OceanModelConfig instance.

Functions needed to work with ocean model output are defined here. Also register on-the-fly instances of ocean models here.
"""

# Standard library imports
import logging

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Any, Callable

# Third-party imports
import xarray as xr

from pydantic import BaseModel, Field, create_model, model_validator
from typing_extensions import Self

# Local imports
from .models.opendrift.utils import make_ciofs_kerchunk, make_nwgoa_kerchunk
from .ocean_model_registry import OceanModelConfig, ocean_model_registry


logger = logging.getLogger()


# Define a function to generate an Enum from the registry
def generate_enum_from_registry() -> Enum:
    """Generate an Enum class from the ocean model registry."""
    # Dynamically create an Enum class using the registry data
    enum_name = "OceanModelEnum"
    enum_members = {name: name for name in ocean_model_registry.all()}
    return Enum(enum_name, enum_members)


# Generate a dynamic Enum using the registry data
# since user might add their own models.
OceanModelEnum = generate_enum_from_registry()


## Set up ocean model simulation configuration: depends on a tracking simulation. ##


class OceanModelSimulation(BaseModel):
    """Ocean model simulation configuration."""

    ocean_model_local: bool

    model_config = {
        "validate_defaults": True,
        "use_enum_values": True,
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def check_config_oceanmodel_lon0_360(self) -> Self:
        """Check if the ocean model is using 0-360 longitude convention."""
        if self.ocean_model_config.oceanmodel_lon0_360:
            if self.lon is not None and self.lon < 0:
                if -180 < self.lon < 0:
                    orig_lon = self.lon
                    self.lon += 360
                    logger.debug(f"Shifting longitude from {orig_lon} to {self.lon}.")
        return self

    def open_dataset(self, drop_vars: list) -> xr.Dataset:
        """Open an xarray dataset"""
        # if local
        if self.ocean_model_local:

            name, kerchunk_func_str = (
                self.ocean_model_config.name,
                self.ocean_model_config.kerchunk_func_str,
            )
            start_time, end_time = self.start_time, self.end_time

            if loc_local(name, kerchunk_func_str, start_time, end_time) is None:
                raise ValueError(
                    "loc_local must be set if ocean_model_local is True, but loc_local is None."
                )
            else:
                # TODO: Make a way to input chunks selection (and maybe other xarray kwargs)
                ds = xr.open_dataset(
                    loc_local(name, kerchunk_func_str, start_time, end_time),
                    engine="kerchunk",
                    # chunks={},  # Looks like it is faster not to include this for kerchunk
                    drop_variables=drop_vars,
                    decode_times=False,
                )
                logger.debug(
                    f"Opened local dataset with start time {start_time} and end time {end_time} and number outputs {ds.ocean_time.size}."
                )

        # otherwise remote
        else:
            if self.ocean_model_config.loc_remote is None:
                raise ValueError(
                    "loc_remote must be set if ocean_model_local is False, but loc_remote is None."
                )
            else:
                if ".nc" in self.ocean_model_config.loc_remote:
                    ds = xr.open_dataset(
                        self.ocean_model_config.loc_remote,
                        chunks=self.ocean_model_config.chunks,
                        drop_variables=drop_vars,
                        decode_times=False,
                    )
                elif ".parquet" in self.ocean_model_config.loc_remote:
                    ds = xr.open_dataset(
                        self.ocean_model_config.loc_remote,
                        engine="kerchunk",
                        # chunks={},  # Looks like it is faster not to include this for kerchunk
                        drop_variables=drop_vars,
                        decode_times=False,
                    )
                else:
                    ds = xr.open_zarr(
                        self.ocean_model_config.loc_remote,
                        chunks=self.ocean_model_config.chunks,
                        drop_variables=drop_vars,
                        decode_times=False,
                    )

                logger.debug(
                    f"Opened remote dataset {self.ocean_model_config.loc_remote} with number outputs {ds.ocean_time.size}."
                )
        return ds


def create_ocean_model_simulation(
    ocean_model: OceanModelConfig,
) -> type[OceanModelSimulation]:
    """Create an ocean model simulation object."""
    ocean_model_name = ocean_model.name
    simulation_model = create_model(
        ocean_model_name,  # Model name
        __base__=OceanModelSimulation,
        lon=(
            float | None,
            Field(
                ...,
                ge=getattr(ocean_model, "lon_min"),
                le=getattr(ocean_model, "lon_max"),
                description="Longitude of the simulation within the model bounds.",
            ),
        ),
        lat=(
            float | None,
            Field(
                ...,
                ge=getattr(ocean_model, "lat_min"),
                le=getattr(ocean_model, "lat_max"),
                description="Latitude of the simulation within the model bounds.",
            ),
        ),
        start_time=(
            datetime,
            Field(
                ...,
                ge=getattr(ocean_model, "start_time_model"),
                le=getattr(ocean_model, "end_time_model"),
                description="Start time of the simulation.",
            ),
        ),
        end_time=(
            datetime,
            Field(
                ...,
                ge=getattr(ocean_model, "start_time_model"),
                le=getattr(ocean_model, "end_time_model"),
                description="End time of the simulation.",
            ),
        ),
        ocean_model_config=(OceanModelConfig, ocean_model),
    )
    return simulation_model


ocean_model_simulation_mapper = {}
for ocean_model in ocean_model_registry.all_models():
    ocean_model_simulation_mapper[ocean_model.name] = create_ocean_model_simulation(
        ocean_model
    )


MAKE_KERCHUNK_FUNCTIONS: dict[
    str, Callable[[datetime, datetime, str], dict[Any, Any]]
] = {
    "make_nwgoa_kerchunk": make_nwgoa_kerchunk,
    "make_ciofs_kerchunk": make_ciofs_kerchunk,
}


def loc_local(
    name: str, kerchunk_func_str: str, start_sim: datetime, end_sim: datetime
) -> dict:
    """This sets up a short kerchunk file for reading in just enough model output."""

    # back each start time back 1 day and end time forward 1 day to make sure enough output is available
    if start_sim < end_sim:
        start_time = start_sim - timedelta(days=1)
        end_time = end_sim + timedelta(days=1)
    else:
        start_time = start_sim + timedelta(days=1)
        end_time = end_sim - timedelta(days=1)

    return MAKE_KERCHUNK_FUNCTIONS[kerchunk_func_str](start_time, end_time, name)


def register_on_the_fly(ds_info: dict, ocean_model: str = "ONTHEFLY") -> None:
    """Update an ocean model in the registry and rebuild its Pydantic model.

    The default model to register is "ONTHEFLY", which is a placeholder for user-defined models.
    However, alternations could also be made to any existing model in the registry.

    ds_info can contain any of the OceanModelConfig fields.
    """

    # Update the ocean model template with user dataset information
    ocean_model_registry.update_model(ocean_model, ds_info)

    # Create the ocean model simulation object for the new ocean model
    simulation_cls = create_ocean_model_simulation(
        ocean_model_registry.get(ocean_model)
    )

    # Update the ocean model simulation mapper with the new ocean model simulation
    ocean_model_simulation_mapper.update({ocean_model: simulation_cls})

    logger.info(
        "Registered new ocean model or altered existing ocean model in the registry."
    )


def update_TXLA_with_download_location() -> None:
    """The user-defined TXLA model is missing the download location

    because it depends on the user's operating system and setup. Run this
    function to update the TXLA model with the location of the downloaded
    file.
    """

    import xroms

    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    ds_info = dict(loc_remote=url)
    register_on_the_fly(ds_info, ocean_model="TXLA")
    logger.debug("Updated TXLA model with download location.")
