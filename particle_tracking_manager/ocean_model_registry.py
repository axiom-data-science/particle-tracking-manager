"""Defines OceanModelConfig with classes stored in ocean_model_registry.

Set up ocean model configuration: doesn't depend on a tracking simulation.
OceanModelConfig instances contain information about ocean models that is relevant to the model itself, separate from a particle tracking simulation.
"""

import itertools

# Standard library imports
import os
import pprint

from datetime import datetime
from pathlib import Path

# Third-party imports
import pandas as pd
import xarray as xr
import yaml

from pydantic import BaseModel, Field
from typing_extensions import Annotated


def calculate_CIOFSOP_max() -> datetime:
    """read in CIOFSOP max time available, as datetime object"""
    try:
        date = (
            xr.open_dataset(
                "/mnt/depot/data/packrat/prod/noaa/coops/ofs/aws_ciofs/processed/aws_ciofs_kerchunk.parq",
                engine="kerchunk",
            )
            .ocean_time[-1]
            .values.astype("datetime64[s]")
            .item()
        )
    except:
        date = (pd.Timestamp.now() + pd.Timedelta("1d")).isoformat()
    return date


def get_model_end_time(name: str) -> datetime:
    """Get the end time of the model based on its name."""
    # This is only run when the property is requested
    if name == "CIOFSOP":
        return calculate_CIOFSOP_max()
    else:
        raise NotImplementedError(f"get_model_end_time not implemented for {name}.")


class OceanModelConfig(BaseModel):
    """Ocean model configuration."""

    name: Annotated[
        str,
        Field(description="Name of the model."),
    ]
    temporal_resolution_str: Annotated[
        str,
        Field(
            description="ISO 8601 format temporal resolution of the model. e.g. 'PT1H' for hourly resolution."
        ),
    ]
    lon_min: Annotated[
        float,
        Field(description="Minimum longitude of the model."),
    ]
    lon_max: Annotated[
        float,
        Field(description="Maximum longitude of the model."),
    ]
    lat_min: Annotated[
        float,
        Field(description="Minimum latitude of the model."),
    ]
    lat_max: Annotated[
        float,
        Field(description="Maximum latitude of the model."),
    ]
    start_time_model: Annotated[
        datetime,
        Field(description="Start time of the model."),
    ]
    oceanmodel_lon0_360: Annotated[
        bool,
        Field(
            description="Set to True to use 0-360 longitude convention for this model."
        ),
    ]
    standard_name_mapping: Annotated[
        dict[str, str],
        Field(description="Mapping of model variable names to standard names."),
    ]
    model_drop_vars: Annotated[
        list[str],
        Field(
            description="List of variables to drop from the model dataset. These variables are not needed for particle tracking."
        ),
    ]
    loc_remote: Annotated[
        str | None,
        Field(description="Remote location of the model dataset."),
    ]
    chunks: Annotated[
        dict | None,
        Field(description="Chunking strategy for the model dataset."),
    ]
    dx: Annotated[
        float | None,
        Field(
            description="Approximate horizontal grid resolution (meters), used to calculate horizontal diffusivity."
        ),
    ]

    end_time_fixed: Annotated[
        datetime | None,
        Field(None, description="End time of the model, if doesn't change."),
    ]

    kerchunk_func_str: Annotated[
        str | None,
        Field(
            description="Name of function to create a kerchunk file for the model, mapped to function name in function_map."
        ),
    ]

    @property
    def end_time_model(self) -> datetime:
        """Get the end time of the model."""
        if self.end_time_fixed:
            return self.end_time_fixed
        else:  # there is only one that uses this currently
            return get_model_end_time(self.name)

    @property
    def horizontal_diffusivity(self) -> float | None:
        """Calculate horizontal diffusivity based on known ocean_model.

        Might be overwritten by user-input in other model config.
        """

        if self.dx is None:
            return None

        # horizontal diffusivity is calculated based on the mean horizontal grid resolution
        # for the model being used.
        # 0.1 is a guess for the magnitude of velocity being missed in the models, the sub-gridscale velocity
        sub_gridscale_velocity = 0.1
        horizontal_diffusivity = sub_gridscale_velocity * self.dx
        return horizontal_diffusivity


class OceanModelRegistry:
    """Registry for OceanModelConfig instances."""

    def __init__(self):
        """Initialize the OceanModelRegistry."""
        self._registry = {}

    def __repr__(self):
        """Return a string representation of the registry."""
        return str(self.all())

    def register(self, name: str, instance: OceanModelConfig):
        """Register an OceanModelConfig instance by its name."""
        self._registry[name] = instance

    def get(self, name: str):
        """Get an OceanModelConfig instance by its name."""
        return self._registry.get(name)

    def show(self, name: str):
        """Show the details of a registered OceanModelConfig instance."""
        return pprint.pprint(self._registry[name].model_dump())

    def get_all(self):
        """Return all registered OceanModelConfig instances."""
        return self._registry.items()

    def all(self):
        """Return all registered OceanModelConfig instance names."""
        return list(self._registry.keys())

    def all_models(self):
        """Return all registered OceanModelConfig instances."""
        return list(self._registry.values())

    def update_model(self, name, changes: dict):
        """Update a registered OceanModelConfig instance with new values."""
        if name in self._registry:
            for key, value in changes.items():
                setattr(self._registry[name], key, value)
        else:
            raise ValueError(f"Model {name} not found in registry.")


# Directory with YAML files
directory = (
    Path(__file__).resolve().parent / "ocean_models"
)  # This is the directory where the current script is located

file_paths: list = list(directory.glob("*.yaml"))

# Directory with user-defined files, if any
config_dir = Path(os.getenv("PTM_CONFIG_DIR", ""))
if len(str(config_dir)) > 1:
    file_paths = list(itertools.chain(file_paths, config_dir.glob("*.yaml")))

# also combine *.yaml files in the user_ocean_models directory specifically (not just by default)
config_dir = Path(__file__).resolve().parent / "user_ocean_models"
file_paths = list(itertools.chain(file_paths, config_dir.glob("*.yaml")))

# Create an instance of the OceanModelRegistry
ocean_model_registry = OceanModelRegistry()

# Iterate through all .yaml files in the directory
for file_path in file_paths:
    with open(file_path, "r") as f:
        config_data = yaml.safe_load(f)[file_path.stem]

        # Assuming your config_data needs to be loaded into a Pydantic model
        # Create the OceanModelConfig instance from the data
        config = OceanModelConfig(**config_data)

        # Register the configuration, perhaps by its name
        ocean_model_registry.register(config.name, config)
