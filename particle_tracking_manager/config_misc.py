
import datetime
import json
import logging
import pathlib
from enum import Enum
from typing import Any, Dict, Optional, Union

import xarray as xr
from dateutil.parser import parse
from pydantic import (
    BaseModel,
    ConfigDict,
    Extra,
    Field,
    computed_field,
    create_model,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from typing_extensions import Self

# from .utils import calc_known_horizontal_diffusivity
from .models.opendrift.utils import make_nwgoa_kerchunk, make_ciofs_kerchunk
from .config_the_manager import TheManagerConfig, OutputFormatEnum

logger = logging.getLogger()


# TODO: write tests for PTMConfig and OpenDriftConfig, and maybe SetupOutputFiles




class ParticleTrackingState(BaseModel):
    """Track simulation state."""
    has_added_reader: bool = False
    has_run_seeding: bool = False
    has_run: bool = False


def generate_default_output_file():
    return f"output-results_{datetime.datetime.now():%Y-%m-%dT%H%M%SZ}"

class SetupOutputFiles(BaseModel):
    """Handle all changes/work on output files.
    
    This class runs first thing. Then logger setup.
    """

    output_file: Optional[str] = Field(TheManagerConfig.model_json_schema()["properties"]["output_file"]["default"])
    output_format: OutputFormatEnum = Field(TheManagerConfig.model_json_schema()["properties"]["output_format"]["default"])

    class Config:
        validate_default: bool = True

    @field_validator("output_file", mode="after")
    def assign_output_file_if_needed(value: Optional[str]) -> str:
        if value is None:
            return generate_default_output_file()
        return value

    @field_validator("output_file", mode="after")
    def clean_output_file(value: str) -> str:
        value = value.replace(".nc", "").replace(".parquet", "").replace(".parq", "")
        return value

    @model_validator(mode="after")
    def add_output_file_extension(self) -> Self:

        if self.output_format is not None:
            if self.output_format == "netcdf":
                self.output_file = str(pathlib.Path(self.output_file).with_suffix(".nc"))
            elif self.output_format == "parquet":
                self.output_file = str(pathlib.Path(self.output_file).with_suffix(".parquet"))
            else:
                raise ValueError(f"output_format {self.output_format} not recognized.")
        return self

    @computed_field
    def logfile_name(self) -> str:
        return pathlib.Path(self.output_file).stem + ".log"
