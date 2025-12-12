"""Defines ParticleTrackingState and SetupOutputFiles."""

# Standard library imports
import datetime
import logging
import pathlib

from os import PathLike
from typing import Any, Mapping, Self

# Third-party imports
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

# Local imports
from .config_the_manager import OutputFormatEnum, TheManagerConfig


OUTPUT_FORMAT_SUFFIX: Mapping[OutputFormatEnum, str] = {
    OutputFormatEnum.netcdf: ".nc",
    OutputFormatEnum.parquet: ".parquet",  # can't be read back in by OpenDrift which is necessary for plots in PTM
    OutputFormatEnum.both: ".nc",  # first netcdf, then additionally output parquet at the end
}

logger = logging.getLogger()


class ParticleTrackingState(BaseModel):
    """Track simulation state."""

    has_run_setup: bool = False  # this may not be required for all models
    has_added_reader: bool = False
    has_run_seeding: bool = False
    has_run: bool = False


def generate_default_output_file() -> str:
    """Generate a default output file name based on the current date and time."""
    return f"output-results_{datetime.datetime.now():%Y-%m-%dT%H%M%SZ}"


class SetupOutputFiles(BaseModel):
    """Handle all changes/work on output files.

    This class runs first thing. Then logger setup.
    """

    output_file: PathLike[str] | None = Field(
        TheManagerConfig.model_json_schema()["properties"]["output_file"]["default"]
    )
    output_format: OutputFormatEnum = Field(
        TheManagerConfig.model_json_schema()["properties"]["output_format"]["default"]
    )

    model_config = {"validate_default": True}

    @field_validator("output_file", mode="after")
    def assign_output_file_if_needed(value: Any) -> str:
        """Assign a default output file name if not provided."""
        if value is None:
            return generate_default_output_file()
        return value

    @model_validator(mode="after")
    def set_output_file_extension(self) -> Self:
        """Set the appropriate file extension based on the output format."""
        assert self.output_file is not None
        if self.output_format not in OUTPUT_FORMAT_SUFFIX:
            raise ValueError(f"output_format {self.output_format} not recognized.")

        self.output_file = pathlib.Path(self.output_file).with_suffix(
            OUTPUT_FORMAT_SUFFIX[self.output_format]
        )

        return self

    @computed_field
    def logfile_name(self) -> str:
        """Generate a log file name based on the output file name."""
        assert self.output_file is not None
        return str(pathlib.Path(self.output_file).with_suffix(".log"))
