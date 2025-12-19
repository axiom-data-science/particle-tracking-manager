"""Defines TheManagerConfig class for the particle tracking manager."""

# Standard library imports
import logging

from datetime import datetime, timedelta
from enum import Enum
from os import PathLike
from typing import Annotated, Self

# Third-party imports
import pandas as pd

# from geojson import GeoJSON
from pydantic import BaseModel, Field, computed_field, model_validator

# Local imports
from .config_ocean_model import (
    OceanModelEnum,
    OceanModelSimulation,
    ocean_model_simulation_mapper,
    register_on_the_fly,
    update_TXLA_with_download_location,
)
from .ocean_model_registry import (
    OceanModelConfig,
    get_model_end_time,
    ocean_model_registry,
)


logger = logging.getLogger()


# Enum for "model"
class ModelEnum(str, Enum):
    """Lagrangian model software to use for simulation."""

    opendrift = "opendrift"


# Enum for "output_format"
class OutputFormatEnum(str, Enum):
    """Output file format."""

    netcdf = "netcdf"
    parquet = "parquet"
    both = "both"


# Enum for "log_level"
class LogLevelEnum(str, Enum):
    """Log verbosity."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# from geojson_pydantic import LineString, Point, Polygon


class TheManagerConfig(BaseModel):
    """Configuration for the particle tracking manager."""

    model: ModelEnum = Field(
        ModelEnum.opendrift,
        description="Lagrangian model software to use for simulation.",
        json_schema_extra=dict(ptm_level=1),
    )
    lon: float | None = Field(
        None,
        ge=-180,
        le=180,
        description="Central longitude for seeding drifters. If this is set, `lat` should also be set, and `geojson` should be None.",
        json_schema_extra=dict(ptm_level=1, units="degrees_east"),
    )
    lat: float | None = Field(
        None,
        ge=-90,
        le=90,
        description="Central latitude for seeding drifters. If this is set, `lon` should also be set, and `geojson` should be None.",
        json_schema_extra=dict(ptm_level=1, units="degrees_north"),
    )
    geojson: dict | None = Field(
        None,
        description='GeoJSON describing a polygon within which to seed drifters; must contain "geometry". If this is set, `lon` and `lat` should be None.',
        json_schema_extra=dict(ptm_level=1),
    )
    start_time: datetime | None = Field(
        datetime(2022, 1, 1),
        description="Start time for drifter simulation. start_time or end_time must be input. If a timezone is included, it will be used and the time will be converted to UTC which is the same timezone as the models.",
        json_schema_extra=dict(ptm_level=1),
    )
    start_time_end: datetime | None = Field(
        None,
        description="If used, this creates a range of start times for drifters, starting with `start_time` and ending with `start_time_end`. Drifters will be initialized linearly between the two start times. If a timezone is included, it will be used and the time will be converted to UTC which is the same timezone as the models.",
        json_schema_extra=dict(ptm_level=2),
    )
    run_forward: bool = Field(
        True, description="Run forward in time.", json_schema_extra=dict(ptm_level=2)
    )
    time_step: float = Field(
        300,
        ge=0.01,
        le=1440,
        description="Interval between particles updates, in seconds.",
        json_schema_extra=dict(ptm_level=3, units="seconds"),
    )
    time_step_output: float = Field(
        3600,
        ge=1,
        le=86400,
        description="Time step at which element properties are stored and eventually written to file. This must be larger than the calculation time step, and be an integer multiple of this.",
        json_schema_extra=dict(ptm_level=3, units="seconds"),
    )
    steps: int | None = Field(
        None,
        ge=1,
        le=10000,
        description="Maximum number of steps. End of simulation will be start_time + steps * time_step.",
        json_schema_extra=dict(ptm_level=1),
    )
    duration: str | None = Field(
        None,
        description="Duration should be input as a string of ISO 8601. The length of the simulation. steps, end_time, or duration must be input by user.",
        json_schema_extra=dict(ptm_level=1),
    )
    end_time: datetime | None = Field(
        None,
        description="The end of the simulation. steps, end_time, or duration must be input by user. start_time or end_time must be input. If a timezone is included, it will be used and the time will be converted to UTC which is the same timezone as the models.",
        json_schema_extra=dict(ptm_level=1),
    )
    # OceanModelEnum was created dynamically and will trigger an issue with mypy, so we suppress it
    ocean_model: OceanModelEnum = Field(  # type: ignore
        "CIOFSOP",
        description="Name of ocean model to use for driving drifter simulation.",
        json_schema_extra=dict(ptm_level=1),
    )
    ocean_model_local: bool = Field(
        True,
        description="Set to True to use local version of known `ocean_model` instead of remote version.",
        json_schema_extra=dict(ptm_level=3),
    )
    do3D: bool = Field(
        False,
        description="Set to True to run drifters in 3D, by default False for most drift models.",
        json_schema_extra=dict(ptm_level=1),
    )
    use_static_masks: bool = Field(
        True,
        description="If False, use static ocean model land masks. This saves some computation time but since the available ocean models run in wet/dry mode, it is inconsistent with the ROMS output files in some places since the drifters may be allowed (due to the static mask) to enter a cell they wouldn't otherwise. However, it doesn't make much of a difference for simulations that aren't in the tidal flats. Use the time-varying wet/dry masks (set to True) if drifters are expected to run in the tidal flats. This costs some more computational time but is fully consistent with the ROMS output files.",
        json_schema_extra=dict(ptm_level=3),
    )
    output_file: PathLike[str] | None = Field(
        None,
        description="Name of file to write output to. If None, default name is used.",
        json_schema_extra=dict(ptm_level=3),
    )
    output_format: OutputFormatEnum = Field(
        OutputFormatEnum.netcdf,
        description='Output file format. Options are "netcdf", "parquet", or "both".',
        json_schema_extra=dict(ptm_level=2),
    )
    use_cache: bool = Field(
        True,
        description="Set to True to use cache for storing interpolators.",
        json_schema_extra=dict(ptm_level=3),
    )
    log_level: LogLevelEnum = Field(
        LogLevelEnum.INFO,
        description="Log verbosity",
        json_schema_extra=dict(ptm_level=3),
    )
    # TODO: change log_level to "verbose" or similar

    horizontal_diffusivity: float | None = Field(
        default=None,
        description="Add horizontal diffusivity (random walk). For known ocean models, the value is calculated as the approximate horizontal grid resolution for the selected ocean model times an estimate of the scale of sub-gridscale velocity of 0.1 m/s.",
        title="Horizontal Diffusivity",
        ge=0,
        le=100000,
        json_schema_extra=dict(units="m2/s", ptm_level=2),
    )

    stokes_drift: bool = Field(
        default=True,
        description="Advection elements with Stokes drift (wave orbital motion).",
        title="Stokes Drift",
        json_schema_extra=dict(ptm_level=2),
    )

    z: float | None = Field(
        default=0,
        description="Depth below sea level where elements are released. This depth is neglected if seafloor seeding is set selected.",
        title="Z",
        le=0,
        ge=-10000,
        json_schema_extra=dict(units="m", ptm_level=1),
    )

    number: int = Field(
        default=1,
        description="The number of elements for the simulation.",
        title="Number",
        ge=1,
        json_schema_extra=dict(
            units=1,
            ptm_level=1,
        ),
    )

    model_config = {
        "validate_default": True,
        # Field values will be returned as the `enum.value` - a string in most our cases
        "use_enum_values": True,
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def check_lon_lat_geojson_consistency(self) -> Self:
        """Check if lon and lat are set when geojson is None, and vice versa."""
        if self.geojson is None:
            if self.lon is None and self.lat is None:
                self.lon = -151.0
                self.lat = 58.0
                logger.info(
                    "Using default lon and lat set to -151.0 and 58.0 respectively."
                )
            elif self.lon is None or self.lat is None:
                raise ValueError(
                    "If `geojson` is None, both `lon` and `lat` must be set."
                )
        else:
            if self.lon is not None or self.lat is not None:
                raise ValueError(
                    "If `geojson` is set, both `lon` and `lat` must be None."
                )
        return self

    @computed_field
    def seed_flag(self) -> str:
        """Determine seed_flag based on whether geojson is set."""
        if self.geojson is None:
            value = "elements"
            logger.info('Using seed_flag "elements".')
        else:
            value = "geojson"
            logger.info('Using seed_flag "geojson".')
        return value

    @model_validator(mode="after")
    def check_config_time_parameters(self) -> Self:
        """Check if exactly two of start_time, end_time, duration, and steps are set."""
        non_none_count = sum(
            x is not None
            for x in [self.start_time, self.end_time, self.duration, self.steps]
        )
        if non_none_count == 4:
            # calculate duration and steps from start_time and end_time and make sure they are the same as what
            # is already saved.
            assert self.start_time is not None
            assert self.end_time is not None
            duration = pd.Timedelta(abs(self.end_time - self.start_time)).isoformat()
            steps = int(
                abs(self.end_time - self.start_time) / timedelta(seconds=self.time_step)
            )
            if duration != self.duration:
                raise ValueError(
                    f"duration and calculated duration are inconsistent: {self.duration} != {duration}"
                )
            if steps != self.steps:
                raise ValueError(
                    f"steps and calculated steps are inconsistent: {self.steps} != {steps}"
                )
        elif non_none_count != 2:
            raise ValueError(
                f"Exactly two of start_time, end_time, duration, and steps must be non-None. "
                f"Current values are: start_time={self.start_time}, end_time={self.end_time}, "
                f"duration={self.duration}, steps={self.steps}."
            )
        if self.start_time is None and self.end_time is None:
            raise ValueError("One of start_time or end_time must be non-None.")
        return self

    @computed_field
    def timedir(self) -> int:
        """Set timedir to 1 for forward, -1 for backward."""
        if self.run_forward:
            value = 1
            logger.info("Running model forward in time.")
        else:
            value = -1
            logger.info("Running model backward in time.")
        return value

    @model_validator(mode="after")
    def match_time_step_sign_to_timedir_sign(self) -> Self:
        """If simulation is backward, make time_step negative.

        Sign of input time_step is ignored.
        """
        self.time_step = abs(self.time_step) * self.timedir
        logger.debug(f"Setting time_step to {self.time_step} to match timedir sign.")
        return self

    @model_validator(mode="after")
    def calculate_config_times(self) -> Self:
        """Calculate start_time, end_time, duration, and steps based on the other parameters."""
        if self.steps is None:
            if self.duration is not None:
                self.steps = int(
                    pd.Timedelta(self.duration) / pd.Timedelta(seconds=self.time_step)
                )
                logger.debug(f"Setting steps to {self.steps} based on duration.")
            elif self.end_time is not None and self.start_time is not None:
                self.steps = int(
                    abs(self.end_time - self.start_time)
                    / timedelta(seconds=self.time_step)
                )
                logger.debug(
                    f"Setting steps to {self.steps} based on end_time and start_time."
                )
            else:
                raise ValueError("steps has not been calculated")

        if self.duration is None:
            if self.end_time is not None and self.start_time is not None:
                self.duration = pd.Timedelta(
                    abs(self.end_time - self.start_time)
                ).isoformat()
                # # convert to ISO 8601 string
                # self.duration = pd.Timedelta(abs(self.end_time - self.start_time)).isoformat()
                logger.debug(
                    f"Setting duration to {self.duration} based on end_time and start_time."
                )
            elif self.steps is not None:
                self.duration = pd.Timedelta(
                    self.steps * timedelta(seconds=self.time_step)
                ).isoformat()
                # # convert to ISO 8601 string
                # self.duration = (self.steps * pd.Timedelta(seconds=self.time_step)).isoformat()
                logger.debug(f"Setting duration to {self.duration} based on steps.")
            else:
                raise ValueError("duration has not been calculated")

        if self.end_time is None:
            if self.steps is not None and self.start_time is not None:
                self.end_time = self.start_time + self.timedir * self.steps * timedelta(
                    seconds=self.time_step
                )
                logger.debug(
                    f"Setting end_time to {self.end_time} based on start_time and steps."
                )
            elif self.duration is not None and self.start_time is not None:
                self.end_time = self.start_time + self.timedir * self.duration
                logger.debug(
                    f"Setting end_time to {self.end_time} based on start_time and duration."
                )
            else:
                raise ValueError("end_time has not been calculated")

        if self.start_time is None:
            if self.end_time is not None and self.steps is not None:
                self.start_time = self.end_time - self.timedir * self.steps * timedelta(
                    seconds=self.time_step
                )
                logger.debug(
                    f"Setting start_time to {self.start_time} based on end_time and steps."
                )
            elif self.duration is not None and self.end_time is not None:
                self.start_time = self.end_time - self.timedir * self.duration
                logger.debug(
                    f"Setting start_time to {self.start_time} based on end_time and duration."
                )
            else:
                raise ValueError("start_time has not been calculated")

        return self

    @model_validator(mode="after")
    def remove_timezone_info(self) -> Self:
        """Remove timezone information from datetime fields."""
        if self.start_time is not None and self.start_time.tzinfo is not None:
            self.start_time = (
                pd.Timestamp(self.start_time)
                .tz_convert("UTC")
                .tz_localize(None)
                .to_pydatetime()
            )
            logger.debug(
                f"Removed timezone info from start_time, new value is {self.start_time}."
            )
        if self.start_time_end is not None and self.start_time_end.tzinfo is not None:
            self.start_time_end = (
                pd.Timestamp(self.start_time_end)
                .tz_convert("UTC")
                .tz_localize(None)
                .to_pydatetime()
            )
            logger.debug(
                f"Removed timezone info from start_time_end, new value is {self.start_time_end}."
            )
        if self.end_time is not None and self.end_time.tzinfo is not None:
            self.end_time = (
                pd.Timestamp(self.end_time)
                .tz_convert("UTC")
                .tz_localize(None)
                .to_pydatetime()
            )
            logger.debug(
                f"Removed timezone info from end_time, new value is {self.end_time}."
            )
        return self

    @computed_field
    def ocean_model_config(self) -> OceanModelConfig:
        """Select ocean model config based on ocean_model."""
        return ocean_model_registry.get(self.ocean_model)

    @computed_field
    def ocean_model_simulation(self) -> OceanModelSimulation:
        """Select ocean model simulation based on ocean_model."""

        # Before validating OceanModelSimulation with CIOFSOP, we need to refresh
        # the model end time because the model is dynamic. That way the time extent
        # validation is up to date.
        if self.ocean_model == "CIOFSOP":
            new_end = get_model_end_time("CIOFSOP")
            # Update the existing model in the registry
            register_on_the_fly({"end_time_fixed": new_end}, ocean_model="CIOFSOP")

        inputs = {
            "lon": self.lon,
            "lat": self.lat,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "ocean_model_local": self.ocean_model_local,
        }
        return ocean_model_simulation_mapper[self.ocean_model](**inputs)

    @model_validator(mode="after")
    def select_ocean_model_simulation_on_init(self) -> Self:
        """Select ocean model simulation based on ocean_model."""
        self.ocean_model_simulation
        return self

    @model_validator(mode="after")
    def assign_horizontal_diffusivity(self) -> Self:
        """Calculate horizontal diffusivity based on ocean model."""

        # check horizontal_diffusivity from TheManagerConfig
        if self.horizontal_diffusivity is not None:
            logger.debug(
                f"Setting horizontal_diffusivity to user-selected value {self.horizontal_diffusivity}."
            )

        # otherwise use ocean_model_config version of horizontal_diffusivity
        elif (
            self.ocean_model_config is not None
            and self.ocean_model_config.name in ocean_model_registry.all()
        ):

            self.horizontal_diffusivity = self.ocean_model_config.horizontal_diffusivity
            logger.debug(
                f"Setting horizontal_diffusivity parameter to one tuned to reader model of value {self.horizontal_diffusivity}."
            )

        elif (
            self.ocean_model_config is not None
            and self.ocean_model_config.name not in ocean_model_registry.all()
            and self.horizontal_diffusivity is None
        ):

            logger.debug(
                """Since ocean_model is user-input, changing horizontal_diffusivity parameter from None to 0.0.
                You can also set it to a specific value with `m.horizontal_diffusivity=[number]`."""
            )

            self.horizontal_diffusivity = 0

        return self

    @model_validator(mode="after")
    def check_config_ocean_model_local(self) -> Self:
        """Descrive how ocean_model_local is set."""
        if self.ocean_model_local:
            logger.debug("Using local output for ocean_model.")
        else:
            logger.debug("Using remote output for ocean_model.")
        return self

    @model_validator(mode="after")
    def setup_TXLA_if_used(self) -> Self:
        """Set up TXLA if used."""
        if self.ocean_model == "TXLA":
            update_TXLA_with_download_location()
        return self
