from datetime import datetime, timedelta
import json
import logging
import pathlib
from enum import Enum
from typing import Any, Dict, Optional, Union
import pandas as pd

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
from functools import cached_property
import logging

logger = logging.getLogger()


# Enum for "model"
class ModelEnum(str, Enum):
    opendrift = "opendrift"


# Enum for "seed_flag"
class SeedFlagEnum(str, Enum):
    elements = "elements"
    geojson = "geojson"


# Enum for "output_format"
class OutputFormatEnum(str, Enum):
    netcdf = "netcdf"
    parquet = "parquet"


# Enum for "log_level"
class LogLevelEnum(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Enum for "ocean_model"
class OceanModelEnum(str, Enum):
    NWGOA = "NWGOA"
    CIOFS = "CIOFS"
    CIOFSOP = "CIOFSOP"
    CIOFSFRESH = "CIOFSFRESH"

_KNOWN_MODELS = [model.value for model in OceanModelEnum]


# from geojson_pydantic import LineString, Point, Polygon

class TheManagerConfig(BaseModel):
    # seed_modulator: 
    # ocean_model: OceanModels.model_na = Field(...)
    model: ModelEnum = Field(ModelEnum.opendrift, description="Lagrangian model software to use for simulation.", ptm_level=1)
    lon: Optional[float] = Field(-151.0, ge=-180, le=180, description="Central longitude for seeding drifters. Only used if `seed_flag==\"elements\"`.", ptm_level=1, units="degrees_east")
    lat: Optional[float] = Field(58.0, ge=-90, le=90, description="Central latitude for seeding drifters. Only used if `seed_flag==\"elements\"`.", ptm_level=1, units="degrees_north")
    geojson: Optional[dict] = Field(None, description="GeoJSON describing a polygon within which to seed drifters. To use this parameter, also have `seed_flag==\"geojson\"`.", ptm_level=1)
#   geojson: Annotated[
#     Union[Point, LineString, Polygon],
#     Field(
#         ...,
#         description="GeoJSON describing a point, line, or polygon for seeding drifters.",  # noqa: E501
#     ),
    # ]
    seed_flag: SeedFlagEnum = Field(SeedFlagEnum.elements, description="Method for seeding drifters. Options are \"elements\" or \"geojson\". If \"elements\", seed drifters at or around a single point defined by lon and lat. If \"geojson\", seed drifters within a polygon described by a GeoJSON object.", ptm_level=1)
    # number: int = Field(100, description="Number of drifters to seed.", ptm_level=1, od_mapping="seed:number")
    start_time: Optional[datetime] = Field(datetime(2022,1,1), description="Start time for drifter simulation.", ptm_level=1,
                                           ge=datetime(1999,1,1), le=datetime(2023,1,2))
    start_time_end: Optional[datetime] = Field(None, description="If used, this creates a range of start times for drifters, starting with `start_time` and ending with `start_time_end`. Drifters will be initialized linearly between the two start times.", ptm_level=2)
    run_forward: bool = Field(True, description="Run forward in time.", ptm_level=2)
    time_step: int = Field(300, ge=1, le=86400, description="Interval between particles updates, in seconds.", ptm_level=3, units="seconds")
    time_step_output: int = Field(3600, ge=1, le=604800, description="Time step at which element properties are stored and eventually written to file. Must be a multiple of time_step.", ptm_level=3, units="seconds")
    steps: Optional[int] = Field(None, ge=1, le=10000, description="Maximum number of steps. End of simulation will be start_time + steps * time_step.", ptm_level=1)
    duration: Optional[timedelta] = Field(None, description="The length of the simulation. steps, end_time, or duration must be input by user.", ptm_level=1)
    duration_str: Optional[str] = Field(None, description="Duration should be input as a string of ISO 8601. The length of the simulation. steps, end_time, or duration must be input by user.", ptm_level=1)
    end_time: Optional[datetime] = Field(None, description="The end of the simulation. steps, end_time, or duration must be input by user.", ptm_level=1,
                                           ge=datetime(1999,1,1), le=datetime(2023,1,2))
    ocean_model: OceanModelEnum = Field(OceanModelEnum.CIOFSOP, description="Name of ocean model to use for driving drifter simulation.", ptm_level=1)
    ocean_model_local: bool = Field(True, description="Set to True to use local version of known `ocean_model` instead of remote version.", ptm_level=3)
    do3D: bool = Field(False, description="Set to True to run drifters in 3D, by default False for most drift models.", ptm_level=1)
    # vertical_mixing: bool = Field(False, description="Set to True to activate vertical mixing in the simulation.", ptm_level=2)
    # z: Optional[float] = Field(0, ge=-100000, le=0, description="Depth of the drifters. None to use `seed_seafloor` flag.", ptm_level=1, od_mapping="seed:z")
    # seed_seafloor: bool = Field(False, description="Set to True to seed drifters on the seafloor.", ptm_level=2, od_mapping="seed:seafloor")
    use_static_masks: bool = Field(True, description="Set to True to use static masks for known models instead of wetdry masks.", ptm_level=3)
    output_file: Optional[str] = Field(None, description="Name of file to write output to. If None, default name is used.", ptm_level=3)
    output_format: OutputFormatEnum = Field(OutputFormatEnum.netcdf, description="Output file format. Options are \"netcdf\" or \"parquet\".", ptm_level=2)
    use_cache: bool = Field(True, description="Set to True to use cache for storing interpolators.", ptm_level=3)
    # wind_drift_factor: Optional[float] = Field(0.02, description="Wind drift factor for the drifters.", ptm_level=2, od_mapping="seed:wind_drift_factor")
    # stokes_drift: bool = Field(True, description="Set to True to enable Stokes drift.", ptm_level=2, od_mapping="drift:stokes_drift")
    # horizontal_diffusivity: Optional[float] = Field(None, description="Horizontal diffusivity for the simulation.", ptm_level=2, od_mapping="drift:horizontal_diffusivity")
    log_level: LogLevelEnum = Field(LogLevelEnum.INFO, description="Log verbosity", ptm_level=3)
    # TODO: change log_level to "verbose" or similar
    
    # ocean_model_config: OceanModelConfig = Field(description="Configuration for the ocean model, comes in during runtime.", ptm_level=1)

    class Config:
        validate_defaults = True
        # use_enum_values=True

    

    @model_validator(mode='after')
    def check_config_seed_flag_elements(self) -> Self:
        if self.seed_flag == "elements" and (self.lon is None or self.lat is None):
            raise ValueError("lon and lat need non-None values if using `seed_flag=\"elements\"`.")
        return self

    @model_validator(mode='after')
    def check_config_seed_flag_geojson(self) -> Self:
        if self.seed_flag == "geojson" and self.geojson is None:
            raise ValueError("geojson need non-None value if using `seed_flag=\"geojson\"`.")
        if self.seed_flag == "geojson" and (self.lon is not None or self.lat is not None):
            raise ValueError("lon and lat need to be None if using `seed_flag=\"geojson\"`.")
        return self
    
    # @model_validator(mode='after')
    # def calculation_duration_from_duration_string(self) -> Self:
    #     # if duration and duration_str are both None, make sure they are consistent
    #     # TODO test this
    #     if self.duration is not None and self.duration_str is not None:
    #         if self.duration != pd.Timedelta(self.duration_str).isoformat():
    #             raise ValueError(f"duration and duration_str are inconsistent: {self.duration} != {self.duration_str}")
            
            
    #     if self.duration_str is not None:
    #         self.duration = pd.Timedelta(self.duration_str).isoformat()
    #         logger.info(f"Setting duration to {self.duration} based on duration_str.")
    #     return self

    @model_validator(mode='after')
    def check_config_time_parameters(self) -> Self:
        non_none_count = sum(x is not None for x in [self.start_time, self.end_time, self.duration, self.steps])
        # if non_none_count >= 2:
        #     raise ValueError(f"At least two of start_time, end_time, duration, and steps must be non-None. "
        #                      f"Current values are: start_time={self.start_time}, end_time={self.end_time}, "
        #                      f"duration={self.duration}, steps={self.steps}.")
        if non_none_count != 2:
            raise ValueError(f"Exactly two of start_time, end_time, duration, and steps must be non-None. "
                             f"Current values are: start_time={self.start_time}, end_time={self.end_time}, "
                             f"duration={self.duration}, steps={self.steps}.")
        if self.start_time is None and self.end_time is None:
            raise ValueError("One of start_time or end_time must be non-None.")
        return self

    @computed_field
    def timedir(self) -> int:
        if self.run_forward:
            value = 1
        else:
            value = -1
        return value

    # TODO change this to making sure the time inputs are consistent with each other
    @model_validator(mode='after')
    def calculate_config_times(self) -> Self:
        if self.steps is None:
            if self.duration is not None:
                self.steps = int(self.duration / timedelta(seconds=self.time_step))
                logger.info(f"Setting steps to {self.steps} based on duration.")
            elif self.end_time is not None and self.start_time is not None:
                self.steps = int(abs(self.end_time - self.start_time) / timedelta(seconds=self.time_step))
                logger.info(f"Setting steps to {self.steps} based on end_time and start_time.")
            else:
                raise ValueError("steps has not been calculated")

        if self.duration is None:
            if self.end_time is not None and self.start_time is not None:
                # import pdb; pdb.set_trace()
                self.duration = abs(self.end_time - self.start_time)
                # # convert to ISO 8601 string
                # self.duration = pd.Timedelta(abs(self.end_time - self.start_time)).isoformat()
                logger.info(f"Setting duration to {self.duration} based on end_time and start_time.")
            elif self.steps is not None:
                self.duration = self.steps * timedelta(seconds=self.time_step)
                # # convert to ISO 8601 string
                # self.duration = (self.steps * pd.Timedelta(seconds=self.time_step)).isoformat()
                logger.info(f"Setting duration to {self.duration} based on steps.")
            else:
                raise ValueError("duration has not been calculated")

        if self.end_time is None:
            if self.steps is not None and self.start_time is not None:
                self.end_time = self.start_time + self.timedir * self.steps * timedelta(seconds=self.time_step)
                logger.info(f"Setting end_time to {self.end_time} based on start_time and steps.")
            elif self.duration is not None and self.start_time is not None:
                self.end_time = self.start_time + self.timedir * self.duration
                logger.info(f"Setting end_time to {self.end_time} based on start_time and duration.")
            else:
                raise ValueError("end_time has not been calculated")

        if self.start_time is None:
            if self.end_time is not None and self.steps is not None:
                self.start_time = self.end_time - self.timedir * self.steps * timedelta(seconds=self.time_step)
                logger.info(f"Setting start_time to {self.start_time} based on end_time and steps.")
            elif self.duration is not None and self.end_time is not None:
                self.start_time = self.end_time - self.timedir * self.duration
                logger.info(f"Setting start_time to {self.start_time} based on end_time and duration.")
            else:
                raise ValueError("start_time has not been calculated")
        
        return self
    
    # # HERE unsure how to handle the time properties being input with and having only 2 of them
    # # but also how to calculate those that aren't input
    # # @cached_property
    # @property
    # def steps(self):
    #     if self.steps is None:
    #         if self.duration is not None:
    #             steps = int(self.duration / timedelta(seconds=self.time_step))
    #             logger.info(f"Setting steps to {steps} based on duration.")
    #         elif self.end_time is not None and self.start_time is not None:
    #             steps = int(abs(self.end_time - self.start_time) / timedelta(seconds=self.time_step))
    #             logger.info(f"Setting steps to {steps} based on end_time and start_time.")
    #         else:
    #             raise ValueError("steps has not been calculated")
    #     return steps
    
    # @property
    # def duration(self):
    #     if self.duration is None:
    #         if self.end_time is not None and self.start_time is not None:
    #             # import pdb; pdb.set_trace()
    #             # self.duration = abs(self.end_time - self.start_time)
    #             # convert to ISO 8601 string
    #             duration = pd.Timedelta(abs(self.end_time - self.start_time)).isoformat()
    #             logger.info(f"Setting duration to {duration} based on end_time and start_time.")
    #         elif self.steps is not None:
    #             # self.duration = self.steps * timedelta(seconds=self.time_step)
    #             # convert to ISO 8601 string
    #             duration = (self.steps * pd.Timedelta(seconds=self.time_step)).isoformat()
    #             logger.info(f"Setting duration to {duration} based on steps.")
    #         else:
    #             raise ValueError("duration has not been calculated")
    #     return duration
    
    # @property
    # def end_time(self):
    #     if self.end_time is None:
    #         if self.steps is not None and self.start_time is not None:
    #             end_time = self.start_time + self.timedir * self.steps * timedelta(seconds=self.time_step)
    #             logger.info(f"Setting end_time to {end_time} based on start_time and steps.")
    #         elif self.duration is not None and self.start_time is not None:
    #             end_time = self.start_time + self.timedir * self.duration
    #             logger.info(f"Setting end_time to {end_time} based on start_time and duration.")
    #         else:
    #             raise ValueError("end_time has not been calculated")
    #     return end_time
    
    # @property
    # def start_time(self):
    #     if self.start_time is None:
    #         if self.end_time is not None and self.steps is not None:
    #             start_time = self.end_time - self.timedir * self.steps * timedelta(seconds=self.time_step)
    #             logger.info(f"Setting start_time to {start_time} based on end_time and steps.")
    #         elif self.duration is not None and self.end_time is not None:             
    #             start_time = self.end_time - self.timedir * self.duration
    #             logger.info(f"Setting start_time to {start_time} based on end_time and duration.")
    #         else:
    #             raise ValueError("start_time has not been calculated")
    #     return start_time
 

    # @model_validator(mode='after')
    # def check_consistency_time_parameters(self) -> Self:
    #     """If all time parameters defined, make sure they are consistent with each other."""
    #     non_none_count = sum(x is not None for x in [self.start_time, self.end_time, self.duration, self.steps])
    #     if non_none_count == 4:
    #         # calculate duration and steps from start_time and end_time and make sure they are the same as what
    #         # is already saved.
    #         duration = pd.Timedelta(abs(self.end_time - self.start_time)).isoformat()
    #         steps = int(abs(self.end_time - self.start_time) / timedelta(seconds=self.time_step))
    #         if duration != self.duration:
    #             raise ValueError(f"duration and calculated duration are inconsistent: {self.duration} != {duration}")
    #         if steps != self.steps:
    #             raise ValueError(f"steps and calculated steps are inconsistent: {self.steps} != {steps}")
    #     return self
 
    # @model_validator(mode='after')
    # def check_config_start_time(self) -> Self:
    #     min_model_time = self.model_fields[f"{self.ocean_model}_time_range"].metadata[0].ge
    #     min_model_time = parse(min_model_time) if isinstance(min_model_time, str) else min_model_time
    #     max_model_time = self.model_fields[f"{self.ocean_model}_time_range"].metadata[1].le
    #     max_model_time = parse(max_model_time) if isinstance(max_model_time, str) else max_model_time
    #     if not min_model_time <= self.start_time <= max_model_time:
    #         raise ValueError(f"start_time must be between {min_model_time} and {max_model_time}")
    #     logger.info(f"start_time is within the time range of the ocean model: {min_model_time} to {max_model_time}")
    #     return self

    # @model_validator(mode='after')
    # def check_config_lon_lat_model_domain(self) -> Self:
    #     keys = ["lon", "lat"]
    #     for key in keys:
    #         if getattr(self, key) is not None:
    #             min_model = self.model_fields[f"{self.ocean_model}_{key}_range"].metadata[0].ge
    #             min_model = parse(min_model) if isinstance(min_model, str) else min_model
    #             max_model = self.model_fields[f"{self.ocean_model}_{key}_range"].metadata[1].le
    #             max_model = parse(max_model) if isinstance(max_model, str) else max_model
    #             if not min_model <= getattr(self, key) <= max_model:
    #                 raise ValueError(f"{key} must be between {min_model} and {max_model}")
    #             logger.info(f"{key} is within the range of the ocean model: {min_model} to {max_model}")
    #     return self
    
    @model_validator(mode='after')
    def check_config_ocean_model_local(self) -> Self:
        if self.ocean_model_local:
            logger.info(
                "Using local output for ocean_model."
            )
        else:
            logger.info(
                "Using remote output for ocean_model."
            )
        return self
