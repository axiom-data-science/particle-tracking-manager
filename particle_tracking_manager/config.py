"""Configuration setup for particle tracking manager."""

import datetime
import json
import logging
import pathlib
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Union

import xarray as xr
from dateutil.parser import parse
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Extra,
    Field,
    AfterValidator,
    create_model,
    field_validator,
    model_validator,
    validator,
)
from typing_extensions import Self

from .utils import calc_known_horizontal_diffusivity



# class LoggerConfig:
#     """Logger configuration."""
    
#     def __init__(self):#, log_level: str):
#         pass

#     def assign_output_file_if_needed(self, value: Optional[str]) -> str: 
#         if value is None:
#             value = generate_default_output_file()
#         return value

#     def clean_output_file(self, value: str) -> str:
#         value = value.replace(".nc", "").replace(".parquet", "").replace(".parq", "")
#         return value    

#     def close_loggers(self, logger):
#         """Close and remove all handlers from the logger."""
#         for handler in logger.handlers[:]:
#             handler.close()
#             logger.removeHandler(handler)

#     def setup_logger(self, output_file: Optional[str], log_level: str) -> (logging.Logger, str):
#         """Setup logger."""

#         output_file = self.assign_output_file_if_needed(output_file)
#         output_file = self.clean_output_file(output_file)
#         # self.output_file = output_file

#         logger = logging.getLogger(__package__)
#         if logger.handlers:
#             self.close_loggers(logger)
            
#         logger.setLevel(getattr(logging, log_level))

#         # Add handlers from the main logger to the OpenDrift logger if not already added
        
#         # Create file handler to save log to file
#         logfile_name = output_file + ".log"
#         file_handler = logging.FileHandler(logfile_name)
#         fmt = "%(asctime)s %(levelname)-7s %(name)s.%(module)s.%(funcName)s:%(lineno)d: %(message)s"
#         datefmt = '%Y-%m-%d %H:%M:%S'
#         formatter = logging.Formatter(fmt, datefmt)
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)

#         # Create stream handler
#         stream_handler = logging.StreamHandler()
#         stream_handler.setFormatter(formatter)
#         logger.addHandler(stream_handler)
        
#         logger.info("Particle tracking manager simulation.")
#         logger.info(f"Output filename: {output_file}")
#         logger.info(f"Log filename: {logfile_name}")
#         return logger, output_file

#     def merge_with_opendrift_log(self, logger: logging.Logger) -> None:
#         """Merge the OpenDrift logger with the main logger."""

#         for logger_name in logging.root.manager.loggerDict:
#             if logger_name.startswith("opendrift"):
#                 od_logger = logging.getLogger(logger_name)
#                 if od_logger.handlers:
#                     self.close_loggers(od_logger)

#                 # Add handlers from the main logger to the OpenDrift logger
#                 for handler in logger.handlers:
#                     od_logger.addHandler(handler)
#                 od_logger.setLevel(logger.level)
#                 od_logger.propagate = False


class ParticleTrackingState(BaseModel):
    """Track simulation state."""
    has_added_reader: bool = False
    has_run_seeding: bool = False
    has_run: bool = False



def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def generate_default_output_file():
    return f"output-results_{datetime.datetime.now():%Y-%m-%dT%H%M%SZ}"


def calculate_CIOFSOP_max():
    """read in CIOFSOP max time available, at datetime object"""
    return xr.open_dataset("/mnt/depot/data/packrat/prod/noaa/coops/ofs/aws_ciofs/processed/aws_ciofs_kerchunk.parq", engine="kerchunk").ocean_time[-1].values.astype('datetime64[s]').item()


def create_enum(name: str, values: list):
    return Enum(name, {v: v for v in values})


def create_pydantic_model(config: Dict[str, Any]) -> (Dict[str, Any], Dict[str, Any]):
    fields = {}
    validators = {}
    
    for key, value in config.items():
        field_type = value.get("type", "str")

        # TODO: not every field type should be Optional. For example, time_step should be required
        if field_type == "datetime.datetime":
            field_type = Optional[datetime.datetime]
        # field_type = Optional[Union[str, datetime.datetime]]
            # default = datetime.datetime.now() if default == "datetime.datetime.now()" else default
        elif field_type == "float":
            field_type = Optional[float]
        elif field_type == "int":
            field_type = Optional[int]
        elif field_type == "bool":
            field_type = Optional[bool]
        elif field_type == "str":
            field_type = Optional[str]
        elif field_type == "geojson":
            field_type = Optional[dict]
        elif field_type == "enum":
            # field_type = Optional[str]
            enum_name = f"{key.capitalize()}Enum"
            enum_values = value.get("enum", [])
            field_type = create_enum(enum_name, enum_values)
            # field_type = Optional[create_enum(enum_name, enum_values)]  # change back to this if want to use ocean_model=None
        elif field_type == "datetime.timedelta":
            field_type = Optional[datetime.timedelta]
        else:
            field_type = Optional[str]

        # bring metadata from config into pydantic model
        kwargs = dict()
        items = ["default", "description", "min", "max", "units", "ptm_level", "od_mapping", "enum"]
        for item in items:

            # TODO: can these be changed to validators?
            
            if value.get(item) is not None:
                if isinstance(value.get(item), str) and value.get(item) == "CIOFSOP_max":
                    value[item] = calculate_CIOFSOP_max()

                if item == "min":
                    kwargs["ge"] = value.get(item)
                elif item == "max":
                    kwargs["le"] = value.get(item)
                else:
                    kwargs[item] = value.get(item)
            else:
                kwargs[item] = None
        
        field_info = Field(**kwargs, validate_default=True)
        fields[key] = (field_type, field_info)

    return fields, validators


def add_special_fields_and_validators_manager(fields: Dict[str, Any], validators: Dict[str, Any]) -> None:
    fields["timedir"] = (int, Field(default=1))
    fields["oceanmodel_lon0_360"] = (bool, Field(default=False))
    fields["output_file_initial"] = (Optional[str], Field(default=None))
    # Add logger to fields
    fields["logger"] = (logging.Logger, Field(default=logging.getLogger(__name__), exclude=True))
    
    # # Track if this config has been run before
    # fields["initialized"] = (bool, Field(default=False))

    # moved to logger config
    # @field_validator("output_file", mode="after")
    # @classmethod
    # def assign_output_file_if_needed(cls, value: Optional[str]) -> str: 
    #     if value is None:
    #         value = generate_default_output_file()
    #     return value

    # @field_validator("output_file", mode="after")
    # @classmethod
    # def clean_output_file(cls, value: str) -> str:
    #     value = value.replace(".nc", "").replace(".parquet", "").replace(".parq", "")
    #     return value

    # @computed_field
    # def output_file_initial(self) -> str:
    #     value = str(pathlib.Path(f"{self.output_file}_initial"))#.with_suffix(".nc"))
    #     print("\nNOTE: ", self.output_file_initial, self.output_file, "\n")
    #     return value

    @model_validator(mode="after")
    def assign_output_file_initial(self) -> Self:
        self.output_file_initial = str(pathlib.Path(f"{self.output_file}_initial"))#.with_suffix(".nc"))
        print("\nNOTE: ", self.output_file_initial, self.output_file, "\n")
        return self

    @model_validator(mode='after')
    def check_interpolator_filename(self) -> Self:
        if self.interpolator_filename is not None and not self.use_cache:
            raise ValueError("If interpolator_filename is input, use_cache must be True.")
        return self

    @model_validator(mode="after")
    def add_output_file_extension(self) -> Self:
        if self.output_format is not None:
            if self.output_format == "netcdf":
                self.output_file = str(pathlib.Path(self.output_file).with_suffix(".nc"))
                self.output_file_initial = str(pathlib.Path(self.output_file_initial).with_suffix(".nc"))
            elif self.output_format == "parquet":
                self.output_file = str(pathlib.Path(self.output_file).with_suffix(".parquet"))
                self.output_file_initial = str(pathlib.Path(self.output_file_initial).with_suffix(".parquet"))
            else:
                raise ValueError(f"output_format {self.output_format} not recognized.")
        return self

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

    @model_validator(mode='after')
    def check_config_z_value(self) -> Self:
        if not self.seed_seafloor and self.z is None:
            raise ValueError("z needs a non-None value if seed_seafloor is False.")
        if self.seed_seafloor and self.z is not None:
            raise ValueError("z needs to be None if seed_seafloor is True.")
        return self

    @model_validator(mode='after')
    def check_config_time_parameters(self) -> Self:
        non_none_count = sum(x is not None for x in [self.start_time, self.end_time, self.duration, self.steps])
        if non_none_count != 2:
            raise ValueError(f"Exactly two of start_time, end_time, duration, and steps must be non-None. "
                             f"Current values are: start_time={self.start_time}, end_time={self.end_time}, "
                             f"duration={self.duration}, steps={self.steps}.")
        if self.start_time is None and self.end_time is None:
            raise ValueError("One of start_time or end_time must be non-None.")
        return self

    @model_validator(mode='after')
    def add_config_timedir(self) -> Self:
        if self.run_forward:
            self.timedir = 1
        else:
            self.timedir = -1
        return self

    @model_validator(mode='after')
    def calculate_config_times(self) -> Self:
        if self.duration is not None:
            self.steps = int(self.duration / datetime.timedelta(seconds=self.time_step))
            self.logger.info(f"Setting steps to {self.steps} based on duration.")
        elif self.end_time is not None and self.start_time is not None:
            self.steps = int(abs(self.end_time - self.start_time) / datetime.timedelta(seconds=self.time_step))
            self.logger.info(f"Setting steps to {self.steps} based on end_time and start_time.")

        if self.end_time is not None and self.start_time is not None:
            self.duration = abs(self.end_time - self.start_time)
            self.logger.info(f"Setting duration to {self.duration} based on end_time and start_time.")
        elif self.steps is not None:
            self.duration = self.steps * datetime.timedelta(seconds=self.time_step)
            self.logger.info(f"Setting duration to {self.duration} based on steps.")

        if self.steps is not None and self.start_time is not None:
            self.end_time = self.start_time + self.timedir * self.steps * datetime.timedelta(seconds=self.time_step)
            self.logger.info(f"Setting end_time to {self.end_time} based on start_time and steps.")
        elif self.duration is not None and self.start_time is not None:
            self.end_time = self.start_time + self.timedir * self.duration
            self.logger.info(f"Setting end_time to {self.end_time} based on start_time and duration.")
            
        if self.end_time is not None and self.steps is not None:
            self.start_time = self.end_time - self.timedir * self.steps * datetime.timedelta(seconds=self.time_step)
            self.logger.info(f"Setting start_time to {self.start_time} based on end_time and steps.")
        elif self.duration is not None and self.end_time is not None:
            self.start_time = self.end_time - self.timedir * self.duration
            self.logger.info(f"Setting start_time to {self.start_time} based on end_time and duration.")
        
        return self

    @model_validator(mode='after')
    def check_config_start_time(self) -> Self:
        min_model_time = self.model_fields[f"{self.ocean_model}_time_range"].metadata[0].ge
        min_model_time = parse(min_model_time) if isinstance(min_model_time, str) else min_model_time
        max_model_time = self.model_fields[f"{self.ocean_model}_time_range"].metadata[1].le
        max_model_time = parse(max_model_time) if isinstance(max_model_time, str) else max_model_time
        if not min_model_time <= self.start_time <= max_model_time:
            raise ValueError(f"start_time must be between {min_model_time} and {max_model_time}")
        self.logger.info(f"start_time is within the time range of the ocean model: {min_model_time} to {max_model_time}")
        return self

    @model_validator(mode='after')
    def check_config_do3D(self) -> Self:
        if not self.do3D and self.vertical_mixing:
            raise ValueError("If do3D is False, vertical_mixing must also be False.")
        return self

    @model_validator(mode='after')
    def define_config_oceanmodel_lon0_360(self) -> Self:
        if self.ocean_model == "NWGOA":
            self.oceanmodel_lon0_360 = True
            self.logger.info("Setting oceanmodel_lon0_360 to True for NWGOA model.")
        elif "CIOFS" in self.ocean_model:
            self.oceanmodel_lon0_360 = False
            self.logger.info("Setting oceanmodel_lon0_360 to False for all CIOFS models.")
        return self

    @model_validator(mode='after')
    def check_config_oceanmodel_lon0_360(self) -> Self:
        if self.oceanmodel_lon0_360:
            if self.lon is not None and self.lon < 0:
                if -180 < self.lon < 0:
                    orig_lon = self.lon
                    self.lon += 360
                    self.logger.info(f"Shifting longitude from {orig_lon} to {self.lon}.")
        return self

    @model_validator(mode='after')
    def check_config_lon_lat_model_domain(self) -> Self:
        keys = ["lon", "lat"]
        for key in keys:
            if getattr(self, key) is not None:
                min_model = self.model_fields[f"{self.ocean_model}_{key}_range"].metadata[0].ge
                min_model = parse(min_model) if isinstance(min_model, str) else min_model
                max_model = self.model_fields[f"{self.ocean_model}_{key}_range"].metadata[1].le
                max_model = parse(max_model) if isinstance(max_model, str) else max_model
                if not min_model <= getattr(self, key) <= max_model:
                    raise ValueError(f"{key} must be between {min_model} and {max_model}")
                self.logger.info(f"{key} is within the range of the ocean model: {min_model} to {max_model}")
        return self

    @model_validator(mode='after')
    def calculate_config_horizontal_diffusivity(self) -> Self:
        """Calculate horizontal diffusivity based on ocean model."""

        if self.horizontal_diffusivity is not None:
            self.logger.info(
                f"Setting horizontal_diffusivity to user-selected value {self.horizontal_diffusivity}."
            )

        elif self.ocean_model in _KNOWN_MODELS:

            hdiff = calc_known_horizontal_diffusivity(self.ocean_model)
            self.logger.info(
                f"Setting horizontal_diffusivity parameter to one tuned to reader model of value {hdiff}."
            )
            self.horizontal_diffusivity = hdiff

        elif (
            self.ocean_model not in _KNOWN_MODELS
            and self.horizontal_diffusivity is None
        ):

            self.logger.info(
                """Since ocean_model is user-input, changing horizontal_diffusivity parameter from None to 0.0.
                You can also set it to a specific value with `m.horizontal_diffusivity=[number]`."""
            )

            self.horizontal_diffusivity = 0

        return self
    
    # validators["assign_output_file_if_needed"] = assign_output_file_if_needed
    # validators["clean_output_file"] = clean_output_file
    validators["assign_output_file_initial"] = assign_output_file_initial
    validators["validate_interpolator_filename"] = check_interpolator_filename
    validators["add_output_file_extension"] = add_output_file_extension
    validators["check_config_seed_flag_elements"] = check_config_seed_flag_elements
    validators["check_config_seed_flag_geojson"] = check_config_seed_flag_geojson
    validators["check_config_z_value"] = check_config_z_value
    validators["check_config_time_parameters"] = check_config_time_parameters
    validators["add_config_timedir"] = add_config_timedir
    validators["calculate_config_times"] = calculate_config_times
    validators["check_config_start_time"] = check_config_start_time
    validators["check_config_do3D"] = check_config_do3D
    validators["define_config_oceanmodel_lon0_360"] = define_config_oceanmodel_lon0_360
    validators["check_config_oceanmodel_lon0_360"] = check_config_oceanmodel_lon0_360
    validators["check_config_lon_lat_model_domain"] = check_config_lon_lat_model_domain
    validators["calculate_config_horizontal_diffusivity"] = calculate_config_horizontal_diffusivity


def create_manager_pydantic_model(config: Dict[str, Any], add_special_fields_and_validators_func, config_name: str, other_inputs: dict) -> BaseModel:
    fields, validators = create_pydantic_model(config)
    add_special_fields_and_validators_func(fields, validators)
    # TODO: change extra forbid to an input so the model can forbid extras but PTM can pass them through?
    model = create_model(
        config_name, 
        # __base__=base_class,
                         __validators__=validators,
                         **other_inputs,
                        #  __config__=ConfigDict(use_enum_values=True),#, extra=Extra.forbid),
                         **fields
)
    return model




# def setup_ptm_config():
#     """Setup PTMConfig and OpenDriftConfig."""
#     # Read PTM configuration information
#     config_path = pathlib.Path(__file__).parent / "the_manager_config.json"
#     config_data = load_config(config_path)
#     PTMConfig = create_manager_pydantic_model(config_data, add_special_fields_and_validators_manager, "PTMConfig", dict(__config__=ConfigDict(use_enum_values=True)))
#     return PTMConfig
#     # # Read OpenDrift configuration file
#     # opendrift_config_path = pathlib.Path(__file__).parent / "models" / "opendrift" / "config.json"
#     # opendrift_config_data = load_config(opendrift_config_path)
#     # OpenDriftConfig = create_manager_pydantic_model(opendrift_config_data, add_special_fields_and_validators_opendrift, "OpenDriftConfig", {"__base__": PTMConfig})

#     # return PTMConfig, OpenDriftConfig, _KNOWN_MODELS

# PTMConfig = setup_ptm_config()
# _KNOWN_MODELS = PTMConfig.model_fields["ocean_model"].json_schema_extra["enum"]


# Read PTM configuration information
config_path = pathlib.Path(__file__).parent / "the_manager_config.json"
config_data = load_config(config_path)
PTMConfig = create_manager_pydantic_model(config_data, add_special_fields_and_validators_manager, "PTMConfig", dict(__config__=ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)))
_KNOWN_MODELS = PTMConfig.model_fields["ocean_model"].json_schema_extra["enum"]



## OpenDriftModel configuration ##

def add_special_fields_and_validators_opendrift(fields: Dict[str, Any], validators: Dict[str, Any]) -> None:
    """Define special fields and validators specific to OpenDrift configuration"""


    @model_validator(mode='after')
    def check_config_drift_model(self) -> Self:
        """Validators related to the drift_model."""
        if self.config.drift_model == "Leeway":
            if self.config.stokes_drift:
                raise ValueError("Stokes drift is not available with the Leeway drift model.")
            if self.config.do3D:
                raise ValueError("do3D must be False with the Leeway drift model.")
            if self.config.wind_drift_factor is not None:
                raise ValueError("wind_drift_factor cannot be used with the Leeway drift model. Instead it must be None.")
            if self.config.wind_drift_depth is not None:
                raise ValueError("wind_drift_depth cannot be used with the Leeway drift model. Instead it must be None.")

        elif self.config.drift_model == "LarvalFish":
            if not self.config.vertical_mixing:
                raise ValueError("Vertical mixing must be True with the LarvalFish drift model.")
            if not self.config.do3D:
                raise ValueError("do3D must be True with the LarvalFish drift model.")
            if self.config.wind_drift_factor is not None:
                raise ValueError("wind_drift_factor cannot be used with the LarvalFish drift model. Instead it must be None.")
            if self.config.wind_drift_depth is not None:
                raise ValueError("wind_drift_depth cannot be used with the LarvalFish drift model. Instead it must be None.")
            
        return self    






# Read OpenDrift configuration file
opendrift_config_path = pathlib.Path(__file__).parent / "models" / "opendrift" / "config.json"
opendrift_config_data = load_config(opendrift_config_path)
OpenDriftConfig = create_manager_pydantic_model(opendrift_config_data, add_special_fields_and_validators_opendrift, "OpenDriftConfig", {"__base__": PTMConfig})#, "__config__": ConfigDict(extra=Extra.forbid)})
# # Add a custom __init__ method for OpenDriftConfig
# def custom_init(self, **kwargs):
#     # Custom initialization logic
#     super(OpenDriftConfig, self).__init__(**kwargs)
#     # print(f"OpenDriftConfig initialized with param1={self.param1}, param2={self.param2}, additional_param={self.additional_param}")

# # Attach the custom __init__ to the dynamically created model
# OpenDriftConfig.__init__ = custom_init

