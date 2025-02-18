import json
import pathlib
from typing import Any, Dict, Optional, Union, ClassVar
from typing_extensions import Self
from pydantic import BaseModel, Field, create_model, model_validator, validator, field_validator, BeforeValidator, AfterValidator, ConfigDict, Extra
import datetime
import pandas as pd
import logging
import pathlib
import appdirs
import xarray as xr
# from dateparser import parse
from dateutil.parser import parse
from enum import Enum
from abc import ABC, abstractmethod

# from .config import load_config, create_pydantic_model
# from .cli import is_None

# def is_date(string, fuzzy=False):
#     """
#     Return whether the string can be interpreted as a date.
    
#     https://stackoverflow.com/a/25341965

#     :param string: str, string to check for date
#     :param fuzzy: bool, ignore unknown tokens in string if True
#     """
#     try: 
#         parse(string, fuzzy=fuzzy)
#         # return parse(string) is not None
#         return True

#     except ValueError:
#         return False

def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)

def generate_default_output_file():
    return f"output-results_{datetime.datetime.now():%Y-%m-%dT%H%M%SZ}"

def create_enum(name: str, values: list):
    return Enum(name, {v: v for v in values})

# class dateutil_parsed:
#     def __init__(self, datetimestr: str):
#         import pdb; pdb.set_trace()
#         self.datetimestr = parse(datetimestr)
    
#     # HERE: trying to use this custom type so I can use "now" and "in 48hours"
#     # in my config file and have them understood

def create_pydantic_model(config: Dict[str, Any]) -> BaseModel:
    
    # import pdb; pdb.set_trace()
    fields = {}
    validators = {}
    # datetime_fields = []
    
    # read in CIOFSOP max time available, at datetime object
    CIOFSOP_max = xr.open_dataset("/mnt/depot/data/packrat/prod/noaa/coops/ofs/aws_ciofs/processed/aws_ciofs_kerchunk.parq", engine="kerchunk").ocean_time[-1].values.astype('datetime64[s]').item()
    
    # TODO: ultimately include config from OpenDrift scenario into this?

    for key, value in config.items():
        field_type = value.get("type", "str")
        # default = value.get("default", None)
        # description = value.get("description", "")
        # ge = value.get("min", None)
        # le = value.get("max", None)
        # units = value.get("units", None)
        # ptm_level = value.get("ptm_level", None)
        # od_mapping = value.get("od_mapping", None)
        # enum = value.get("enum", None)


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
            # import pdb; pdb.set_trace()
            field_type = create_enum(enum_name, enum_values)
            # field_type = Optional[create_enum(enum_name, enum_values)]  # change back to this if want to use ocean_model=None
        elif field_type == "datetime.timedelta":
            field_type = Optional[datetime.timedelta]
        else:
            field_type = Optional[str]

        # bring metadat from config into pydantic model
        kwargs = dict()
        items = ["default", "description", "min", "max", "units", "ptm_level", "od_mapping", "enum"]
        for item in items:
            
            # TODO: can these be changed to validators?
            
            if value.get(item) is not None:

                # if value.get(item) == "datetime.datetime.now()":
                #     value[item] = (datetime.datetime.now()).isoformat()
                # elif value.get(item) == "datetime.datetime.now() + timedelta(hours=48)":
                #     value[item] = (datetime.datetime.now() + datetime.timedelta(hours=48)).isoformat()

                # if isinstance(value.get(item), str) and is_date(value.get(item)):
                #     # import pdb; pdb.set_trace()
                #     value[item] = parse_dates(value[item])
                #     # value[item] = datetime.datetime.fromisoformat(value[item])
                
                # search and replace "CIOFSOP_max"
                if isinstance(value.get(item), str) and value.get(item) == "CIOFSOP_max":
                    value[item] = CIOFSOP_max

                if item == "min":
                    kwargs["ge"] = value.get(item)
                elif item == "max":
                    kwargs["le"] = value.get(item)
                else:
                    kwargs[item] = value.get(item)
            else:
                kwargs[item] = None
        
        # use default_factory because the file name is uniquely generated per simulation if used
        if key == "output_file" and value["default"] is None:
            kwargs.pop("default")
            kwargs = dict(default_factory=generate_default_output_file, **kwargs)
        
        field_info = Field(**kwargs, validate_default=True)

        fields[key] = (field_type, field_info)

        # if field_type == Optional[Union[str, datetime.datetime]]:
        #     # validators[f'validate_{key}'] = AfterValidator(parse_dates)
        #     # validators[f'validate_{key}'] = field_validator(key, mode="after")#(parse_dates)
        #     datetime_fields.append(key)


    # @field_validator(*datetime_fields, mode='before')
    # # @classmethod
    # # def parse_dates(cls, v):
    # def parse_dates(v):
    #     import pdb; pdb.set_trace()
    #     if isinstance(v, str):
    #         return datetime.datetime.fromisoformat(v)
    #     return v

# def parse_dates(value):
#     if isinstance(value, str):
#         # return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")
#         return datetime.datetime.fromisoformat(value)
#         # return datetime.datetime(value)
#     return value

    # add timedir so it is available for definition from run_forward
    fields["timedir"] = (int, Field(default=1))
    
    # add oceanmodel_lon0_360 so it is available for definition from ocean_model
    fields["oceanmodel_lon0_360"] = (bool, Field(default=False))

    # model = create_model('PTMConfig', **fields)

    # for name, validator_func in validators.items():
    #     setattr(model, name, validator_func)


    # Add custom validators
    @model_validator(mode='after')
    def check_interpolator_filename(self) -> Self:
        if self.interpolator_filename is not None and not self.use_cache:
            raise ValueError("If interpolator_filename is input, use_cache must be True.")
        return self

    @model_validator(mode='after')
    def check_seed_flag_elements(self) -> Self:
        if self.seed_flag == "elements" and (self.lon is None or self.lat is None):
            raise ValueError("lon and lat need non-None values if using `seed_flag=\"elements\"`.")
        return self

    @model_validator(mode='after')
    def check_seed_flag_geojson(self) -> Self:
        if self.seed_flag == "geojson" and self.geojson is None:
            raise ValueError("geojson need non-None value if using `seed_flag=\"geojson\"`.")
        if self.seed_flag == "geojson" and (self.lon is not None or self.lat is not None):
            raise ValueError("lon and lat need to be None if using `seed_flag=\"geojson\"`.")
        return self

    @model_validator(mode='after')
    def check_z_value(self) -> Self:
        if not self.seed_seafloor and self.z is None:
            raise ValueError("z needs a non-None value if seed_seafloor is False.")
        if self.seed_seafloor and self.z is not None:
            raise ValueError("z needs to be None if seed_seafloor is True.")
        return self

    @model_validator(mode='after')
    def check_time_parameters(self) -> Self:
        non_none_count = sum(x is not None for x in [self.start_time, self.end_time, self.duration, self.steps])
        if non_none_count != 2:
            raise ValueError(f"Exactly two of start_time, end_time, duration, and steps must be non-None. "
                             f"Current values are: start_time={self.start_time}, end_time={self.end_time}, "
                             f"duration={self.duration}, steps={self.steps}.")
        # One of start_time or end_time must be not None
        if self.start_time is None and self.end_time is None:
            raise ValueError("One of start_time or end_time must be non-None.")
        return self

    @model_validator(mode='after')
    def add_timedir(self) -> Self:
        if self.run_forward:
            self.timedir = 1
        else:
            self.timedir = -1
        return self

    @model_validator(mode='after')
    def calculate_times(self) -> Self:
        if self.duration is not None:
            self.steps = int(self.duration / datetime.timedelta(seconds=self.time_step))
        elif self.end_time is not None and self.start_time is not None:
            self.steps = int(abs(self.end_time - self.start_time) / datetime.timedelta(seconds=self.time_step))

        if self.end_time is not None and self.start_time is not None:
            self.duration = abs(self.end_time - self.start_time)
        elif self.steps is not None:
            self.duration = self.steps * datetime.timedelta(seconds=self.time_step)

        if self.steps is not None and self.start_time is not None:
            # self.end_time = self.start_time + self.timedir * self.steps * f"00:00:{self.time_step}"
            self.end_time = self.start_time + self.timedir * self.steps * datetime.timedelta(seconds=self.time_step)
        elif self.duration is not None and self.start_time is not None:
            self.end_time = self.start_time + self.timedir * self.duration
            
        if self.end_time is not None and self.steps is not None:
            self.start_time = self.end_time - self.timedir * self.steps * datetime.timedelta(seconds=self.time_step)
        elif self.duration is not None and self.end_time is not None:
            self.start_time = self.end_time - self.timedir * self.duration
        
        return self

    @model_validator(mode='after')
    def check_start_time(self) -> Self:
        # make sure that start_time is in the time range given by the ocean model
        # the ocean_model time range is given by an input config of the ocean_model's name
        min_model_time = self.model_fields[f"{self.ocean_model}_time_range"].metadata[0].ge
        min_model_time = parse(min_model_time) if isinstance(min_model_time, str) else min_model_time
        max_model_time = self.model_fields[f"{self.ocean_model}_time_range"].metadata[1].le
        max_model_time = parse(max_model_time) if isinstance(max_model_time, str) else max_model_time
        if not min_model_time <= self.start_time <= max_model_time:
            raise ValueError(f"start_time must be between {min_model_time} and {max_model_time}")
        return self
            
    # TODO: remove surface_only flags from docs


    @model_validator(mode='after')
    def check_do3D(self) -> Self:
        if not self.do3D and self.vertical_mixing:
            raise ValueError("If do3D is False, vertical_mixing must also be False.")
        return self
    
    
    @model_validator(mode='after')
    def define_oceanmodel_lon0_360(self) -> Self:
        if self.ocean_model == "NWGOA":
            self.oceanmodel_lon0_360 = True
        elif "CIOFS" in self.ocean_model:
            self.oceanmodel_lon0_360 = False
        return self
    
    
    @model_validator(mode='after')
    def check_oceanmodel_lon0_360(self) -> Self:
        if self.oceanmodel_lon0_360:
            # this will only occur if seed_flag is "elements"
            if self.lon is not None and self.lon < 0:
                # move longitude to be 0 to 360 for this model
                if -180 < self.lon < 0:
                    self.lon += 360
        return self

    
    @model_validator(mode='after')
    def check_lon_lat_model_domain(self) -> Self:
        """this occurs after longitude is shifted if needed"""
        # make sure that lon and lat are in the ranges specified by the ocean model
        # the ocean_model time range is given by an input config of the ocean_model's name
        keys = ["lon", "lat"]
        for key in keys:
            if getattr(self, key) is not None:
                min_model = self.model_fields[f"{self.ocean_model}_{key}_range"].metadata[0].ge
                min_model = parse(min_model) if isinstance(min_model, str) else min_model
                max_model = self.model_fields[f"{self.ocean_model}_{key}_range"].metadata[1].le
                max_model = parse(max_model) if isinstance(max_model, str) else max_model
                if not min_model <= getattr(self, key) <= max_model:
                    raise ValueError(f"{key} must be between {min_model} and {max_model}")
        return self


    # TODO: make sure to maintain the possibility of inputting a user-defined ds
    # including setting oceanmodel_lon0_360

    # validators["validate_datetimes"] = parse_dates
    validators["validate_interpolator_filename"] = check_interpolator_filename
    validators["validate_seed_flag_elements"] = check_seed_flag_elements
    validators["validate_seed_flag_geojson"] = check_seed_flag_geojson
    validators["validate_z_value"] = check_z_value
    validators["validate_time_parameters"] = check_time_parameters
    validators["validate_timedir"] = add_timedir
    validators["validate_times"] = calculate_times
    validators["validate_start_time"] = check_start_time
    validators["validate_do3D"] = check_do3D
    validators["validate_oceanmodel_lon0_360"] = define_oceanmodel_lon0_360
    validators["validate_oceanmodel_lon0_360_lon"] = check_oceanmodel_lon0_360
    validators["validate_lon_lat_model_domain"] = check_lon_lat_model_domain

    model = create_model('PTMConfig', 
                         __validators__ = validators,
                        #  __config__ = ConfigDict(arbitrary_types_allowed=True),
                        __config__ = ConfigDict(use_enum_values=True, extra=Extra.forbid),  # https://stackoverflow.com/a/76973367
                         **fields)

    
    # TODO: ultimately include config from OpenDrift scenario into this?
    # TODO: add validator for start_time/end_time/duration/steps being in the model time range
    # TODO: add validator for start_time and ocean_model combination between correct time range

    return model

# def parse_dates(value):
#     if isinstance(value, str):
#         # return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")
#         return datetime.datetime.fromisoformat(value)
#         # return datetime.datetime(value)
#     return value


# _KNOWN_MODELS = [
#     "NWGOA",
#     "CIOFS",
#     "CIOFSFRESH",
#     "CIOFSOP",
# ]

# Read PTM configuration information
config_path = pathlib.Path(__file__).parent / "the_manager_config.json"
config_data = load_config(config_path)
PTMConfig = create_pydantic_model(config_data)
_KNOWN_MODELS = PTMConfig.model_fields["ocean_model"].json_schema_extra["enum"]

class ParticleTrackingManager:
    """Manager class that controls particle tracking model."""
    
    # TODO: update docs and tests to not demonstrate doing things in steps
    # since won't be able to anymore

    def __init__(self, **kwargs):
        # this inputs everything from PTMConfig to *.config.*
        self.config = PTMConfig(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.setup_logger()
        self.setup_interpolator()
        
        self.has_added_reader = False
        self.has_run_seeding = False
        self.has_run = False
        self.output_file_initial = None

    def setup_logger(self):
        """Setup logger."""
        output_file = self.config.output_file.replace(".nc", "").replace(".parquet", "").replace(".parq", "")
        logfile_name = output_file + ".log"
        self.file_handler = logging.FileHandler(logfile_name)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.info(f"filename: {logfile_name}")
        self.logfile_name = logfile_name

    def setup_interpolator(self):
        """Setup interpolator."""
        if self.config.use_cache:
            cache_dir = pathlib.Path(appdirs.user_cache_dir(appname="particle-tracking-manager", appauthor="axiom-data-science"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            if self.config.interpolator_filename is None:
                self.config.interpolator_filename = cache_dir / pathlib.Path(f"{self.config.ocean_model}_interpolator").with_suffix(".pickle")
            else:
                self.config.interpolator_filename = pathlib.Path(self.config.interpolator_filename).with_suffix(".pickle")
            self.save_interpolator = True
            
            # change interpolator_filename to string
            self.config.interpolator_filename = str(self.config.interpolator_filename)
            
            if pathlib.Path(self.config.interpolator_filename).exists():
                self.logger.info(f"Loading the interpolator from {self.config.interpolator_filename}.")
            else:
                self.logger.info(f"A new interpolator will be saved to {self.config.interpolator_filename}.")
        else:
            self.save_interpolator = False
            self.logger.info("Interpolators will not be saved.")
    
    # TODO: add abstractmethod decorator, remove add_reader (or which is not defined in opendrift) and change "has_added_reader" in that function
    # def run_add_reader(self, **kwargs):
    #     """Define in child class."""
    #     pass

    @abstractmethod
    def add_reader(self, **kwargs):
        """Here is where the model output is opened.
        
        Subclasses must implement this method and:
        
        * Set `self.has_added_reader = True` at the end of the method.
        """
        pass
        # self.run_add_reader(**kwargs)
        # self.has_added_reader = True
        
    # def run_seed(self):
    #     """Define in child class."""
    #     pass

    # TODO: change methods to _methods as appropriate

    @abstractmethod
    def seed(self, lon=None, lat=None, z=None):
    # def seed(self, lon=None, lat=None, z=None):
        """Initialize the drifters in space and time.
        
        Subclasses must implement this method and:
        
        * Raise a ValueError if not self.has_added_reader
        * Set `self.has_run_seeding = True` at the end of the method.
        """
        pass
        # for key in [lon, lat, z]:
        #     if key is not None:
        #         self.__setattr__(self, f"{key}", key)

        # if self.config.ocean_model is not None and not self.has_added_reader:
        #     self.add_reader()

        # if self.config.start_time is None:
        #     raise KeyError("first add reader with `manager.add_reader(**kwargs)` or input a start_time.")

        # self.run_seed()
        # self.has_run_seeding = True

    @abstractmethod
    def run(self):
        """Call model run function.
        
        Subclasses must implement this method and:
        
        * Raise a ValueError if not self.has_run_seeding
        * Set `self.has_run = True` at the end of the method.
        * Should close the file handler and logger.
        """
        pass
        # if not self.has_run_seeding:
        #     raise KeyError("first run seeding with `manager.seed()`.")

        # self.logger.info(f"start_time: {self.config.start_time}, end_time: {self.config.end_time}, steps: {self.config.steps}, duration: {self.config.duration}")

        # self.run_drifters()
        # self.logger.removeHandler(self.file_handler)
        # self.file_handler.close()
        # self.has_run = True

    def run_all(self):
        """Run all steps."""
        if not self.has_added_reader:
            self.add_reader()
        if not self.has_run_seeding:
            self.seed()
        if not self.has_run:
            self.run()

    def output(self):
        """Hold for future output function."""
        pass

    @abstractmethod
    def _config(self):
        """Model should have its own version which returns variable config."""
        pass

    @abstractmethod
    def _add_ptm_config(self):
        """Have this in the model class to modify config."""
        pass

    @abstractmethod
    def _add_model_config(self):
        """Have this in the model class to modify config."""
        pass

    # def _update_config(self) -> None:
    #     """Update configuration between model, PTM additions, and model additions."""
    #     self._add_ptm_config()
    #     self._add_model_config()

    @abstractmethod
    def show_config_model(self):
        """Define in child class."""
        pass

    def show_config(self, **kwargs) -> dict:
        """Show parameter configuration across both model and PTM."""
        # self._update_config()
        config = self.show_config_model(**kwargs)
        return config

    @abstractmethod
    def reader_metadata(self, key):
        """Define in child class."""
        pass

    @abstractmethod
    def query_reader(self):
        """Define in child class."""
        pass

    @abstractmethod
    def all_export_variables(self):
        """Output list of all possible export variables."""
        pass

    @abstractmethod
    def export_variables(self):
        """Output list of all actual export variables."""
        pass

    @property
    @abstractmethod
    def outfile_name(self):
        """Output file name."""
        pass