from pydantic import BaseModel, Field, model_validator, create_model
from typing import Optional, List, Dict, Self, Annotated, Callable
from datetime import datetime, timedelta
import xarray as xr
from .ocean_model_registry import ocean_model_registry, OceanModelConfig
from .models.opendrift.utils import make_nwgoa_kerchunk, make_ciofs_kerchunk
import logging
from enum import Enum

logger = logging.getLogger()



# Define a function to generate an Enum from the registry
def generate_enum_from_registry() -> Enum:
    """Generate an Enum class from the ocean model registry."""
    # Dynamically create an Enum class using the registry data
    enum_name = "OceanModelEnum"
    enum_members = {name: name for name in ocean_model_registry.all()}
    return Enum(enum_name, enum_members)


# Generate a dynamic Enum using the registry data
OceanModelEnum = generate_enum_from_registry()
# ## Set up ocean model configuration: doesn't depend on a tracking simulation. ##


# def calculate_CIOFSOP_max():
#     """read in CIOFSOP max time available, at datetime object"""
#     return xr.open_dataset("/mnt/depot/data/packrat/prod/noaa/coops/ofs/aws_ciofs/processed/aws_ciofs_kerchunk.parq", engine="kerchunk").ocean_time[-1].values.astype('datetime64[s]').item()


# def get_model_end_time(name) -> datetime:
#     # This is only run when the property is requested
#     if name == "CIOFSOP":
#         return calculate_CIOFSOP_max()
#     else:
#         raise NotImplementedError(f"get_model_end_time not implemented for {name}.")

# def get_file_date_string(name: str, date: datetime) -> str:
#     if name == "NWGOA":
#         return f"{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}"
#     elif name == "CIOFSOP":
#         return f"{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}"
#     elif name == "CIOFS":
#         return f"{date.year}_{str(date.timetuple().tm_yday - 1).zfill(4)}"
#     elif name == "CIOFSFRESH":
#         return f"{date.year}_{str(date.timetuple().tm_yday - 1).zfill(4)}"

# function_map: Dict[str, Callable[[int, int], int]] = {
#     'make_nwgoa_kerchunk': make_nwgoa_kerchunk,
#     'make_ciofs_kerchunk': make_ciofs_kerchunk,
# }

# class OceanModelRegistry:
#     def __init__(self):
#         self._registry = {}

#     def register(self, name, instance):
#         self._registry[name] = instance

#     def get(self, name):
#         return self._registry.get(name)
    
#     def all_models(self):
#         return self._registry.values()
    
#     def all(self):
#         return self._registry.keys()

# # class MyClass:
# #     def __init__(self, name):
# #         self.name = name

# # Create an instance of the registry
# ocean_model_registry = OceanModelRegistry()

# # # Create an instance of MyClass
# # obj1 = MyClass("First Object")

# # Register the known model instances
# ocean_model_registry.register(NWGOA.name, NWGOA)
# ocean_model_registry.register(CIOFS.name, CIOFS)
# ocean_model_registry.register(CIOFSOP.name, CIOFSOP)
# ocean_model_registry.register(CIOFSFRESH.name, CIOFSFRESH)

# # # Access the registered instance
# # retrieved_obj = registry.get("First Object")
# # print(retrieved_obj.name)  # Output: First Object



# class OceanModelEnum(str, Enum):
#     NWGOA = "NWGOA"
#     CIOFS = "CIOFS"
#     CIOFSOP = "CIOFSOP"
#     CIOFSFRESH = "CIOFSFRESH"

# _KNOWN_MODELS = [model.value for model in OceanModelEnum]



# ocean_model_mapper = {
#     "NWGOA": NWGOA,
#     "CIOFS": CIOFS,
#     "CIOFSOP": CIOFSOP,
#     "CIOFSFRESH": CIOFSFRESH,
# }



## Set up ocean model simulation configuration: depends on a tracking simulation. ##

class OceanModelSimulation(BaseModel):
    # ocean_model_config: OceanModelConfig
    # oceanmodel_lon0_360: bool
    ocean_model_local: bool
    
    model_config = {
        "validate_defaults": True,
        "use_enum_values": True,
        "extra": "forbid",
    }
    
    
    @model_validator(mode='after')
    def check_config_oceanmodel_lon0_360(self) -> Self:
        print("RUNNING VALIDATOR")
        if self.ocean_model_config.oceanmodel_lon0_360:
            if self.lon is not None and self.lon < 0:
                if -180 < self.lon < 0:
                    orig_lon = self.lon
                    self.lon += 360
                    logger.info(f"Shifting longitude from {orig_lon} to {self.lon}.")
        return self

    def open_dataset(self, drop_vars: list) -> xr.Dataset:
        """Open an xarray dataset 
        
        """
        # if local
        if self.ocean_model_local:
            
            name, kerchunk_func_str = self.ocean_model_config.name, self.ocean_model_config.kerchunk_func_str
            start_time, end_time = self.start_time, self.end_time
            
            if loc_local(name, kerchunk_func_str, start_time, end_time) is None:
                raise ValueError("loc_local must be set if ocean_model_local is True, but loc_local is None.")
            else:
                # TODO: Make a way to input chunks selection (and maybe other xarray kwargs)
                ds = xr.open_dataset(
                    loc_local(name, kerchunk_func_str, start_time, end_time),
                    engine="kerchunk",
                    # chunks={},  # Looks like it is faster not to include this for kerchunk
                    drop_variables=drop_vars,
                    decode_times=False,
                )
                logger.info(
                    f"Opened local dataset with start time {start_time} and end time {end_time} and number outputs {ds.ocean_time.size}."
                )

        # otherwise remote
        else:
            if self.ocean_model_config.loc_remote is None:
                raise ValueError("loc_remote must be set if ocean_model_local is False, but loc_remote is None.")
            else:
                if ".nc" in self.ocean_model_config.loc_remote:
                    ds = xr.open_dataset(
                        self.ocean_model_config.loc_remote,
                        chunks={},
                        drop_variables=drop_vars,
                        decode_times=False,
                    )
                else:
                    ds = xr.open_zarr(
                        self.ocean_model_config.loc_remote,
                        chunks={},
                        drop_variables=drop_vars,
                        decode_times=False,
                    )

                logger.info(
                    f"Opened remote dataset {self.ocean_model_config.loc_remote} with number outputs {ds.ocean_time.size}."
                )
        return ds


# Using `create_model` to generate a dynamic simulation model class
# ocean_model = NWGOA
# NWGOASimulation = create_model(
#     ocean_model.name,  # Model name
#     __base__=OceanModelSimulation,
#     lon=(float, Field(..., ge=getattr(ocean_model, "lon_min"), le=getattr(ocean_model, "lon_max"), description="Longitude of the simulation within the model bounds.")),
#     lat=(float, Field(..., ge=getattr(ocean_model, "lat_min"), le=getattr(ocean_model, "lat_max"), description="Latitude of the simulation within the model bounds.")),
#     start_time=(datetime, Field(..., ge=getattr(ocean_model, "start_time_model"), le=getattr(ocean_model, "end_time_model"), description="Start time of the simulation.")),
#     end_time=(datetime, Field(..., ge=getattr(ocean_model, "start_time_model"), le=getattr(ocean_model, "end_time_model"), description="End time of the simulation.")),
# )
# ocean_model_simulation_mapper = {}
# for ocean_model in ocean_model_registry.all_models():# [NWGOA, CIOFS, CIOFSOP, CIOFSFRESH]:
# # for ocean_model in ocean_model_mapper.values():# [NWGOA, CIOFS, CIOFSOP, CIOFSFRESH]:
#     ocean_model_name = ocean_model.name
#     simulation_model = create_model(
#         ocean_model_name,  # Model name
#         __base__=OceanModelSimulation,
#         lon=(Optional[float], Field(..., ge=getattr(ocean_model, "lon_min"), le=getattr(ocean_model, "lon_max"), description="Longitude of the simulation within the model bounds.")),
#         lat=(Optional[float], Field(..., ge=getattr(ocean_model, "lat_min"), le=getattr(ocean_model, "lat_max"), description="Latitude of the simulation within the model bounds.")),
#         start_time=(datetime, Field(..., ge=getattr(ocean_model, "start_time_model"), le=getattr(ocean_model, "end_time_model"), description="Start time of the simulation.")),
#         end_time=(datetime, Field(..., ge=getattr(ocean_model, "start_time_model"), le=getattr(ocean_model, "end_time_model"), description="End time of the simulation.")),
#         ocean_model_config=(OceanModelConfig, ocean_model)
#     )
#     ocean_model_simulation_mapper[ocean_model_name] = simulation_model


def create_ocean_model_simulation(ocean_model: OceanModelConfig) -> OceanModelSimulation:
    """Create an ocean model simulation object."""
    ocean_model_name = ocean_model.name
    simulation_model = create_model(
        ocean_model_name,  # Model name
        __base__=OceanModelSimulation,
        lon=(Optional[float], Field(..., ge=getattr(ocean_model, "lon_min"), le=getattr(ocean_model, "lon_max"), description="Longitude of the simulation within the model bounds.")),
        lat=(Optional[float], Field(..., ge=getattr(ocean_model, "lat_min"), le=getattr(ocean_model, "lat_max"), description="Latitude of the simulation within the model bounds.")),
        start_time=(datetime, Field(..., ge=getattr(ocean_model, "start_time_model"), le=getattr(ocean_model, "end_time_model"), description="Start time of the simulation.")),
        end_time=(datetime, Field(..., ge=getattr(ocean_model, "start_time_model"), le=getattr(ocean_model, "end_time_model"), description="End time of the simulation.")),
        ocean_model_config=(OceanModelConfig, ocean_model)
    )
    return simulation_model    

ocean_model_simulation_mapper = {}
for ocean_model in ocean_model_registry.all_models():# [NWGOA, CIOFS, CIOFSOP, CIOFSFRESH]:
    ocean_model_simulation_mapper[ocean_model.name] = create_ocean_model_simulation(ocean_model)

# def create_ocean_model_simulation_mapper


# ocean_model_simulation_mapper = create_ocean_model_simulation_mapper()



def get_file_date_string(name: str, date: datetime) -> str:
    if name == "NWGOA":
        return f"{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}"
    elif name == "CIOFSOP":
        return f"{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}"
    elif name == "CIOFS":
        return f"{date.year}_{str(date.timetuple().tm_yday - 1).zfill(4)}"
    elif name == "CIOFSFRESH":
        return f"{date.year}_{str(date.timetuple().tm_yday - 1).zfill(4)}"


function_map: Dict[str, Callable[[int, int], int]] = {
    'make_nwgoa_kerchunk': make_nwgoa_kerchunk,
    'make_ciofs_kerchunk': make_ciofs_kerchunk,
}

def loc_local(name, kerchunk_func_str, start_sim, end_sim) -> dict:
    """This sets up a short kerchunk file for reading in just enough model output."""
    
    # back each start time back 1 day and end time forward 1 day to make sure enough output is available
    if start_sim < end_sim:
        start_time = start_sim - timedelta(days=1)
        end_time = end_sim + timedelta(days=1)
    else:
        start_time = start_sim + timedelta(days=1)
        end_time = end_sim - timedelta(days=1)
    
    start = get_file_date_string(name, start_time)
    end = get_file_date_string(name, end_time)
    loc_local = function_map[kerchunk_func_str](start=start, end=end, name=name)
    return loc_local
    
    
    
def register_on_the_fly(ds_info: dict) -> None:
    """Register a new ocean model on the fly.
    
    It has to be called "ONTHEFLY".
    
    ds_info can contain any of the OceanModelConfig fields.
    """

    # Update the "ONTHEFLY" user ocean model template with user dataset information
    ocean_model_registry.update_model("ONTHEFLY", ds_info)

    # Create the ocean model simulation object for the new ocean model
    ONTHEFLYSimulation = create_ocean_model_simulation(ocean_model_registry.get("ONTHEFLY"))

    # Update the ocean model simulation mapper with the new ocean model simulation
    ocean_model_simulation_mapper.update({"ONTHEFLY": ONTHEFLYSimulation})

    logger.info("Registered new ocean model")