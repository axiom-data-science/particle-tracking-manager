"""Register instances of ocean models here. This allows for temporary registrations too."""

# from .config_ocean_model import NWGOA, CIOFS, CIOFSOP, CIOFSFRESH
from enum import Enum
# from typing import Dict, Any
from typing import Callable, Dict, List, Optional
from datetime import datetime
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ValidationError
from .models.opendrift.utils import make_nwgoa_kerchunk, make_ciofs_kerchunk
import xarray as xr
from pathlib import Path
import yaml



## Set up ocean model configuration: doesn't depend on a tracking simulation. ##


def calculate_CIOFSOP_max():
    """read in CIOFSOP max time available, at datetime object"""
    return xr.open_dataset("/mnt/depot/data/packrat/prod/noaa/coops/ofs/aws_ciofs/processed/aws_ciofs_kerchunk.parq", engine="kerchunk").ocean_time[-1].values.astype('datetime64[s]').item()


def get_model_end_time(name) -> datetime:
    # This is only run when the property is requested
    if name == "CIOFSOP":
        return calculate_CIOFSOP_max()
    else:
        raise NotImplementedError(f"get_model_end_time not implemented for {name}.")

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


class OceanModelConfig(BaseModel):
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
        Field(description="Set to True to use 0-360 longitude convention for this model."),
    ]
    standard_name_mapping: Annotated[
        Dict[str, str],
        Field(description="Mapping of model variable names to standard names."),
    ]
    model_drop_vars: Annotated[
        List[str],
        Field(description="List of variables to drop from the model dataset. These variables are not needed for particle tracking."),
    ]
    loc_remote: Annotated[
        Optional[str],
        Field(description="Remote location of the model dataset."),
    ]
    dx: Annotated[
        float,
        Field(description="Approximate horizontal grid resolution (meters), used to calculate horizontal diffusivity."),
    ]
    
    end_time_fixed: Annotated[Optional[datetime], Field(None, description="End time of the model, if doesn't change.")]

    kerchunk_func_str: Annotated[
        str,
        Field(description="Name of function to create a kerchunk file for the model, mapped to function name in function_map."),
    ]

    @property
    def end_time_model(self) -> datetime:
        if self.end_time_fixed:
            return self.end_time_fixed
        else:  # there is only one that uses this currently
            return get_model_end_time(self.name)

    @property
    def horizontal_diffusivity(self) -> float:
        """Calculate horizontal diffusivity based on known ocean_model.
        
        Might be overwritten by user-input in other model config.
        """

        # horizontal diffusivity is calculated based on the mean horizontal grid resolution
        # for the model being used.
        # 0.1 is a guess for the magnitude of velocity being missed in the models, the sub-gridscale velocity
        sub_gridscale_velocity = 0.1
        horizontal_diffusivity = sub_gridscale_velocity * self.dx
        return horizontal_diffusivity




# standard_name_mapping={
#     "mask_rho": "mask_rho",
#     "wetdry_mask_rho": "wetdry_mask_rho",
#     "u_eastward": "x_sea_water_velocity",
#     "v_northward": "y_sea_water_velocity",
#     "Uwind_eastward": "x_wind",
#     "Vwind_northward": "y_wind",
# }

# NWGOA = OceanModelConfig(
#     name="NWGOA",
#     loc_remote="http://xpublish-nwgoa.srv.axds.co/datasets/nwgoa_all/zarr/",
#     temporal_resolution_str="PT1H",
#     lon_min=199.66946652-360,
#     lon_max=220.02187714-360,
#     lat_min=52.25975392,
#     lat_max=63.38656094,
#     start_time_model=datetime(1999,1,1,0,0,0),
#     end_time_fixed=datetime(2009,1,1,0,0,0),
#     oceanmodel_lon0_360=True,
#     standard_name_mapping=standard_name_mapping,
#     model_drop_vars=["hice", "hraw", "snow_thick"],
#     dx=1500,
#     kerchunk_func_str="make_nwgoa_kerchunk"
# )

# standard_name_mapping_CIOFS={
#     "mask_rho": "land_binary_mask",
#     "wetdry_mask_rho": "land_binary_mask",
#     "u_eastward": "x_sea_water_velocity",
#     "v_northward": "y_sea_water_velocity"
# }

# CIOFS = OceanModelConfig(
#     name="CIOFS",
#     loc_remote="http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/",
#     temporal_resolution_str="PT1H",
#     lon_min=-156.485291,
#     lon_max=-148.925125,
#     lat_min=56.7004919,
#     lat_max=61.5247774,
#     start_time_model=datetime(1999,1,1,0,0,0),
#     end_time_fixed=datetime(2023,1,1,0,0,0),
#     oceanmodel_lon0_360=False,
#     standard_name_mapping=standard_name_mapping_CIOFS,
#     model_drop_vars=["wetdry_mask_psi"],
#     dx=100,
#     kerchunk_func_str="make_ciofs_kerchunk"
# )

# CIOFSOP = OceanModelConfig(
#     name="CIOFSOP",
#     loc_remote="https://thredds.aoos.org/thredds/dodsC/AWS_CIOFS.nc",
#     temporal_resolution_str="PT1H",
#     lon_min=-156.485291,
#     lon_max=-148.925125,
#     lat_min=56.7004919,
#     lat_max=61.5247774,
#     start_time_model=datetime(2021,8,31,19,0,0),
#     oceanmodel_lon0_360=False,
#     standard_name_mapping=standard_name_mapping_CIOFS,
#     model_drop_vars=["wetdry_mask_psi"],
#     dx=100,
#     kerchunk_func_str="make_ciofs_kerchunk"
# )

# CIOFSFRESH = OceanModelConfig(
#     name="CIOFSFRESH",
#     loc_remote=None,
#     temporal_resolution_str="PT1H",
#     lon_min=-156.485291,
#     lon_max=-148.925125,
#     lat_min=56.7004919,
#     lat_max=61.5247774,
#     start_time_model=datetime(2003,1,1,0,0,0),
#     end_time_fixed=datetime(2016,1,1,0,0,0),
#     oceanmodel_lon0_360=False,
#     standard_name_mapping=standard_name_mapping_CIOFS,
#     model_drop_vars=["wetdry_mask_psi"],
#     dx=100,
#     kerchunk_func_str="make_ciofs_kerchunk"
# )



class OceanModelRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name, instance):
        self._registry[name] = instance

    def get(self, name):
        return self._registry.get(name)
    
    def get_all(self):
        return self._registry.items()
    
    def all(self):
        return list(self._registry.keys())
    
    def all_models(self):
        return list(self._registry.values())
    
    
    
# Directory with YAML files
directory = Path(__file__).resolve().parent / 'ocean_models'  # This is the directory where the current script is located

# directory = Path('ocean_models')

ocean_model_registry = OceanModelRegistry()

# Iterate through all .yaml files in the directory
for file_path in directory.glob('*.yaml'):
    with open(file_path, 'r') as f:
        config_data = yaml.safe_load(f)[file_path.stem]
        
        # Assuming your config_data needs to be loaded into a Pydantic model
        # Create the OceanModelConfig instance from the data
        config = OceanModelConfig(**config_data)
        
        # Register the configuration, perhaps by its name
        ocean_model_registry.register(config.name, config)

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

# # Access the registered instance
# retrieved_obj = registry.get("First Object")
# print(retrieved_obj.name)  # Output: First Object



# class OceanModelEnum(str, Enum):
#     NWGOA = "NWGOA"
#     CIOFS = "CIOFS"
#     CIOFSOP = "CIOFSOP"
#     CIOFSFRESH = "CIOFSFRESH"

# _KNOWN_MODELS = [model.value for model in OceanModelEnum]
