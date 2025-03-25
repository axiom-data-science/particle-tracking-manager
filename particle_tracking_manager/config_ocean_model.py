from pydantic import BaseModel, Field, model_validator, create_model
from typing import Optional, List, Dict, Self, Annotated, Callable
from datetime import datetime, timedelta
import xarray as xr
from .models.opendrift.utils import make_nwgoa_kerchunk, make_ciofs_kerchunk
import logging
from enum import Enum

logger = logging.getLogger()




## Set up ocean model configuration: doesn't depend on a tracking simulation. ##

class OceanModelEnum(str, Enum):
    NWGOA = "NWGOA"
    CIOFS = "CIOFS"
    CIOFSOP = "CIOFSOP"
    CIOFSFRESH = "CIOFSFRESH"

_KNOWN_MODELS = [model.value for model in OceanModelEnum]


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




standard_name_mapping={
    "mask_rho": "mask_rho",
    "wetdry_mask_rho": "wetdry_mask_rho",
    "u_eastward": "u_eastward",
    "v_northward": "v_northward",
    "Uwind_eastward": "Uwind_eastward",
    "Vwind_northward": "Vwind_northward"
}

NWGOA = OceanModelConfig(
    name="NWGOA",
    loc_remote="http://xpublish-nwgoa.srv.axds.co/datasets/nwgoa_all/zarr/",
    temporal_resolution_str="PT1H",
    lon_min=199.66946652-360,
    lon_max=220.02187714-360,
    lat_min=52.25975392,
    lat_max=63.38656094,
    start_time_model=datetime(1999,1,1,0,0,0),
    end_time_fixed=datetime(2009,1,1,0,0,0),
    oceanmodel_lon0_360=True,
    standard_name_mapping=standard_name_mapping,
    model_drop_vars=["hice", "hraw", "snow_thick"],
    dx=1500,
    kerchunk_func_str="make_nwgoa_kerchunk"
)

standard_name_mapping_CIOFS={
    "mask_rho": "land_binary_mask",
    "wetdry_mask_rho": "land_binary_mask",
    "u_eastward": "x_sea_water_velocity",
    "v_northward": "y_sea_water_velocity"
}

CIOFS = OceanModelConfig(
    name="CIOFS",
    loc_remote="http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/",
    temporal_resolution_str="PT1H",
    lon_min=-156.485291,
    lon_max=-148.925125,
    lat_min=56.7004919,
    lat_max=61.5247774,
    start_time_model=datetime(1999,1,1,0,0,0),
    end_time_fixed=datetime(2023,1,1,0,0,0),
    oceanmodel_lon0_360=False,
    standard_name_mapping=standard_name_mapping_CIOFS,
    model_drop_vars=["wetdry_mask_psi"],
    dx=100,
    kerchunk_func_str="make_ciofs_kerchunk"
)

CIOFSOP = OceanModelConfig(
    name="CIOFSOP",
    loc_remote="https://thredds.aoos.org/thredds/dodsC/AWS_CIOFS.nc",
    temporal_resolution_str="PT1H",
    lon_min=-156.485291,
    lon_max=-148.925125,
    lat_min=56.7004919,
    lat_max=61.5247774,
    start_time_model=datetime(2021,8,31,19,0,0),
    oceanmodel_lon0_360=False,
    standard_name_mapping=standard_name_mapping_CIOFS,
    model_drop_vars=["wetdry_mask_psi"],
    dx=100,
    kerchunk_func_str="make_ciofs_kerchunk"
)

CIOFSFRESH = OceanModelConfig(
    name="CIOFSFRESH",
    loc_remote=None,
    temporal_resolution_str="PT1H",
    lon_min=-156.485291,
    lon_max=-148.925125,
    lat_min=56.7004919,
    lat_max=61.5247774,
    start_time_model=datetime(2003,1,1,0,0,0),
    end_time_fixed=datetime(2016,1,1,0,0,0),
    oceanmodel_lon0_360=False,
    standard_name_mapping=standard_name_mapping_CIOFS,
    model_drop_vars=["wetdry_mask_psi"],
    dx=100,
    kerchunk_func_str="make_ciofs_kerchunk"
)




ocean_model_mapper = {
    "NWGOA": NWGOA,
    "CIOFS": CIOFS,
    "CIOFSOP": CIOFSOP,
    "CIOFSFRESH": CIOFSFRESH,
}



## Set up ocean model simulation configuration: depends on a tracking simulation. ##

class OceanModelSimulation(BaseModel):
    ocean_model_config: OceanModelConfig
    oceanmodel_lon0_360: bool
    ocean_model_local: bool
    
    
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
ocean_model_simulation_mapper = {}
for ocean_model in ocean_model_mapper.values():# [NWGOA, CIOFS, CIOFSOP, CIOFSFRESH]:
    ocean_model_name = ocean_model.name
    simulation_model = create_model(
        ocean_model_name,  # Model name
        __base__=OceanModelSimulation,
        lon=(float, Field(..., ge=getattr(ocean_model, "lon_min"), le=getattr(ocean_model, "lon_max"), description="Longitude of the simulation within the model bounds.")),
        lat=(float, Field(..., ge=getattr(ocean_model, "lat_min"), le=getattr(ocean_model, "lat_max"), description="Latitude of the simulation within the model bounds.")),
        start_time=(datetime, Field(..., ge=getattr(ocean_model, "start_time_model"), le=getattr(ocean_model, "end_time_model"), description="Start time of the simulation.")),
        end_time=(datetime, Field(..., ge=getattr(ocean_model, "start_time_model"), le=getattr(ocean_model, "end_time_model"), description="End time of the simulation.")),
    )
    ocean_model_simulation_mapper[ocean_model_name] = simulation_model


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
    
    