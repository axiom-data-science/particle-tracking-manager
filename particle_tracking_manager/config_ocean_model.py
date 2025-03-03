from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Self
from datetime import datetime
import xarray as xr
from .models.opendrift.utils import make_nwgoa_kerchunk, make_ciofs_kerchunk
import logging

logger = logging.getLogger(__name__)


def calculate_CIOFSOP_max():
    """read in CIOFSOP max time available, at datetime object"""
    return xr.open_dataset("/mnt/depot/data/packrat/prod/noaa/coops/ofs/aws_ciofs/processed/aws_ciofs_kerchunk.parq", engine="kerchunk").ocean_time[-1].values.astype('datetime64[s]').item()


class BaseOceanModel(BaseModel):
    """Contains functions and validators for all ocean models."""
    loc_local: dict = Field(default={}, exclude=True)
    ocean_model_local: bool = Field(True, description="Set to True to use local ocean model data, False for remote access.")
    end_time: datetime
    timedir: int
    
    def open_dataset(self, drop_vars: list) -> xr.Dataset:
        """Open an xarray dataset 
        
        """
        # if local
        if self.ocean_model_local:
            
            if self.loc_local is None:
                raise ValueError("loc_local must be set if ocean_model_local is True, but loc_local is None.")
            else:
                # TODO: Make a way to input chunks selection (and maybe other xarray kwargs)
                ds = xr.open_dataset(
                    self.loc_local,
                    engine="kerchunk",
                    # chunks={},  # Looks like it is faster not to include this for kerchunk
                    drop_variables=drop_vars,
                    decode_times=False,
                )

                logger.info(
                    f"Opened local dataset starting {self.start_time} and ending {self.end_time} with number outputs {ds.ocean_time.size}."
                )

        # otherwise remote
        else:
            if self.loc_remote is None:
                raise ValueError("loc_remote must be set if ocean_model_local is False, but loc_remote is None.")
            else:
                if ".nc" in self.loc_remote:
                    ds = xr.open_dataset(
                        self.loc_remote,
                        chunks={},
                        drop_variables=drop_vars,
                        decode_times=False,
                    )
                else:
                    ds = xr.open_zarr(
                        self.loc_remote,
                        chunks={},
                        drop_variables=drop_vars,
                        decode_times=False,
                    )

                logger.info(
                    f"Opened remote dataset {self.loc_remote} with number outputs {ds.ocean_time.size}."
                )
        return ds

    @model_validator(mode='after')
    def check_config_oceanmodel_lon0_360(self) -> Self:
        if self.oceanmodel_lon0_360:
            if self.lon is not None and self.lon < 0:
                if -180 < self.lon < 0:
                    orig_lon = self.lon
                    self.lon += 360
                    logger.info(f"Shifting longitude from {orig_lon} to {self.lon}.")
        return self

    @model_validator(mode='after')
    def make_loc_local(self) -> Self:
        """This sets up a short kerchunk file for reading in just enough model output."""
        # TODO: put this in each ocean model class
        if self.ocean_model_local:
            if self.ocean_model == "NWGOA":
                start_time = self.start_time
                start = f"{start_time.year}-{str(start_time.month).zfill(2)}-{str(start_time.day).zfill(2)}"
                end_time = self.end_time
                end = f"{end_time.year}-{str(end_time.month).zfill(2)}-{str(end_time.day).zfill(2)}"
                loc_local = make_nwgoa_kerchunk(start=start, end=end)

            elif self.ocean_model == "CIOFS":
                # TODO: check this for other years and see if need for other models
                start_time_doy = self.start_time.timetuple().tm_yday
                start_time_year = self.start_time.year
                end_time_doy = self.end_time.timetuple().tm_yday
                end_time_year = self.end_time.year
                # account for times at the beginning and end of the year, to make sure they have
                # enough output available
                start_dfile, end_dfile = -1, 1
                if (self.timedir == 1 and start_time_doy == 1) or (self.timedir == -1 and end_time_doy == 1):
                    if self.timedir == 1:
                        start_time_year -= 1
                        start_time_doy = 365
                        start_dfile = -1
                    elif self.timedir == -1:
                        end_time_year -= 1
                        end_time_doy = 365
                        end_dfile = -1
                if (self.timedir == 1 and end_time_doy == 365) or (self.timedir == -1 and start_time_doy == 365):
                    if self.timedir == 1:
                        end_time_year += 1
                        end_time_doy = 1
                        end_dfile = 1
                    elif self.timedir == -1:
                        start_time_year += 1
                        start_time_doy = 1
                        start_dfile = 1
                start = f"{start_time_year}_{str(start_time_doy + start_dfile).zfill(4)}"
                end = f"{end_time_year}_{str(end_time_doy + end_dfile).zfill(4)}"
                loc_local = make_ciofs_kerchunk(
                    start=start, end=end, name="ciofs"
                )

            elif self.ocean_model == "CIOFSFRESH":

                start = f"{self.start_time.year}_{str(self.start_time.timetuple().tm_yday - 1).zfill(4)}"

                end = f"{self.end_time.year}_{str(self.end_time.timetuple().tm_yday).zfill(4)}"
                loc_local = make_ciofs_kerchunk(
                    start=start, end=end, name="ciofs_fresh"
                )

            elif self.ocean_model == "CIOFSOP":

                start = f"{self.start_time.year}-{str(self.start_time.month).zfill(2)}-{str(self.start_time.day).zfill(2)}"
                end = f"{self.end_time.year}-{str(self.end_time.month).zfill(2)}-{str(self.end_time.day).zfill(2)}"

                loc_local = make_ciofs_kerchunk(
                    start=start, end=end, name="aws_ciofs_with_angle"
                )
            self.loc_local = loc_local
        else:
            self.loc_local = None
        return self



standard_name_mapping={
    "mask_rho": "mask_rho",
    "wetdry_mask_rho": "wetdry_mask_rho",
    "u_eastward": "u_eastward",
    "v_northward": "v_northward",
    "Uwind_eastward": "Uwind_eastward",
    "Vwind_northward": "Vwind_northward"
}
class SetupNWGOA(BaseOceanModel):
    start_time: datetime = Field(description="Time range of the ocean model", ge=datetime(1999,1,1,0,0,0), le=datetime(2009,1,1,0,0,0))
    lon: float = Field(-151, description="Longitude range of the ocean model", ge=199.66946652-360, le=220.02187714-360)
    lat: float = Field(58, description="Latitude range of the ocean model", ge=52.25975392, le=63.38656094)
    oceanmodel_lon0_360: bool = Field(True, description="Set to True to use 0-360 longitude convention for this model.")
    standard_name_mapping: Dict[str, str] = Field(standard_name_mapping, description="Mapping of model variable names to standard names.")
    model_drop_vars: List[str] = Field(["hice", "hraw", "snow_thick"], description="List of variables to drop from the model dataset. These variables are not needed for particle tracking.")
    loc_remote: str = Field("http://xpublish-nwgoa.srv.axds.co/datasets/nwgoa_all/zarr/", description="Remote location of the model dataset.")
    ocean_model: str = Field("NWGOA", description="Name of the ocean model.")


standard_name_mapping_CIOFS={
    "mask_rho": "land_binary_mask",
    "wetdry_mask_rho": "land_binary_mask",
    "u_eastward": "x_sea_water_velocity",
    "v_northward": "y_sea_water_velocity"
}
class SetupCIOFS(BaseOceanModel):
    start_time: datetime = Field(description="Time range of the ocean model", ge=datetime(1999,1,1,0,0,0), le=datetime(2023,1,1,0,0,0))
    lon: float = Field(-151, description="Longitude range of the ocean model", ge=-156.485291, le=-148.925125)
    lat: float = Field(58, description="Latitude range of the ocean model", ge=56.7004919, le=61.5247774)
    oceanmodel_lon0_360: bool = Field(False, description="Set to True to use 0-360 longitude convention for this model.")
    standard_name_mapping: Dict[str, str] = Field(standard_name_mapping_CIOFS, description="Mapping of model variable names to standard names.")
    model_drop_vars: List[str] = Field(["wetdry_mask_psi"], description="List of variables to drop from the model dataset. These variables are not needed for particle tracking.")
    loc_remote: str = Field("http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/", description="Remote location of the model dataset.")
    ocean_model: str = Field("CIOFS", description="Name of the ocean model.")

class SetupCIOFSOP(BaseOceanModel):
    start_time: datetime = Field(description="Time range of the ocean model", ge=datetime(2021,8,31,19,0,0))
    lon: float = Field(-151, description="Longitude range of the ocean model", ge=-156.485291, le=-148.925125)
    lat: float = Field(58, description="Latitude range of the ocean model", ge=56.7004919, le=61.5247774)
    oceanmodel_lon0_360: bool = Field(False, description="Set to True to use 0-360 longitude convention for this model.")
    standard_name_mapping: Dict[str, str] = Field(standard_name_mapping_CIOFS, description="Mapping of model variable names to standard names.")
    model_drop_vars: List[str] = Field(["wetdry_mask_psi"], description="List of variables to drop from the model dataset. These variables are not needed for particle tracking.")
    loc_remote: str = Field("https://thredds.aoos.org/thredds/dodsC/AWS_CIOFS.nc", description="Remote location of the model dataset.")
    ocean_model: str = Field("CIOFSOP", description="Name of the ocean model.")

    @model_validator(mode="before")
    def set_start_time_max(cls, values):
        """
        Set the maximum value for start_time based on the CIOFSOP_max field.
        """
        CIOFSOP_max = calculate_CIOFSOP_max()
        if CIOFSOP_max:
            # Dynamically update the `start_time` field's `le` (less than or equal) constraint
            # Note: you would adjust the `start_time` field to ensure the `le` constraint works dynamically.
            values['start_time'] = values.get('start_time', datetime(2021,8,31,19,0,0))  # default time if not provided
            if values['start_time'] > CIOFSOP_max:
                raise ValueError(f"start_time cannot be later than CIOFSOP_max: {CIOFSOP_max}")
        return values


ocean_model_mapper = {
    "NWGOA": SetupNWGOA,
    "CIOFS": SetupCIOFS,
    "CIOFSOP": SetupCIOFSOP
}

def select_ocean_model(start_time: datetime, end_time: datetime, timedir: int,
                       lon: float, lat: float, 
                       ocean_model: str, ocean_model_local: bool):
    """Select the ocean model based on the input parameters."""
    
    return ocean_model_mapper[ocean_model](start_time=start_time, end_time=end_time, timedir=timedir, lon=lon, lat=lat, ocean_model_local=ocean_model_local)
