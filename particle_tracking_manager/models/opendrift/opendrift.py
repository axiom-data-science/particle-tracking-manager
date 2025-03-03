"""Using OpenDrift for particle tracking."""
import copy
import datetime
import gc
import json
import logging
import os
import platform
import tempfile
from pathlib import Path
from typing import Optional, Union

import appdirs
import pandas as pd
import xarray as xr
from opendrift.readers import reader_ROMS_native

# from ...config_replacement import OpenDriftConfig
from .config_opendrift import OpenDriftConfig
from ...the_manager import ParticleTrackingManager
from ...config_logging import LoggerMethods
from .plot import check_plots, make_plots
from .utils import make_ciofs_kerchunk, make_nwgoa_kerchunk, narrow_dataset_to_simulation_time, \
    apply_known_ocean_model_specific_changes, apply_user_input_ocean_model_specific_changes


logger = logging.getLogger()
# logger = logging.getLogger(__name__)

class OpenDriftModel(ParticleTrackingManager):
    """Open drift particle tracking model.

    Defaults all come from config_model configuration file.

    Parameters
    ----------
    drift_model : str, optional
        Options: "OceanDrift", "LarvalFish", "OpenOil", "Leeway", by default "OceanDrift"
    export_variables : list, optional
        List of variables to export, by default None. See PTM docs for options.
    radius : int, optional
        Radius around each lon-lat pair, within which particles will be randomly seeded. This is used by function `seed_elements`.
    radius_type : str
        If 'gaussian' (default), the radius is the standard deviation in x-y-directions. If 'uniform', elements are spread evenly and always inside a circle with the given radius. This is used by function `seed_elements`.
    current_uncertainty : float
        Add gaussian perturbation with this standard deviation to current components at each time step.
    wind_uncertainty : float
        Add gaussian perturbation with this standard deviation to wind components at each time step.
    use_auto_landmask : bool
        Set as True to use general landmask instead of that from ocean_model.
        Use for testing primarily. Default is False.
    diffusivitymodel : str
        Algorithm/source used for profile of vertical diffusivity. Environment means that diffusivity is acquired from readers or environment constants/fallback. Turned on if ``vertical_mixing==True``.
    stokes_drift : bool, optional
        # TODO: move this to the relevant validator
        Set to True to turn on Stokes drift, by default True. This enables 3 settings in OpenDrift:

        * o.set_config('drift:use_tabularised_stokes_drift', True)
        * o.set_config('drift:tabularised_stokes_drift_fetch', '25000')  # default
        * o.set_config('drift:stokes_drift_profile', 'Phillips')  # default

        The latter two configurations are not additionally set in OpenDriftModel since they are already the default once stokes_drift is True.
    mixed_layer_depth : float
        Fallback value for ocean_mixed_layer_thickness if not available from any reader. This is used in the calculation of vertical diffusivity.
    coastline_action : str, optional
        Action to perform if a drifter hits the coastline, by default "previous". Options
        are 'stranding', 'previous'.
    seafloor_action : str, optional
        Action to perform if a drifter hits the seafloor, by default "deactivate". Options
        are 'deactivate', 'previous', 'lift_to_seafloor'.
    max_speed : int
        Typical maximum speed of elements, used to estimate reader buffer size.
    wind_drift_depth : float
        The direct wind drift (windage) is linearly decreasing from the surface value (wind_drift_factor) until 0 at this depth.
    vertical_mixing_timestep : float
        Time step used for inner loop of vertical mixing.
    object_type: str = config_model["object_type"]["default"],
        Leeway object category for this simulation.

    diameter : float
        Seeding value of diameter.
    neutral_buoyancy_salinity : float
        Seeding value of neutral_buoyancy_salinity.
    stage_fraction : float
        Seeding value of stage_fraction.
    hatched : float
        Seeding value of hatched.
    length : float
        Seeding value of length.
    weight : float
        Seeding value of weight.

    oil_type : str
        Oil type to be used for the simulation, from the NOAA ADIOS database.
    m3_per_hour : float
        The amount (volume) of oil released per hour (or total amount if release is instantaneous).
    oil_film_thickness : float
        Seeding value of oil_film_thickness.
    droplet_size_distribution : str
        Droplet size distribution used for subsea release.
    droplet_diameter_mu : float
        The mean diameter of oil droplet for a subsea release, used in normal/lognormal distributions.
    droplet_diameter_sigma : float
        The standard deviation in diameter of oil droplet for a subsea release, used in normal/lognormal distributions.
    droplet_diameter_min_subsea : float
        The minimum diameter of oil droplet for a subsea release, used in uniform distribution.
    droplet_diameter_max_subsea : float
        The maximum diameter of oil droplet for a subsea release, used in uniform distribution.
    emulsification : bool
        Surface oil is emulsified, i.e. water droplets are mixed into oil due to wave mixing, with resulting increase of viscosity.
    dispersion : bool
        Oil is removed from simulation (dispersed), if entrained as very small droplets.
    evaporation : bool
        Surface oil is evaporated.
    update_oilfilm_thickness : bool
        Oil film thickness is calculated at each time step. The alternative is that oil film thickness is kept constant with value provided at seeding.
    biodegradation : bool
        Oil mass is biodegraded (eaten by bacteria).
    plots : dict, optional
        Dictionary of plot names, their filetypes, and any kwargs to pass along, by default None.
        Available plot names are "spaghetti", "animation", "oil", "all".

    Notes
    -----
    Docs available for more initialization options with ``ptm.ParticleTrackingManager?``

    """
    
    _config: dict = None
    _all_config: dict = None
    
    # TODO: is kwargs needed in init? test this.
    def __init__(self, **kwargs):
        """Initialize OpenDriftModel."""
        # TODO: I think there is no reason to have "model" defined in the_manager_config.json since the 
        # model object is used to instantiate the combined object.


        
        # Initialize the parent class
        # This sets up the logger and ParticleTrackingState and SetupOutputFiles.
        super().__init__(**kwargs)

        # OpenDriftConfig, _KNOWN_MODELS = setup_opendrift_config(**kwargs)
        
        # OpenDriftConfig is a subclass of PTMConfig so it knows about all the
        # PTMConfig parameters. PTMConfig is run with OpenDriftConfig.
        # output_file was altered in PTM when setting up logger, so want to use
        # that version.
        # kwargs.update({"output_file": self.output_file})
        keys_from_the_manager = ["use_cache", "stokes_drift", "do3D", "wind_drift_factor", "use_static_masks", "vertical_mixing", "ocean_model"]
        inputs = {key: getattr(self.manager_config,key) for key in keys_from_the_manager}
        keys_from_ocean_model = ["model_drop_vars"]
        inputs.update({key: getattr(self.ocean_model,key) for key in keys_from_ocean_model})
        self.config = OpenDriftConfig(**inputs)  # this runs both OpenDriftConfig and PTMConfig
        # logger = self.config.logger  # this is where logger is expected to be found
        # import pdb; pdb.set_trace()

        self._KNOWN_MODELS = self.manager_config.model_json_schema()['$defs']['OceanModelEnum']["enum"]
        
        # self._setup_interpolator()

        # model = "opendrift"

        # I left this code here but it isn't used for now
        # it will be used if we can export to parquet/netcdf directly
        # without needing to resave the file with extra config
        # # need output_format defined right away
        # self.__dict__["output_format"] = output_format

        # LoggerMethods().merge_with_opendrift_log(logger)
        
        self._create_opendrift_model_object()
        self._update_od_config_from_this_config()
        self._modify_opendrift_model_object()

        # # Extra keyword parameters are not currently allowed so they might be a typo
        # if len(self.kw) > 0:
        #     raise KeyError(f"Unknown input parameter(s) {self.kw} input.")

        # Note that you can see configuration possibilities for a given model with
        # o.list_configspec()
        # You can check the metadata for a given configuration with (min/max/default/type)
        # o.get_configspec('vertical_mixing:timestep')
        # You can check required variables for a model with
        # o.required_variables

        # TODO: streamline this
        self.checked_plot = False


    # def _setup_interpolator(self):
    #     """Setup interpolator."""
    #     # TODO: this isn't working correctly at the moment

    #     if self.config.use_cache:
    #         # TODO: fix this for Ahmad
    #         cache_dir = Path(appdirs.user_cache_dir(appname="particle-tracking-manager", appauthor="axiom-data-science"))
    #         cache_dir.mkdir(parents=True, exist_ok=True)
    #         if self.config.interpolator_filename is None:
    #             self.config.interpolator_filename = cache_dir / Path(f"{self.manager_config.ocean_model.name}_interpolator").with_suffix(".pickle")
    #         else:
    #             self.config.interpolator_filename = Path(self.config.interpolator_filename).with_suffix(".pickle")
    #         self.save_interpolator = True
            
    #         # change interpolator_filename to string
    #         self.config.interpolator_filename = str(self.config.interpolator_filename)
            
    #         if Path(self.config.interpolator_filename).exists():
    #             logger.info(f"Loading the interpolator from {self.config.interpolator_filename}.")
    #         else:
    #             logger.info(f"A new interpolator will be saved to {self.config.interpolator_filename}.")
    #     else:
    #         self.save_interpolator = False
    #         logger.info("Interpolators will not be saved.")

    def _create_opendrift_model_object(self):
        # do this right away so I can query the object
        # we don't actually input output_format here because we first output to netcdf, then
        # resave as parquet after adding in extra config
        # TODO: should drift_model be instantiated in OpenDriftConfig or here?
        # import pdb; pdb.set_trace()
        log_level = logger.level
        if self.config.drift_model == "Leeway":
            from opendrift.models.leeway import Leeway
            # getattr(logging, self.config.log_level) converts from, e.g., "INFO" to 20
            o = Leeway(loglevel=log_level)  # , output_format=self.output_format)

        elif self.config.drift_model == "OceanDrift":
            from opendrift.models.oceandrift import OceanDrift
            o = OceanDrift(
                loglevel=log_level,
            )  # , output_format=self.output_format)

        elif self.config.drift_model == "LarvalFish":
            from opendrift.models.larvalfish import LarvalFish
            o = LarvalFish(
                loglevel=log_level
            )  # , output_format=self.output_format)

        elif self.config.drift_model == "OpenOil":
            from opendrift.models.openoil import OpenOil
            o = OpenOil(
                loglevel=log_level, weathering_model="noaa"
            )  # , output_format=self.output_format)

        else:
            raise ValueError(f"Drifter model {self.config.drift_model} is not recognized.")
        # TODO: Should I keep this sort of ValueError when the input parameter has already been validated?
        
        self.o = o

    def _update_od_config_from_this_config(self):
        """Update OpenDrift's config with OpenDriftConfig and PTMConfig.
        
        Update the default value in OpenDrift's config dict with the 
        config value from OpenDriftConfig (which includes PTMConfig).
        
        This uses the metadata key "od_mapping" to map from the PTM parameter
        name to the OpenDrift parameter name.
        """
        # import pdb; pdb.set_trace()

        for key in self.config.model_fields:
            if getattr(self.config.model_fields[key], "json_schema_extra") is not None:
                if "od_mapping" in self.config.model_fields[key].json_schema_extra:
                    od_key = self.config.model_fields[key].json_schema_extra["od_mapping"]
                    if od_key in self.o._config:# and od_key is not None:
                        self.o._config[od_key]["value"] = getattr(self.config, key)

    def _modify_opendrift_model_object(self):
        # import pdb; pdb.set_trace()
        
        # TODO: where to put these things
        # turn on other things if using stokes_drift
        if self.manager_config.stokes_drift:
            self.o.set_config("drift:use_tabularised_stokes_drift", True)
            # self.o.set_config('drift:tabularised_stokes_drift_fetch', '25000')  # default
            # self.o.set_config('drift:stokes_drift_profile', 'Phillips')  # default

        # If 2D surface simulation (and not Leeway since not available), truncate model output below 0.5 m
        if not self.manager_config.do3D and self.manager_config.z == 0 and self.config.drift_model != "Leeway":
            self.o.set_config("drift:truncate_ocean_model_below_m", 0.5)
            logger.info("Truncating model output below 0.5 m.")


        # If 2D simulation (and not Leeway since not available), turn off vertical advection
        if not self.manager_config.do3D and self.config.drift_model != "Leeway":
            self.o.set_config("drift:vertical_advection", False)
            logger.info("Disabling vertical advection.")

        # If 3D simulation, turn on vertical advection
        if self.manager_config.do3D:
            self.o.set_config("drift:vertical_advection", True)
            logger.info("do3D is True so turning on vertical advection.")


    def add_reader(
        self,
        ds=None,
        name=None,
        oceanmodel_lon0_360=False,
        standard_name_mapping=None,
    ):
        """Might need to cache this if its still slow locally.

        Parameters
        ----------
        ds : xr.Dataset, optional
            Previously-opened Dataset containing ocean model output, if user wants to input
            unknown reader information.
        name : str, optional
            If ds is input, user can also input name of ocean model, otherwise will be called "user_input".
        oceanmodel_lon0_360 : bool
            True if ocean model longitudes span 0 to 360 instead of -180 to 180.
        standard_name_mapping : dict
            Mapping of model variable names to standard names.
        """
        
        # TODO: move out all dataset handling from this method
        
        # TODO: have standard_name_mapping as an initial input only with initial call to OpenDrift?
        # TODO: has ds as an initial input for user-input ds?
        if (
            self.manager_config.ocean_model not in self._KNOWN_MODELS
            and self.manager_config.ocean_model != "test"
            and ds is None
        ):
            raise ValueError(
                "ocean_model must be a known model or user must input a Dataset."
            )

        # standard_name_mapping = standard_name_mapping or {}

        if ds is not None:
            if name is None:
                self.manager_config.ocean_model = "user_input"
            else:
                self.manager_config.ocean_model = name

        # TODO: do I still need a pathway for ocean_model of "test"?
        # TODO: move tests from test_manager to other files
        else:
            ds = self.ocean_model.open_dataset(drop_vars=self.config.drop_vars)
        
        ds = narrow_dataset_to_simulation_time(ds, self.manager_config.start_time, self.manager_config.end_time)
        logger.info("Narrowed model output to simulation time")
        
        ds = apply_known_ocean_model_specific_changes(ds, self.manager_config.ocean_model, self.manager_config.use_static_masks)
        
        # TODO: the stuff in apply_user_input_ocean_model_specific_changes can be moved to OceanModelConfig
        # validation I think
        if self.manager_config.ocean_model not in self._KNOWN_MODELS and self.manager_config.ocean_model != "test":
            ds = apply_user_input_ocean_model_specific_changes(ds, self.manager_config.use_static_mask)

        self.ds = ds

        # if self.manager_config.ocean_model == "test":
        #     pass
        #     # oceanmodel_lon0_360 = True
        #     # loc = "test"
        #     # kwargs_xarray = dict()

        # elif self.manager_config.ocean_model is not None or ds is not None:
        #     # pass
            
        #     # TODO: should I change to computed_fields and where should this go?

        #     # # set drop_vars initial values based on the PTM settings, then add to them for the specific model
        #     # drop_vars = []
        #     # # don't need w if not 3D movement
        #     # if not self.config.do3D:
        #     #     drop_vars += ["w"]
        #     #     logger.info("Dropping vertical velocity (w) because do3D is False")
        #     # else:
        #     #     logger.info("Retaining vertical velocity (w) because do3D is True")

        #     # # don't need winds if stokes drift, wind drift, added wind_uncertainty, and vertical_mixing are off
        #     # # It's possible that winds aren't required for every OpenOil simulation but seems like
        #     # # they would usually be required and the cases are tricky to navigate so also skipping for that case.
        #     # if (
        #     #     not self.config.stokes_drift
        #     #     and self.config.wind_drift_factor == 0
        #     #     and self.config.wind_uncertainty == 0
        #     #     and self.config.drift_model != "OpenOil"
        #     #     and not self.config.vertical_mixing
        #     # ):
        #     #     drop_vars += ["Uwind", "Vwind", "Uwind_eastward", "Vwind_northward"]
        #     #     logger.info(
        #     #         "Dropping wind variables because stokes_drift, wind_drift_factor, wind_uncertainty, and vertical_mixing are all off and drift_model is not 'OpenOil'"
        #     #     )
        #     # else:
        #     #     logger.info(
        #     #         "Retaining wind variables because stokes_drift, wind_drift_factor, wind_uncertainty, or vertical_mixing are on or drift_model is 'OpenOil'"
        #     #     )

        #     # # only keep salt and temp for LarvalFish or OpenOil
        #     # if self.config.drift_model not in ["LarvalFish", "OpenOil"]:
        #     #     drop_vars += ["salt", "temp"]
        #     #     logger.info(
        #     #         "Dropping salt and temp variables because drift_model is not LarvalFish nor OpenOil"
        #     #     )
        #     # else:
        #     #     logger.info(
        #     #         "Retaining salt and temp variables because drift_model is LarvalFish or OpenOil"
        #     #     )

        #     # # keep some ice variables for OpenOil (though later see if these are used)
        #     # if self.config.drift_model != "OpenOil":
        #     #     drop_vars += ["aice", "uice_eastward", "vice_northward"]
        #     #     logger.info(
        #     #         "Dropping ice variables because drift_model is not OpenOil"
        #     #     )
        #     # else:
        #     #     logger.info(
        #     #         "Retaining ice variables because drift_model is OpenOil"
        #     #     )

        #     # # if using static masks, drop wetdry masks.
        #     # # if using wetdry masks, drop static masks.
        #     # if self.config.use_static_masks:
        #     #     standard_name_mapping.update({"mask_rho": "land_binary_mask"})
        #     #     drop_vars += ["wetdry_mask_rho", "wetdry_mask_u", "wetdry_mask_v"]
        #     #     logger.info(
        #     #         "Dropping wetdry masks because using static masks instead."
        #     #     )
        #     # else:
        #     #     standard_name_mapping.update({"wetdry_mask_rho": "land_binary_mask"})
        #     #     drop_vars += ["mask_rho", "mask_u", "mask_v", "mask_psi"]
        #     #     logger.info(
        #     #         "Dropping mask_rho, mask_u, mask_v, mask_psi because using wetdry masks instead."
        #     #     )

        #     # if self.manager_config.ocean_model == "NWGOA":
        #     #     oceanmodel_lon0_360 = True

        #     #     standard_name_mapping.update(
        #     #         {
        #     #             "u_eastward": "x_sea_water_velocity",
        #     #             "v_northward": "y_sea_water_velocity",
        #     #             # NWGOA, there are east/north oriented and will not be rotated
        #     #             # because "east" "north" in variable names
        #     #             "Uwind_eastward": "x_wind",
        #     #             "Vwind_northward": "y_wind",
        #     #         }
        #     #     )

        #     #     # remove all other grid masks because variables are all on rho grid
        #     #     drop_vars += [
        #     #         "hice",
        #     #         "hraw",
        #     #         "snow_thick",
        #     #     ]

        #     #     if self.manager_config.ocean_model_local:

        #     #         if self.config.start_time is None:
        #     #             raise ValueError(
        #     #                 "Need to set start_time ahead of time to add local reader."
        #     #             )
        #     #         start_time = self.config.start_time
        #     #         start = f"{start_time.year}-{str(start_time.month).zfill(2)}-{str(start_time.day).zfill(2)}"
        #     #         end_time = self.config.end_time
        #     #         end = f"{end_time.year}-{str(end_time.month).zfill(2)}-{str(end_time.day).zfill(2)}"
        #     #         loc_local = make_nwgoa_kerchunk(start=start, end=end)

        #     #     # loc_local = "/mnt/depot/data/packrat/prod/aoos/nwgoa/processed/nwgoa_kerchunk.parq"
        #     #     loc_remote = (
        #     #         "http://xpublish-nwgoa.srv.axds.co/datasets/nwgoa_all/zarr/"
        #     #     )

        #     # elif "CIOFS" in self.manager_config.ocean_model:
        #     #     oceanmodel_lon0_360 = False

        #     #     drop_vars += [
        #     #         "wetdry_mask_psi",
        #     #     ]
        #     #     if self.manager_config.ocean_model == "CIOFS":

        #     #         if self.manager_config.ocean_model_local:

        #     #             if self.config.start_time is None:
        #     #                 raise ValueError(
        #     #                     "Need to set start_time ahead of time to add local reader."
        #     #                 )
        #     #             start = f"{self.config.start_time.year}_{str(self.config.start_time.dayofyear - 1).zfill(4)}"
        #     #             end = f"{self.config.end_time.year}_{str(self.config.end_time.dayofyear).zfill(4)}"
        #     #             loc_local = make_ciofs_kerchunk(
        #     #                 start=start, end=end, name="ciofs"
        #     #             )
        #     #         loc_remote = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"

        #     #     elif self.manager_config.ocean_model == "CIOFSFRESH":

        #     #         if self.manager_config.ocean_model_local:

        #     #             if self.config.start_time is None:
        #     #                 raise ValueError(
        #     #                     "Need to set start_time ahead of time to add local reader."
        #     #                 )
        #     #             start = f"{self.config.start_time.year}_{str(self.config.start_time.dayofyear - 1).zfill(4)}"

        #     #             end = f"{self.config.end_time.year}_{str(self.config.end_time.dayofyear).zfill(4)}"
        #     #             loc_local = make_ciofs_kerchunk(
        #     #                 start=start, end=end, name="ciofs_fresh"
        #     #             )
        #     #         loc_remote = None

        #     #     elif self.manager_config.ocean_model == "CIOFSOP":

        #     #         standard_name_mapping.update(
        #     #             {
        #     #                 "u_eastward": "x_sea_water_velocity",
        #     #                 "v_northward": "y_sea_water_velocity",
        #     #             }
        #     #         )

        #     #         if self.manager_config.ocean_model_local:

        #     #             if self.config.start_time is None:
        #     #                 raise ValueError(
        #     #                     "Need to set start_time ahead of time to add local reader."
        #     #                 )
        #     #             start = f"{self.config.start_time.year}-{str(self.config.start_time.month).zfill(2)}-{str(self.config.start_time.day).zfill(2)}"
        #     #             end = f"{self.config.end_time.year}-{str(self.config.end_time.month).zfill(2)}-{str(self.config.end_time.day).zfill(2)}"

        #     #             loc_local = make_ciofs_kerchunk(
        #     #                 start=start, end=end, name="aws_ciofs_with_angle"
        #     #             )
        #     #             # loc_local = "/mnt/depot/data/packrat/prod/noaa/coops/ofs/aws_ciofs/processed/aws_ciofs_kerchunk.parq"

        #     #         loc_remote = "https://thredds.aoos.org/thredds/dodsC/AWS_CIOFS.nc"

        #     # if self.manager_config.ocean_model == "user_input":

        #     #     # check for case that self.config.use_static_masks False (which is the default)
        #     #     # but user input doesn't have wetdry masks
        #     #     # then raise exception and tell user to set use_static_masks True
        #     #     if "wetdry_mask_rho" not in ds.data_vars and not self.config.use_static_masks:
        #     #         raise ValueError(
        #     #             "User input does not have wetdry_mask_rho variable. Set use_static_masks True to use static masks instead."
        #     #         )

        #     #     ds = ds.drop_vars(self.config.drop_vars, errors="ignore")

        #     # # if local and not a user-input ds
        #     # if ds is None:
        #     #     if self.manager_config.ocean_model_local:

        #     #         ds = xr.open_dataset(
        #     #             self.config.loc_local,
        #     #             engine="kerchunk",
        #     #             # chunks={},  # Looks like it is faster not to include this for kerchunk
        #     #             drop_variables=self.config.drop_vars,
        #     #             decode_times=False,
        #     #         )

        #     #         logger.info(
        #     #             f"Opened local dataset starting {self.config.start_time} and ending {self.config.end_time} with number outputs {ds.ocean_time.size}."
        #     #         )

        #     #     # otherwise remote
        #     #     else:
        #     #         if ".nc" in self.config.loc_remote:

        #     #             if self.manager_config.ocean_model == "CIOFSFRESH":
        #     #                 raise NotImplementedError

        #     #             ds = xr.open_dataset(
        #     #                 self.config.loc_remote,
        #     #                 chunks={},
        #     #                 drop_variables=self.config.drop_vars,
        #     #                 decode_times=False,
        #     #             )
        #     #         else:
        #     #             ds = xr.open_zarr(
        #     #                 self.config.loc_remote,
        #     #                 chunks={},
        #     #                 drop_variables=self.config.drop_vars,
        #     #                 decode_times=False,
        #     #             )

        #     #         logger.info(
        #     #             f"Opened remote dataset {self.config.loc_remote} with number outputs {ds.ocean_time.size}."
        #     #         )

        #     # # For NWGOA, need to calculate wetdry mask from a variable
        #     # if self.manager_config.ocean_model == "NWGOA" and not self.config.use_static_masks:
        #     #     ds["wetdry_mask_rho"] = (~ds.zeta.isnull()).astype(int)

        #     # # For CIOFSOP need to rename u/v to have "East" and "North" in the variable names
        #     # # so they aren't rotated in the ROMS reader (the standard names have to be x/y not east/north)
        #     # elif self.manager_config.ocean_model == "CIOFSOP":
        #     #     ds = ds.rename_vars({"urot": "u_eastward", "vrot": "v_northward"})
        #     #     # grid = xr.open_dataset("/mnt/vault/ciofs/HINDCAST/nos.ciofs.romsgrid.nc")
        #     #     # ds["angle"] = grid["angle"]

        #     # try:
        #     #     units = ds.ocean_time.attrs["units"]
        #     # except KeyError:
        #     #     units = ds.ocean_time.encoding["units"]
        #     # datestr = units.split("since ")[1]
        #     # units_date = pd.Timestamp(datestr)

        #     # # use reader start time if not otherwise input
        #     # if self.config.start_time is None:
        #     #     logger.info("setting reader start_time as simulation start_time")
        #     #     # self.config.start_time = reader.start_time
        #     #     # convert using pandas instead of netCDF4
        #     #     self.config.start_time = units_date + pd.to_timedelta(
        #     #         ds.ocean_time[0].values, unit="s"
        #     #     )
        #     # # narrow model output to simulation time if possible before sending to Reader
        #     # if self.config.start_time is not None and self.config.end_time is not None:
        #     #     dt_model = float(
        #     #         ds.ocean_time[1] - ds.ocean_time[0]
        #     #     )  # time step of the model output in seconds
        #     #     # want to include the next ocean model output before the first drifter simulation time
        #     #     # in case it starts before model times
        #     #     start_time_num = (
        #     #         self.config.start_time - units_date
        #     #     ).total_seconds() - dt_model
        #     #     # want to include the next ocean model output after the last drifter simulation time
        #     #     end_time_num = (self.config.end_time - units_date).total_seconds() + dt_model
        #     #     ds = ds.sel(ocean_time=slice(start_time_num, end_time_num))
        #     #     logger.info("Narrowed model output to simulation time")
        #     #     if len(ds.ocean_time) == 0:
        #     #         raise ValueError(
        #     #             "No model output left for simulation time. Check start_time and end_time."
        #     #         )
        #     #     if len(ds.ocean_time) == 1:
        #     #         raise ValueError(
        #     #             "Only 1 model output left for simulation time. Check start_time and end_time."
        #     #         )
        #     # else:
        #     #     raise ValueError(
        #     #         "start_time and end_time must be set to narrow model output to simulation time"
        #     #     )
        reader = reader_ROMS_native.Reader(
            filename=ds,
            name=self.manager_config.ocean_model,
            standard_name_mapping=self.ocean_model.standard_name_mapping,
            save_interpolator=self.config.save_interpolator,
            interpolator_filename=self.config.interpolator_filename,
        )

        self.o.add_reader([reader])
        self.reader = reader
        # can find reader at manager.o.env.readers[self.manager_config.ocean_model]

        # self.oceanmodel_lon0_360 = oceanmodel_lon0_360

        # else:
        #     raise ValueError("reader did not set an ocean_model")
        
        self.state.has_added_reader = True


    @property
    def seed_kws(self):
        """Gather seed input kwargs.

        This could be run more than once.
        """

        already_there = [
            "seed:number",
            "seed:z",
            "seed:seafloor",
            "seed:droplet_diameter_mu",
            "seed:droplet_diameter_min_subsea",
            "seed:droplet_size_distribution",
            "seed:droplet_diameter_sigma",
            "seed:droplet_diameter_max_subsea",
            "seed:object_type",
            "seed:ocean_only",
            "seed_flag",
            "drift:use_tabularised_stokes_drift",
            "drift:vertical_advection",
            "drift:truncate_ocean_model_below_m",
        ]

        if self.manager_config.start_time_end is not None:
            # time can be a list to start drifters linearly in time
            time = [
                self.manager_config.start_time.to_pydatetime(),
                self.manager_config.start_time_end.to_pydatetime(),
            ]
        elif self.manager_config.start_time is not None:
            time = self.manager_config.start_time
            # time = self.manager_config.start_time.to_pydatetime()
        else:
            time = None

        _seed_kws = {
            "time": time,
            "z": self.manager_config.z,
        }
        
        # TODO: are the opendrift config parameters updated with input values?
        # if so, drift_model_config() can maybe only return Opendrift config

        # update seed_kws with drift_model-specific seed parameters
        # seedlist = self.o.get_configspec(prefix="seed")
        seedlist = {k: v["value"] for k, v in self.o.get_configspec(prefix="seed").items()}
        # seedlist = self.drift_model_config(prefix="seed")  # TODO: replace this functionality
        # seedlist = [(one, two) for one, two in seedlist if one not in already_there]
        # seedlist = [(one.replace("seed:", ""), two) for one, two in seedlist]
        seedlist = {one.replace("seed:", ""): two for one, two in seedlist.items() if one not in already_there}
        # seedlist = [(one.replace("seed:", ""), two) for one, two in seedlist]
        _seed_kws.update(seedlist)

        if self.manager_config.seed_flag == "elements":
            # add additional seed parameters
            _seed_kws.update(
                {
                    "lon": self.manager_config.lon,
                    "lat": self.manager_config.lat,
                    "radius": self.config.radius,
                    "radius_type": self.config.radius_type,
                }
            )

        elif self.manager_manager_config.seed_flag == "geojson":

            # geojson needs string representation of time
            _seed_kws["time"] = (
                self.config.start_time.isoformat() if self.config.start_time is not None else None
            )

        self._seed_kws = _seed_kws
        return self._seed_kws


    def seed(self):
        """Actually seed drifters for model."""

        if not self.state.has_added_reader:
            raise ValueError("first add reader with `manager.add_reader(**kwargs)`.")

        if self.manager_config.seed_flag == "elements":
            self.o.seed_elements(**self.seed_kws)

        elif self.manager_config.seed_flag == "geojson":

            # # geojson needs string representation of time
            # self.seed_kws["time"] = self.config.start_time.isoformat()
            self.geojson["properties"] = self.seed_kws
            json_string_dumps = json.dumps(self.geojson)
            self.o.seed_from_geojson(json_string_dumps)

        else:
            raise ValueError(f"seed_flag {self.manager_config.seed_flag} not recognized.")

        self.initial_drifters = self.o.elements_scheduled

        self.state.has_run_seeding = True

    def run(self):
        """Run the drifters!"""
        # TODO: WHy isn't this running correctly? Actually I think just logging isn't working correctly.
        if not self.state.has_run_seeding:
            raise ValueError("first run seeding with `manager.seed()`.")

        logger.info(f"start_time: {self.manager_config.start_time}, end_time: {self.manager_config.end_time}, steps: {self.manager_config.steps}, duration: {self.manager_config.duration}")
        
        # TODO log messages are being repeated
        # TODO: revalidate in PTM run before running?
        # TODO: check warnings when saving to file and fix
        
        # add input config to model config
        self.o.metadata_dict.update(self.manager_config.dict())
        self.o.metadata_dict.update(self.config.dict())
        self.o.metadata_dict.update(self.ocean_model.dict())
        self.o.metadata_dict.update(self.files.dict())

        # actually run
        self.o.run(
            time_step=self.manager_config.time_step,
            time_step_output=self.manager_config.time_step_output,
            steps=self.manager_config.steps,
            export_variables=self.config.export_variables,
            outfile=self.files.output_file,
        )

        # plot if requested
        if self.config.plots:
            # TODO: fix this for Ahmad
            # return plots because now contains the filenames for each plot
            self.config.plots = make_plots(
                self.config.plots, self.o, self.files.output_file.split(".")[0], self.config.drift_model
            )

            # convert plots dict into string representation to save in output file attributes
            # https://github.com/pydata/xarray/issues/1307
            self.config.plots = repr(self.config.plots)

        LoggerMethods().close_loggers(logger)
        self.state.has_run = True

    def run_all(self):
        """Run all steps."""
        if not self.state.has_added_reader:
            self.add_reader()
        if not self.state.has_run_seeding:
            self.seed()
        if not self.state.has_run:
            self.run()


    @property
    def _model_config(self):
        """Surface the model configuration."""

        # save for reinstatement when running the drifters
        if self._config is None:
            self._config = copy.deepcopy(self.o._config)

        return self._config
    
    @property
    def all_config(self):
        """Combined dict of this class config and OpenDrift native config."""
        
        if self._all_config is None:
            self._all_config = {**self._model_config, **self.config.dict()}
        return self._all_config

    def all_export_variables(self):
        """Output list of all possible export variables."""

        vars = (
            list(self.o.elements.variables.keys())
            + ["trajectory", "time"]
            + list(self.o.required_variables.keys())
        )

        return vars

    def export_variables(self):
        """Output list of all actual export variables."""

        return self.o.export_variables

    # def drift_model_config(self, ptm_level=[1, 2, 3], prefix=""):
    def drift_model_config(self, prefix=""):
        """Show config for this drift model selection.
        
        TODO: changing this so update doc strings

        This shows all PTM-controlled parameters for the OpenDrift
        drift model selected and their current values, at the selected ptm_level
        of importance. It includes some additional configuration parameters
        that are indirectly controlled by PTM parameters.

        Parameters
        ----------
        ptm_level : int, list, optional
            Options are 1, 2, 3, or lists of combinations. Use [1,2,3] for all.
            Default is 1.
        prefix : str, optional
            prefix to search config for, only for OpenDrift parameters (not PTM).
        """

        outlist = [
            (key, value_dict["value"])
            for key, value_dict in self.show_config(
                substring=":", ptm_level=ptm_level, level=[1, 2, 3], prefix=prefix
            ).items()
            if "value" in value_dict and value_dict["value"] is not None
        ]

        # also PTM config parameters that are separate from OpenDrift parameters
        outlist2 = [
            (key, value_dict["value"])
            for key, value_dict in self.show_config(
                ptm_level=ptm_level, prefix=prefix
            ).items()
            if "od_mapping" not in value_dict
            and "value" in value_dict
            and value_dict["value"] is not None
        ]

        # extra parameters that are not in the config_model but are set by PTM indirectly
        extra_keys = [
            "drift:vertical_advection",
            "drift:truncate_ocean_model_below_m",
            "drift:use_tabularised_stokes_drift",
        ]
        outlist += [
            (key, self.show_config(key=key)["value"])
            for key in extra_keys
            if "value" in self.show_config(key=key)
        ]

        return outlist + outlist2

    # def get_configspec(self, prefix, substring, excludestring, level, ptm_level):
    #     """Copied from OpenDrift, then modified."""

    #     if not isinstance(level, list) and level is not None:
    #         level = [level]
    #     if not isinstance(ptm_level, list) and ptm_level is not None:
    #         ptm_level = [ptm_level]

    #     # check for prefix or substring comparison
    #     configspec = {
    #         k: v
    #         for (k, v) in self.all_config.items()
    #         if k.startswith(prefix) and substring in k and excludestring not in k
    #     }
    #     import pdb; pdb.set_trace()
    #     # HERE: trying to be able to use config so that open_drift tests can run 
    #     # maybe my config files are the problem and need to follow JSON Schema?
    #     # EXPERIMENT first to make sure the schema output from pydantic will do what 
    #     # I am hoping before converting anything to it.
    #     if level is not None:
    #         # check for levels (if present)
    #         configspec = {
    #             k: v
    #             for (k, v) in configspec.items()
    #             if "level" in configspec[k] and configspec[k]["level"] in level
    #         }
    #     import pdb; pdb.set_trace()

    #     if ptm_level is not None:
    #         # check for ptm_levels (if present)
    #         configspec = {
    #             k: v
    #             for (k, v) in configspec.items()
    #             if "ptm_level" in configspec[k]
    #             and configspec[k]["ptm_level"] in ptm_level
    #         }
    #     import pdb; pdb.set_trace()

    #     return configspec

    def show_all_config(
        self,
        key=None,
        prefix="",
        level=None,
        ptm_level=None,
        substring="",
        excludestring="excludestring",
    ) -> dict:
        """Show configuring for the drift model selected in configuration.

        Runs configuration for you if it hasn't yet been run.

        Parameters
        ----------
        key : str, optional
            If input, show configuration for just that key.
        prefix : str, optional
            prefix to search config for, only for OpenDrift parameters (not PTM).
        level : int, list, optional
            Limit search by level:

            * CONFIG_LEVEL_ESSENTIAL = 1
            * CONFIG_LEVEL_BASIC = 2
            * CONFIG_LEVEL_ADVANCED = 3

            e.g. 1, [1,2], [1,2,3]
        ptm_level : int, list, optional
            Limit search by level:

            * Surface to user = 1
            * Medium surface to user = 2
            * Surface but bury = 3

            e.g. 1, [1,2], [1,2,3]. To access all PTM parameters search for
            `ptm_level=[1,2,3]`.
        substring : str, optional
            If input, show configuration that contains that substring.
        excludestring : str, optional
            configuration parameters are not shown if they contain this string.

        Examples
        --------
        Show all possible configuration for the previously-selected drift model:

        >>> manager.show_config()

        Show configuration with a specific prefix:

        >>> manager.show_config(prefix="seed")

        Show configuration matching a substring:

        >>> manager.show_config(substring="stokes")

        Show configuration at a specific level (from OpenDrift):

        >>> manager.show_config(level=1)

        Show all OpenDrift configuration:

        >>> manager.show_config(level=[1,2,3])

        Show configuration for only PTM-specified parameters:

        >>> manager.show_config(ptm_level=[1,2,3])

        Show configuration for a specific PTM level:

        >>> manager.show_config(ptm_level=2)

        Show configuration for a single key:

        >>> manager.show_config("seed:oil_type")

        Show configuration for parameters that are both OpenDrift and PTM-modified:

        >>> m.show_config(ptm_level=[1,2,3], level=[1,2,3])

        """

        if key is not None:
            prefix = key

        output = self.get_configspec(
            prefix=prefix,
            level=level,
            ptm_level=ptm_level,
            substring=substring,
            excludestring=excludestring,
        )
        import pdb; pdb.set_trace()
        if key is not None:
            if key in output:
                return output[key]
            else:
                return output
        else:
            return output

    def reader_metadata(self, key):
        """allow manager to query reader metadata."""

        if not self.state.has_added_reader:
            raise ValueError("reader has not been added yet.")
        return self.o.env.readers[self.manager_config.ocean_model].__dict__[key]

    # @property
    # def outfile_name(self):
    #     """Output file name."""

    #     return self.o.outfile_name
