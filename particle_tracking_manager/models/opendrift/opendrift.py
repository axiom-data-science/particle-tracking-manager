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

# using my own version of ROMS reader
# from .reader_ROMS_native import Reader
import pandas as pd
import xarray as xr

from opendrift.models.larvalfish import LarvalFish
from opendrift.models.leeway import Leeway
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_ROMS_native

from ...cli import is_None
from ...the_manager import _KNOWN_MODELS, ParticleTrackingManager
from .utils import make_ciofs_kerchunk, make_nwgoa_kerchunk


# from .cli import is_None
# from .the_manager import ParticleTrackingManager


# Read OpenDrift configuration information
loc = Path(__file__).parent / Path("config.json")
with open(loc, "r") as f:
    # Load the JSON file into a Python object
    config_model = json.load(f)

# convert "None"s to Nones
for key in config_model.keys():
    if "default" in config_model[key] and is_None(config_model[key]["default"]):
        config_model[key]["default"] = None


# logger = logging.getLogger("opendrift")


# @copydocstring( ParticleTrackingManager )
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

    horizontal_diffusivity : float
        Horizontal diffusivity is None by default but will be set to a grid-dependent value for known ocean_model values. This is calculated as 0.1 m/s sub-gridscale velocity that is missing from the model output and multiplied by an estimate of the horizontal grid resolution. This leads to a larger value for NWGOA which has a larger value for mean horizontal grid resolution (lower resolution). If the user inputs their own ocean_model information, they can input their own horizontal_diffusivity value. A user can use a known ocean_model and then overwrite the horizontal_diffusivity value to some value.
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
    wind_drift_factor : float
        Elements at surface are moved with this fraction of the wind vector, in addition to currents and Stokes drift.
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
    log : str, optional
        Options are "low" and "high" verbosity for log, by default "low"

    Notes
    -----
    Docs available for more initialization options with ``ptm.ParticleTrackingManager?``

    """

    logger: logging.Logger
    log: str
    loglevel: str
    vertical_mixing_timestep: float
    diffusivitymodel: str
    mixed_layer_depth: float
    wind_drift_factor: float
    wind_drift_depth: float
    stokes_drift: bool
    drift_model: str
    o: Union[OceanDrift, Leeway, LarvalFish, OpenOil]
    horizontal_diffusivity: Optional[float]
    config_model: dict

    def __init__(
        self,
        drift_model: str = config_model["drift_model"]["default"],
        export_variables: str = config_model["export_variables"]["default"],
        radius: int = config_model["radius"]["default"],
        radius_type: str = config_model["radius_type"]["default"],
        horizontal_diffusivity: float = config_model["horizontal_diffusivity"][
            "default"
        ],
        current_uncertainty: float = config_model["current_uncertainty"]["default"],
        wind_uncertainty: float = config_model["wind_uncertainty"]["default"],
        use_auto_landmask: bool = config_model["use_auto_landmask"]["default"],
        diffusivitymodel: str = config_model["diffusivitymodel"]["default"],
        stokes_drift: bool = config_model["stokes_drift"]["default"],
        mixed_layer_depth: float = config_model["mixed_layer_depth"]["default"],
        coastline_action: str = config_model["coastline_action"]["default"],
        seafloor_action: str = config_model["seafloor_action"]["default"],
        max_speed: int = config_model["max_speed"]["default"],
        wind_drift_factor: float = config_model["wind_drift_factor"]["default"],
        wind_drift_depth: float = config_model["wind_drift_depth"]["default"],
        vertical_mixing_timestep: float = config_model["vertical_mixing_timestep"][
            "default"
        ],
        object_type: str = config_model["object_type"]["default"],
        diameter: float = config_model["diameter"]["default"],
        neutral_buoyancy_salinity: float = config_model["neutral_buoyancy_salinity"][
            "default"
        ],
        stage_fraction: float = config_model["stage_fraction"]["default"],
        hatched: float = config_model["hatched"]["default"],
        length: float = config_model["length"]["default"],
        weight: float = config_model["weight"]["default"],
        oil_type: str = config_model["oil_type"]["default"],
        m3_per_hour: float = config_model["m3_per_hour"]["default"],
        oil_film_thickness: float = config_model["oil_film_thickness"]["default"],
        droplet_size_distribution: str = config_model["droplet_size_distribution"][
            "default"
        ],
        droplet_diameter_mu: float = config_model["droplet_diameter_mu"]["default"],
        droplet_diameter_sigma: float = config_model["droplet_diameter_sigma"][
            "default"
        ],
        droplet_diameter_min_subsea: float = config_model[
            "droplet_diameter_min_subsea"
        ]["default"],
        droplet_diameter_max_subsea: float = config_model[
            "droplet_diameter_max_subsea"
        ]["default"],
        emulsification: bool = config_model["emulsification"]["default"],
        dispersion: bool = config_model["dispersion"]["default"],
        evaporation: bool = config_model["evaporation"]["default"],
        update_oilfilm_thickness: bool = config_model["update_oilfilm_thickness"][
            "default"
        ],
        biodegradation: bool = config_model["biodegradation"]["default"],
        log: str = config_model["log"]["default"],
        **kw,
    ) -> None:
        """Inputs for OpenDrift model."""

        # get all named parameters input to ParticleTrackingManager class
        from inspect import signature

        sig = signature(OpenDriftModel)

        # initialize all class attributes to None without triggering the __setattr__ method
        # which does a bunch more stuff
        for key in sig.parameters.keys():
            self.__dict__[key] = None

        self.__dict__["config_model"] = config_model

        model = "opendrift"

        if log == "low":
            self.__dict__["loglevel"] = 20
        elif log == "high":
            self.__dict__["loglevel"] = 0

        # need drift_model defined for the log to work properly for both manager and model
        # so do this before super initialization
        self.__dict__["drift_model"] = drift_model

        # do this right away so I can query the object
        if self.drift_model == "Leeway":
            o = Leeway(loglevel=self.loglevel)

        elif self.drift_model == "OceanDrift":
            o = OceanDrift(loglevel=self.loglevel)

        elif self.drift_model == "LarvalFish":
            o = LarvalFish(loglevel=self.loglevel)

        elif self.drift_model == "OpenOil":
            o = OpenOil(loglevel=self.loglevel, weathering_model="noaa")

        else:
            raise ValueError(f"Drifter model {self.drift_model} is not recognized.")

        self.__dict__["o"] = o

        self.__dict__["logger"] = logging.getLogger(
            model
        )  # use this syntax to avoid __setattr__

        super().__init__(model, **kw)

        # Extra keyword parameters are not currently allowed so they might be a typo
        if len(self.kw) > 0:
            raise KeyError(f"Unknown input parameter(s) {self.kw} input.")

        # Note that you can see configuration possibilities for a given model with
        # o.list_configspec()
        # You can check the metadata for a given configuration with (min/max/default/type)
        # o.get_configspec('vertical_mixing:timestep')
        # You can check required variables for a model with
        # o.required_variables

        # Set all attributes which will trigger some checks and changes in __setattr__
        # these will also update "value" in the config dict
        for key in sig.parameters.keys():
            # no need to run through for init if value is None (already set to None)
            if locals()[key] is not None:
                self.__setattr__(key, locals()[key])

    def calc_known_horizontal_diffusivity(self):
        """Calculate horizontal diffusivity based on known ocean_model."""

        # dx: approximate horizontal grid resolution (meters), used to calculate horizontal diffusivity
        if self.ocean_model == "NWGOA":
            dx = 1500
        elif "CIOFS" in self.ocean_model:
            dx = 100

        # horizontal diffusivity is calculated based on the mean horizontal grid resolution
        # for the model being used.
        # 0.1 is a guess for the magnitude of velocity being missed in the models, the sub-gridscale velocity
        sub_gridscale_velocity = 0.1
        horizontal_diffusivity = sub_gridscale_velocity * dx
        return horizontal_diffusivity

    def __setattr_model__(self, name: str, value) -> None:
        """Implement my own __setattr__ but here to enforce actions."""

        # don't allow drift_model to be reset, have to reinitialize object instead
        # check for type of m.o and drift_model matching to enforce this
        if (name in ["o", "drift_model"]) and (
            self.drift_model not in str(type(self.o))
        ):
            raise KeyError(
                "Can't overwrite `drift_model`; instead initialize OpenDriftModel with desired drift_model."
            )

        # create/update "value" keyword in config to keep it up to date
        if (
            self.config_model is not None
            # and name != "config_model"
            # and name != "config_ptm"
            and name in self.config_model.keys()
        ):
            self.config_model[name]["value"] = value
        self._update_config()

        if name in ["ocean_model", "horizontal_diffusivity"]:

            # just set the value and move on if purposely setting a non-None value
            # of horizontal_diffusivity; specifying this for clarity (already set
            # value above).
            if name == "horizontal_diffusivity" and value is not None:
                self.logger.info(
                    f"Setting horizontal_diffusivity to user-selected value {value}."
                )

            # in all other cases that ocean_model is a known model, want to use the
            # grid-dependent value
            elif self.ocean_model in _KNOWN_MODELS:

                hdiff = self.calc_known_horizontal_diffusivity()
                self.logger.info(
                    f"Setting horizontal_diffusivity parameter to one tuned to reader model of value {hdiff}."
                )
                # when editing the __dict__ directly have to also update config_model
                self.__dict__["horizontal_diffusivity"] = hdiff
                self.config_model["horizontal_diffusivity"]["value"] = hdiff

            # if user not using a known ocean_model, change horizontal_diffusivity from None to 0
            # so it has a value. User can subsequently overwrite it too.
            elif (
                self.ocean_model not in _KNOWN_MODELS
                and self.horizontal_diffusivity is None
            ):

                self.logger.info(
                    """Since ocean_model is user-input, changing horizontal_diffusivity parameter from None to 0.0.
                    You can also set it to a specific value with `m.horizontal_diffusivity=[number]`."""
                )

                self.__dict__["horizontal_diffusivity"] = 0
                self.config_model["horizontal_diffusivity"]["value"] = 0

        # turn on other things if using stokes_drift
        if name == "stokes_drift" and value:
            if self.drift_model != "Leeway":
                self.o.set_config("drift:use_tabularised_stokes_drift", True)
            # self.o.set_config('drift:tabularised_stokes_drift_fetch', '25000')  # default
            # self.o.set_config('drift:stokes_drift_profile', 'Phillips')  # default

        # too soon to do this, need to run it later
        # Leeway model doesn't have this option built in
        if name in ["surface_only", "drift_model"]:
            if self.surface_only and self.drift_model != "Leeway":
                self.logger.info("Truncating model output below 0.5 m.")
                self.o.set_config("drift:truncate_ocean_model_below_m", 0.5)
            elif (
                not self.surface_only
                and self.drift_model != "Leeway"
                and self.show_config(key="drift:truncate_ocean_model_below_m")["value"]
                is not None
            ):
                self.logger.info("Un-truncating model output below 0.5 m.")
                self.o.set_config("drift:truncate_ocean_model_below_m", None)

        # Leeway doesn't have this option available
        if name == "do3D" and not value and self.drift_model != "Leeway":
            self.logger.info("do3D is False so disabling vertical motion.")
            self.o.disable_vertical_motion()
        elif name == "do3D" and not value and self.drift_model == "Leeway":
            self.logger.info(
                "do3D is False but drift_model is Leeway so doing nothing."
            )

        if name == "do3D" and value and self.drift_model != "Leeway":
            self.logger.info("do3D is True so turning on vertical advection.")
            self.o.set_config("drift:vertical_advection", True)
        elif name == "do3D" and value and self.drift_model == "Leeway":
            self.logger.info(
                "do3D is True but drift_model is Leeway so changing do3D to False."
            )
            self.do3D = False

        # if drift_model is LarvalFish, vertical_mixing has to be True
        if name == "vertical_mixing" and not value and self.drift_model == "LarvalFish":
            raise ValueError(
                "drift_model is LarvalFish which always has vertical mixing on in OpenDrift so vertical_mixing must be True."
            )

        # if drift_model is LarvalFish, surface_only can't be True
        if name == "surface_only" and value and self.drift_model == "LarvalFish":
            raise ValueError(
                "drift_model is LarvalFish which is always 3D in OpenDrift so surface_only must be False."
            )

        # if drift_model is LarvalFish, do3D has to be True
        if name == "do3D" and not value and self.drift_model == "LarvalFish":
            raise ValueError(
                "drift_model is LarvalFish which is always 3D in OpenDrift so do3D must be True."
            )

        # Make sure vertical_mixing_timestep equals None if vertical_mixing False
        if name in ["vertical_mixing", "vertical_mixing_timestep"]:
            if not self.vertical_mixing:
                self.logger.info(
                    "vertical_mixing is False, so setting value of vertical_mixing_timestep "
                    "to None."
                )
                self.__dict__["vertical_mixing_timestep"] = None
                self.config_model["vertical_mixing_timestep"]["value"] = None

        # Make sure diffusivitymodel equals default value if vertical_mixing False
        if name in ["vertical_mixing", "diffusivitymodel"]:
            dmodeldef = self.config_model["diffusivitymodel"]["default"]
            if (
                not self.vertical_mixing
                and self.diffusivitymodel != dmodeldef
                and self.diffusivitymodel is not None
            ):
                self.logger.info(
                    "vertical_mixing is False, so resetting value of diffusivitymodel to default and not using."
                )
                self.__dict__["diffusivitymodel"] = dmodeldef
                self.config_model["diffusivitymodel"]["value"] = dmodeldef

        # Make sure mixed_layer_depth equals default value if vertical_mixing False
        if name in ["vertical_mixing", "mixed_layer_depth"]:
            mlddef = self.config_model["mixed_layer_depth"]["default"]
            if (
                not self.vertical_mixing
                and self.mixed_layer_depth != mlddef
                and self.mixed_layer_depth is not None
            ):
                self.logger.info(
                    "vertical_mixing is False, so resetting value of mixed_layer_depth to default and not using."
                )
                self.__dict__["mixed_layer_depth"] = mlddef
                self.config_model["mixed_layer_depth"]["value"] = mlddef

        # make sure user isn't try to use Leeway or LarvalFish and "wind_drift_factor" at the same time
        if name == "wind_drift_factor":
            if self.drift_model in ["Leeway", "LarvalFish"]:
                self.logger.info(
                    "wind_drift_factor cannot be used with Leeway or LarvalFish models, "
                    "so setting to None."
                )
                self.__dict__["wind_drift_factor"] = None
                self.config_model["wind_drift_factor"]["value"] = None

        # make sure user isn't try to use Leeway or LarvalFish models and "wind_drift_depth" at the same time
        if name == "wind_drift_depth":
            if self.drift_model in ["Leeway", "LarvalFish"]:
                self.logger.info(
                    "wind_drift_depth cannot be used with Leeway or LarvalFish models, "
                    "so setting to None."
                )
                self.__dict__["wind_drift_depth"] = None
                self.config_model["wind_drift_depth"]["value"] = None

        # make sure user isn't try to use Leeway and "stokes_drift" at the same time
        if name == "stokes_drift":
            if self.drift_model == "Leeway" and self.stokes_drift:
                self.logger.info(
                    "stokes_drift cannot be used with Leeway model, so changing to False."
                )
                self.__dict__["stokes_drift"] = False
                self.config_model["stokes_drift"]["value"] = False

        # Add export variables for certain drift_model values
        # drift_model is always set initially only
        if name == "export_variables" and self.drift_model == "OpenOil":
            oil_vars = [
                "mass_oil",
                "density",
                "mass_evaporated",
                "mass_dispersed",
                "mass_biodegraded",
                "viscosity",
                "water_fraction",
            ]
            self.__dict__["export_variables"] += oil_vars
            self.config_model["export_variables"]["value"] += oil_vars
        elif name == "export_variables" and self.drift_model == "Leeway":
            vars = ["object_type"]
            self.__dict__["export_variables"] += vars
            self.config_model["export_variables"]["value"] += vars
        elif name == "export_variables" and self.drift_model == "LarvalFish":
            vars = [
                "diameter",
                "neutral_buoyancy_salinity",
                "stage_fraction",
                "hatched",
                "length",
                "weight",
            ]
            self.__dict__["export_variables"] += vars
            self.config_model["export_variables"]["value"] += vars

        self._update_config()

    def run_add_reader(
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

        if (
            self.ocean_model not in _KNOWN_MODELS
            and self.ocean_model != "test"
            and ds is None
        ):
            raise ValueError(
                "ocean_model must be a known model or user must input a Dataset."
            )

        standard_name_mapping = standard_name_mapping or {}

        if ds is not None:
            if name is None:
                self.ocean_model = "user_input"
            else:
                self.ocean_model = name

        if self.ocean_model == "test":
            pass
            # oceanmodel_lon0_360 = True
            # loc = "test"
            # kwargs_xarray = dict()

        elif self.ocean_model is not None or ds is not None:
            if self.ocean_model_local:
                self.logger.info(
                    f"Using local output for ocean_model {self.ocean_model}"
                )
            else:
                self.logger.info(
                    f"Using remote output for ocean_model {self.ocean_model}"
                )

            # set drop_vars initial values based on the PTM settings, then add to them for the specific model
            drop_vars = []
            # don't need w if not 3D movement
            if not self.do3D:
                drop_vars += ["w"]
                self.logger.info("Dropping vertical velocity (w) because do3D is False")
            else:
                self.logger.info("Retaining vertical velocity (w) because do3D is True")

            # don't need winds if stokes drift, wind drift, added wind_uncertainty, and vertical_mixing are off
            # It's possible that winds aren't required for every OpenOil simulation but seems like
            # they would usually be required and the cases are tricky to navigate so also skipping for that case.
            if (
                not self.stokes_drift
                and self.wind_drift_factor == 0
                and self.wind_uncertainty == 0
                and self.drift_model != "OpenOil"
                and not self.vertical_mixing
            ):
                drop_vars += ["Uwind", "Vwind", "Uwind_eastward", "Vwind_northward"]
                self.logger.info(
                    "Dropping wind variables because stokes_drift, wind_drift_factor, wind_uncertainty, and vertical_mixing are all off and drift_model is not 'OpenOil'"
                )
            else:
                self.logger.info(
                    "Retaining wind variables because stokes_drift, wind_drift_factor, wind_uncertainty, or vertical_mixing are on or drift_model is 'OpenOil'"
                )

            # only keep salt and temp for LarvalFish or OpenOil
            if self.drift_model not in ["LarvalFish", "OpenOil"]:
                drop_vars += ["salt", "temp"]
                self.logger.info(
                    "Dropping salt and temp variables because drift_model is not LarvalFish nor OpenOil"
                )
            else:
                self.logger.info(
                    "Retaining salt and temp variables because drift_model is LarvalFish or OpenOil"
                )

            # keep some ice variables for OpenOil (though later see if these are used)
            if self.drift_model != "OpenOil":
                drop_vars += ["aice", "uice_eastward", "vice_northward"]
                self.logger.info(
                    "Dropping ice variables because drift_model is not OpenOil"
                )
            else:
                self.logger.info(
                    "Retaining ice variables because drift_model is OpenOil"
                )

            # if using static masks, drop wetdry masks.
            # if using wetdry masks, drop static masks.
            if self.use_static_masks:
                standard_name_mapping.update({"mask_rho": "land_binary_mask"})
                drop_vars += ["wetdry_mask_rho", "wetdry_mask_u", "wetdry_mask_v"]
                self.logger.info(
                    "Dropping wetdry masks because using static masks instead."
                )
            else:
                standard_name_mapping.update({"wetdry_mask_rho": "land_binary_mask"})
                drop_vars += ["mask_rho", "mask_u", "mask_v", "mask_psi"]
                self.logger.info(
                    "Dropping mask_rho, mask_u, mask_v, mask_psi because using wetdry masks instead."
                )

            if self.ocean_model == "NWGOA":
                oceanmodel_lon0_360 = True

                standard_name_mapping.update(
                    {
                        "u_eastward": "x_sea_water_velocity",
                        "v_northward": "y_sea_water_velocity",
                        # NWGOA, there are east/north oriented and will not be rotated
                        # because "east" "north" in variable names
                        "Uwind_eastward": "x_wind",
                        "Vwind_northward": "y_wind",
                    }
                )

                # remove all other grid masks because variables are all on rho grid
                drop_vars += [
                    "hice",
                    "hraw",
                    "snow_thick",
                ]
                start = f"{self.start_time.year}-{str(self.start_time.month).zfill(2)}-{str(self.start_time.day).zfill(2)}"
                end = f"{self.end_time.year}-{str(self.end_time.month).zfill(2)}-{str(self.end_time.day).zfill(2)}"
                loc_local = make_nwgoa_kerchunk(start=start, end=end)

                # loc_local = "/mnt/depot/data/packrat/prod/aoos/nwgoa/processed/nwgoa_kerchunk.parq"
                loc_remote = (
                    "http://xpublish-nwgoa.srv.axds.co/datasets/nwgoa_all/zarr/"
                )

            elif "CIOFS" in self.ocean_model:
                oceanmodel_lon0_360 = False

                drop_vars += [
                    "wetdry_mask_psi",
                ]
                if self.ocean_model == "CIOFS":
                    start = f"{self.start_time.year}_{str(self.start_time.dayofyear - 1).zfill(4)}"
                    end = (
                        f"{self.end_time.year}_{str(self.end_time.dayofyear).zfill(4)}"
                    )
                    loc_local = make_ciofs_kerchunk(start=start, end=end, name="ciofs")
                    loc_remote = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"

                elif self.ocean_model == "CIOFSFRESH":
                    start = f"{self.start_time.year}_{str(self.start_time.dayofyear - 1).zfill(4)}"
                    end = (
                        f"{self.end_time.year}_{str(self.end_time.dayofyear).zfill(4)}"
                    )
                    loc_local = make_ciofs_kerchunk(
                        start=start, end=end, name="ciofs_fresh"
                    )
                    loc_remote = None

                elif self.ocean_model == "CIOFSOP":

                    standard_name_mapping.update(
                        {
                            "u_eastward": "x_sea_water_velocity",
                            "v_northward": "y_sea_water_velocity",
                        }
                    )

                    loc_local = "/mnt/depot/data/packrat/prod/noaa/coops/ofs/aws_ciofs/processed/aws_ciofs_kerchunk.parq"
                    loc_remote = "https://thredds.aoos.org/thredds/dodsC/AWS_CIOFS.nc"

            elif self.ocean_model == "user_input":

                # check for case that self.use_static_masks False (which is the default)
                # but user input doesn't have wetdry masks
                # then raise exception and tell user to set use_static_masks True
                if "wetdry_mask_rho" not in ds.data_vars and not self.use_static_masks:
                    raise ValueError(
                        "User input does not have wetdry_mask_rho variable. Set use_static_masks True to use static masks instead."
                    )

                ds = ds.drop_vars(drop_vars, errors="ignore")

            # if local and not a user-input ds
            if ds is None:
                if self.ocean_model_local:

                    ds = xr.open_dataset(
                        loc_local,
                        engine="kerchunk",
                        chunks={},
                        drop_variables=drop_vars,
                        decode_times=False,
                    )

                # otherwise remote
                else:
                    if ".nc" in loc_remote:

                        if self.ocean_model == "CIOFSFRESH":
                            raise NotImplementedError

                        ds = xr.open_dataset(
                            loc_remote,
                            chunks={},
                            drop_variables=drop_vars,
                            decode_times=False,
                        )
                    else:
                        ds = xr.open_zarr(
                            loc_remote,
                            chunks={},
                            drop_variables=drop_vars,
                            decode_times=False,
                        )

            # For NWGOA, need to calculate wetdry mask from a variable
            if self.ocean_model == "NWGOA" and not self.use_static_masks:
                ds["wetdry_mask_rho"] = (~ds.zeta.isnull()).astype(int)

            # For CIOFSOP need to rename u/v to have "East" and "North" in the variable names
            # so they aren't rotated in the ROMS reader (the standard names have to be x/y not east/north)
            elif self.ocean_model == "CIOFSOP":
                ds = ds.rename_vars({"urot": "u_eastward", "vrot": "v_northward"})
                # grid = xr.open_dataset("/mnt/vault/ciofs/HINDCAST/nos.ciofs.romsgrid.nc")
                # ds["angle"] = grid["angle"]

            try:
                units = ds.ocean_time.attrs["units"]
            except KeyError:
                units = ds.ocean_time.encoding["units"]
            datestr = units.split("since ")[1]
            units_date = pd.Timestamp(datestr)

            # use reader start time if not otherwise input
            if self.start_time is None:
                self.logger.info("setting reader start_time as simulation start_time")
                # self.start_time = reader.start_time
                # convert using pandas instead of netCDF4
                self.start_time = units_date + pd.to_timedelta(
                    ds.ocean_time[0].values, unit="s"
                )
            # narrow model output to simulation time if possible before sending to Reader
            if self.start_time is not None and self.end_time is not None:
                dt_model = float(
                    ds.ocean_time[1] - ds.ocean_time[0]
                )  # time step of the model output in seconds
                # want to include the next ocean model output before the first drifter simulation time
                # in case it starts before model times
                start_time_num = (
                    self.start_time - units_date
                ).total_seconds() - dt_model
                # want to include the next ocean model output after the last drifter simulation time
                end_time_num = (self.end_time - units_date).total_seconds() + dt_model
                ds = ds.sel(ocean_time=slice(start_time_num, end_time_num))
                self.logger.info("Narrowed model output to simulation time")
                if len(ds.ocean_time) == 0:
                    raise ValueError(
                        "No model output left for simulation time. Check start_time and end_time."
                    )
            else:
                raise ValueError(
                    "start_time and end_time must be set to narrow model output to simulation time"
                )

            # save interpolators to save time
            cache_dir = Path(
                appdirs.user_cache_dir(
                    appname="particle-tracking-manager", appauthor="axiom-data-science"
                )
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir = cache_dir
            self.interpolator_filename = cache_dir / Path(
                f"{self.ocean_model}_interpolator"
            )
            reader = reader_ROMS_native.Reader(
                filename=ds,
                name=self.ocean_model,
                standard_name_mapping=standard_name_mapping,
                save_interpolator=True,
                interpolator_filename=self.interpolator_filename,
            )

            self.o.add_reader([reader])
            self.reader = reader
            # can find reader at manager.o.env.readers[self.ocean_model]

            self.oceanmodel_lon0_360 = oceanmodel_lon0_360

        else:
            raise ValueError("reader did not set an ocean_model")

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
            "seed_flag",
            "drift:use_tabularised_stokes_drift",
            "drift:vertical_advection",
            "drift:truncate_ocean_model_below_m",
        ]

        if self.start_time_end is not None:
            # time can be a list to start drifters linearly in time
            time = [
                self.start_time.to_pydatetime(),
                self.start_time_end.to_pydatetime(),
            ]
        elif self.start_time is not None:
            time = self.start_time.to_pydatetime()
        else:
            time = None

        _seed_kws = {
            "time": time,
            "z": self.z,
        }

        # update seed_kws with drift_model-specific seed parameters
        seedlist = self.drift_model_config(prefix="seed")
        seedlist = [(one, two) for one, two in seedlist if one not in already_there]
        seedlist = [(one.replace("seed:", ""), two) for one, two in seedlist]
        _seed_kws.update(seedlist)

        if self.seed_flag == "elements":
            # add additional seed parameters
            _seed_kws.update(
                {
                    "lon": self.lon,
                    "lat": self.lat,
                    "radius": self.radius,
                    "radius_type": self.radius_type,
                }
            )

        elif self.seed_flag == "geojson":

            # geojson needs string representation of time
            _seed_kws["time"] = (
                self.start_time.isoformat() if self.start_time is not None else None
            )

        self._seed_kws = _seed_kws
        return self._seed_kws

    def run_seed(self):
        """Actually seed drifters for model."""

        if self.seed_flag == "elements":
            self.o.seed_elements(**self.seed_kws)

        elif self.seed_flag == "geojson":

            # geojson needs string representation of time
            self.seed_kws["time"] = self.start_time.isoformat()
            self.geojson["properties"] = self.seed_kws
            json_string_dumps = json.dumps(self.geojson)
            self.o.seed_from_geojson(json_string_dumps)

        else:
            raise ValueError(f"seed_flag {self.seed_flag} not recognized.")

        self.initial_drifters = self.o.elements_scheduled

    def run_drifters(self):
        """Run the drifters!"""

        if self.steps is None and self.duration is None and self.end_time is None:
            raise ValueError(
                "Exactly one of steps, duration, or end_time must be input and not be None."
            )

        if self.run_forward:
            timedir = 1
        else:
            timedir = -1

        # drop non-OpenDrift parameters now so they aren't brought into simulation (they mess up the write step)
        full_config = copy.deepcopy(self._config)  # save
        config_input_to_opendrift = {
            k: full_config[k] for k in self._config_orig.keys()
        }

        self.o._config = config_input_to_opendrift  # only OpenDrift config

        output_file_initial = (
            f"{self.output_file}_initial"
            or f"output-results_{datetime.datetime.now():%Y-%m-%dT%H%M:%SZ}.nc"
        )

        self.o.run(
            time_step=timedir * self.time_step,
            time_step_output=self.time_step_output,
            steps=self.steps,
            export_variables=self.export_variables,
            outfile=output_file_initial,
        )

        self.o._config = full_config  # reinstate config

        # open outfile file and add config to it
        # config can't be present earlier because it breaks opendrift
        ds = xr.open_dataset(output_file_initial)
        for k, v in self.drift_model_config():
            if isinstance(v, (bool, type(None), pd.Timestamp, pd.Timedelta)):
                v = str(v)
            ds.attrs[f"ptm_config_{k}"] = v

        # Make new output file
        output_file = (
            self.output_file
            or f"output-results_{datetime.datetime.utcnow():%Y-%m-%dT%H%M:%SZ}.nc"
        )

        ds.to_netcdf(output_file)

        # update with new path name
        self.o.outfile_name = output_file

        try:
            # remove initial file to save space
            os.remove(output_file_initial)
        except PermissionError:
            # windows issue
            pass

    @property
    def _config(self):
        """Surface the model configuration."""

        # save for reinstatement when running the drifters
        if self._config_orig is None:
            self._config_orig = copy.deepcopy(self.o._config)

        return self.o._config

    def _add_ptm_config(self):
        """Add PTM config to overall config."""

        dict1 = copy.deepcopy(self._config)
        dict2 = copy.deepcopy(self.config_ptm)

        # dictB has the od_mapping version of the keys of dict2
        # e.g.  'processes:emulsification' instead of 'emulsification'
        # dictB values are the OpenDriftModel config parameters with config_od parameters added on top
        dictB = {
            v["od_mapping"]: (
                {**dict1[v["od_mapping"]], **v}
                if "od_mapping" in v and v["od_mapping"] in dict1.keys()
                else {**v}
            )
            for k, v in dict2.items()
            if "od_mapping" in v
        }

        # dictC is the same as dictB except the names are the PTM/OpenDriftModel names instead of the
        # original OpenDrift names
        dictC = {
            k: {**dict1[v["od_mapping"]], **v}
            if "od_mapping" in v and v["od_mapping"] in dict1.keys()
            else {**v}
            for k, v in dict2.items()
            if "od_mapping" in v
        }

        # this step copies in parameter info from config_ptm to _config
        self._config.update(dict2)

        # this step brings config overrides from config_ptm into the overall config
        self._config.update(dictB)
        # this step brings other model config (plus additions from mapped parameter config) into the overall config
        self._config.update(dictC)
        # # this step brings other model config into the overall config
        # self._config.update(dict2)

    def _add_model_config(self):
        """Goal is to combine the config both directions:

        * override OpenDrift config defaults with those from opendrift_config as well
          as include extra information like ptm_level
        * bring OpenDrift config parameter metadata into config_model so application
          could query it to get the ranges, options, etc.
        """

        dict1 = copy.deepcopy(self._config)
        dict2 = copy.deepcopy(self.config_model)

        # dictB has the od_mapping version of the keys of dict2
        # e.g.  'processes:emulsification' instead of 'emulsification'
        # dictB values are the OpenDrift config parameters with config_od parameters added on top
        dictB = {
            v["od_mapping"]: {**dict1[v["od_mapping"]], **v}
            if "od_mapping" in v and v["od_mapping"] in dict1.keys()
            else {**v}
            for k, v in dict2.items()
            if "od_mapping" in v
        }

        # dictC is the same as dictB except the names are the PTM/OpenDriftModel names instead of the
        # original OpenDrift names
        dictC = {
            k: {**dict1[v["od_mapping"]], **v}
            if "od_mapping" in v and v["od_mapping"] in dict1.keys()
            else {**v}
            for k, v in dict2.items()
            if "od_mapping" in v
        }

        # this step copies in parameter info from config_ptm to _config
        self._config.update(dict2)

        # this step brings config overrides from config_model into the overall config
        self._config.update(dictB)
        # this step brings other model config (plus additions from mapped parameter config) into the overall config
        self._config.update(dictC)

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

    def drift_model_config(self, ptm_level=[1, 2, 3], prefix=""):
        """Show config for this drift model selection.

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

    def get_configspec(self, prefix, substring, excludestring, level, ptm_level):
        """Copied from OpenDrift, then modified."""

        if not isinstance(level, list) and level is not None:
            level = [level]
        if not isinstance(ptm_level, list) and ptm_level is not None:
            ptm_level = [ptm_level]

        # check for prefix or substring comparison
        configspec = {
            k: v
            for (k, v) in self._config.items()
            if k.startswith(prefix) and substring in k and excludestring not in k
        }

        if level is not None:
            # check for levels (if present)
            configspec = {
                k: v
                for (k, v) in configspec.items()
                if "level" in configspec[k] and configspec[k]["level"] in level
            }

        if ptm_level is not None:
            # check for ptm_levels (if present)
            configspec = {
                k: v
                for (k, v) in configspec.items()
                if "ptm_level" in configspec[k]
                and configspec[k]["ptm_level"] in ptm_level
            }

        return configspec

    def show_config_model(
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

        if key is not None:
            if key in output:
                return output[key]
            else:
                return output
        else:
            return output

    def reader_metadata(self, key):
        """allow manager to query reader metadata."""

        if not self.has_added_reader:
            raise ValueError("reader has not been added yet.")
        return self.o.env.readers[self.ocean_model].__dict__[key]

    @property
    def outfile_name(self):
        """Output file name."""

        return self.o.outfile_name
