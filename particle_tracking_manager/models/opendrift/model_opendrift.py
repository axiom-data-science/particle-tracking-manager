"""Using OpenDrift for particle tracking."""
import copy
import datetime
import json
import logging
import pathlib

from opendrift.models.larvalfish import LarvalFish
from opendrift.models.leeway import Leeway
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.openoil import OpenOil

from ...cli import is_None
from ...the_manager import ParticleTrackingManager

# from opendrift.readers import reader_ROMS_native
# using my own version of ROMS reader
from .reader_ROMS_native import Reader


# from .cli import is_None
# from .the_manager import ParticleTrackingManager


# Read OpenDrift configuration information
loc = pathlib.Path(__file__).parent / pathlib.Path("opendrift_config.json")
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
        Horizontal diffusivity is None by default but will be set to a grid-dependent value for known ocean_model values. This is calculated as 0.1 m/s sub-gridscale velocity that is missing from the model output and multiplied by an estimate of the horizontal grid resolution. This leads to a larger value for NWGOA which has a larger value for mean horizontal grid resolution (lower resolution). If the user inputs their own ocean_model information, they can input their own horizontal_diffusivity value. A user can use a built-in ocean_model and the overwrite the horizontal_diffusivity value to 0.
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
    vertical_mixing_timestep: float
    diffusivitymodel: str
    mixed_layer_depth: float
    wind_drift_factor: float
    wind_drift_depth: float
    stokes_drift: bool

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

        model = "opendrift"

        if log == "low":
            self.loglevel = 20
        elif log == "high":
            self.loglevel = 0

        # need drift_model defined for the log to work properly for both manager and model
        # so do this before super initialization
        self.drift_model = drift_model

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

        self.o = o

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

        # get all named parameters input to ParticleTrackingManager class
        from inspect import signature

        sig = signature(OpenDriftModel)
        self.config_model = config_model

        # Set all attributes which will trigger some checks and changes in __setattr__
        # these will also update "value" in the config dict
        for key in sig.parameters.keys():
            self.__setattr__(key, locals()[key])

    def __setattr_model__(self, name: str, value) -> None:
        """Implement my own __setattr__ but here to enforce actions."""

        # don't allow drift_model to be reset, have to reinitialize object instead
        # check for type of m.o and drift_model matching to enforce this
        if (
            (name in ["o", "drift_model"])
            and hasattr(self, "o")
            and (self.drift_model not in str(type(self.o)))
        ):
            raise KeyError(
                "Can't overwrite `drift_model`; instead initialize OpenDriftModel with desired drift_model."
            )

        # create/update "value" keyword in config to keep it up to date
        if (
            hasattr(self, "config_model")
            and name != "config_model"
            and name != "config_ptm"
            and name in self.config_model.keys()
        ):
            self.config_model[name]["value"] = value
        self._update_config()

        # if user sets ocean_model and horizontal_diffusivity is set up, overwrite it
        if name == "ocean_model":
            if value in ["NWGOA", "CIOFS", "CIOFS_now"] and hasattr(
                self, "horizontal_diffusivity"
            ):

                self.logger.info(
                    "overriding horizontal_diffusivity parameter with one tuned to reader model"
                )

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

                self.horizontal_diffusivity = horizontal_diffusivity

            # if user not using a known ocean_model, change horizontal_diffusivity from None to 0
            # so it has a value. User can subsequently overwrite it too.
            elif (
                hasattr(self, "horizontal_diffusivity")
                and self.horizontal_diffusivity is None
            ):

                self.logger.info(
                    "changing horizontal_diffusivity parameter from None to 0.0. Otherwise set it to a specific value."
                )

                self.horizontal_diffusivity = 0

        # if user sets horizontal_diffusivity as None and ocean_model is set, overwrite horizontal_diffusivity
        # if user changes horizontal_diffusivity subsequently without changing model, allow it
        if name == "horizontal_diffusivity" and value is None:
            if hasattr(self, "ocean_model") and self.ocean_model in [
                "NWGOA",
                "CIOFS",
                "CIOFS_now",
            ]:

                self.logger.info(
                    "overriding horizontal_diffusivity parameter with one tuned to reader model"
                )

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

                # when editing the __dict__ directly have to also update config_model
                self.__dict__["horizontal_diffusivity"] = horizontal_diffusivity
                self.config_model["horizontal_diffusivity"][
                    "value"
                ] = horizontal_diffusivity

            # if user not using a known ocean_model, change horizontal_diffusivity from None to 0
            # so it has a value. User can subsequently overwrite it too.
            elif hasattr(self, "ocean_model") and self.ocean_model not in [
                "NWGOA",
                "CIOFS",
                "CIOFS_now",
            ]:

                self.logger.info(
                    "changing horizontal_diffusivity parameter from None to 0.0. Otherwise set it to a specific value."
                )

                self.__dict__["horizontal_diffusivity"] = 0
                self.config_model["horizontal_diffusivity"]["value"] = 0

        # turn on other things if using stokes_drift
        if name == "stokes_drift" and value:
            if hasattr(self, "drift_model") and self.drift_model != "Leeway":
                self.o.set_config("drift:use_tabularised_stokes_drift", True)
            # self.o.set_config('drift:tabularised_stokes_drift_fetch', '25000')  # default
            # self.o.set_config('drift:stokes_drift_profile', 'Phillips')  # default

        # too soon to do this, need to run it later
        # Leeway model doesn't have this option built in
        if (
            name == "surface_only"
            and hasattr(self, "drift_model")
            and hasattr(self, "o")
        ) or (
            name == "drift_model"
            and hasattr(self, "surface_only")
            and hasattr(self, "o")
        ):
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
        if (
            name == "do3D"
            and not value
            and hasattr(self, "drift_model")
            and self.drift_model != "Leeway"
        ):
            self.o.disable_vertical_motion()
        elif name == "do3D" and value:
            self.o.set_config("drift:vertical_advection", True)

        # Make sure vertical_mixing_timestep equal to default value if vertical_mixing False
        # same for diffusivitymodel and mixed_layer_depth
        if hasattr(self, "vertical_mixing") and not self.vertical_mixing:
            if (
                hasattr(self, "vertical_mixing_timestep")
                and self.vertical_mixing_timestep
                != self.show_config(key="vertical_mixing_timestep")["default"]
            ):
                self.logger.info(
                    "`vertical_mixing_timestep` is not used if `vertical_mixing` is False, resetting value to default and not using."
                )
                self.vertical_mixing_timestep = self.show_config(
                    key="vertical_mixing_timestep"
                )["default"]
            if (
                hasattr(self, "diffusivitymodel")
                and self.diffusivitymodel
                != self.show_config(key="diffusivitymodel")["default"]
            ):
                self.logger.info(
                    "`diffusivitymodel` is not used if `vertical_mixing` is False, resetting value to default and not using."
                )
                self.diffusivitymodel = self.show_config(key="diffusivitymodel")[
                    "default"
                ]
            if (
                hasattr(self, "mixed_layer_depth")
                and self.mixed_layer_depth
                != self.show_config(key="mixed_layer_depth")["default"]
            ):
                self.logger.info(
                    "`mixed_layer_depth` is not used if `vertical_mixing` is False, resetting value to default and not using."
                )
                self.mixed_layer_depth = self.show_config(key="mixed_layer_depth")[
                    "default"
                ]

        # make sure user isn't try to use Leeway and "wind_drift_factor", "stokes_drift",
        # "wind_drift_depth" at the same time
        if self.drift_model == "Leeway":
            if (
                hasattr(self, "wind_drift_factor")
                and self.wind_drift_factor
                != self.show_config(key="wind_drift_factor")["default"]
            ):
                self.logger.info(
                    "wind_drift_factor cannot be used with Leeway model, resetting value to default and not using."
                )
                self.wind_drift_factor = self.show_config(key="wind_drift_factor")[
                    "default"
                ]
            if (
                hasattr(self, "wind_drift_depth")
                and self.wind_drift_depth
                != self.show_config(key="wind_drift_depth")["default"]
            ):
                self.logger.info(
                    "wind_drift_depth cannot be used with Leeway model, resetting value to default and not using."
                )
                self.wind_drift_depth = self.show_config(key="wind_drift_depth")[
                    "default"
                ]
            if hasattr(self, "stokes_drift") and self.stokes_drift:
                self.logger.info(
                    "stokes_drift cannot be used with Leeway model, changing to False."
                )
                self.stokes_drift = False

        self._update_config()

    def run_add_reader(
        self,
        loc=None,
        kwargs_xarray=None,
        oceanmodel_lon0_360=False,
    ):
        """Might need to cache this if its still slow locally.

        Parameters
        ----------
        loc : str
            Location of ocean model output, if user wants to input unknown reader information.
        kwargs_xarray : dict
            Keywords for reading in ocean model output with xarray, if user wants to input unknown reader information.
        oceanmodel_lon0_360 : bool
            True if ocean model longitudes span 0 to 360 instead of -180 to 180.
        """

        # ocean_model = self.ocean_model
        kwargs_xarray = kwargs_xarray or {}

        if loc is not None and self.ocean_model is None:
            self.ocean_model = "user_input"
        # import pdb; pdb.set_trace()
        if self.ocean_model.upper() == "TEST":
            pass
            # oceanmodel_lon0_360 = True
            # loc = "test"
            # kwargs_xarray = dict()

        elif self.ocean_model is not None or loc is not None:
            if self.ocean_model == "NWGOA":
                oceanmodel_lon0_360 = True
                loc = "http://xpublish-nwgoa.srv.axds.co/datasets/nwgoa_all/zarr/"
                kwargs_xarray = dict(engine="zarr", chunks={"ocean_time": 1})
            elif self.ocean_model == "CIOFS":
                oceanmodel_lon0_360 = False
                loc = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"
                kwargs_xarray = dict(engine="zarr", chunks={"ocean_time": 1})
                reader = Reader(filename=loc, kwargs_xarray=kwargs_xarray)
            elif self.ocean_model == "CIOFS_now":
                pass
                # loc = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"
                # kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
                # reader = Reader(loc, kwargs_xarray=kwargs_xarray)

            reader = Reader(filename=loc, kwargs_xarray=kwargs_xarray)
            self.o.add_reader([reader])
            self.reader = reader
            # can find reader at manager.o.env.readers['roms native']

            self.oceanmodel_lon0_360 = oceanmodel_lon0_360

        else:
            raise ValueError("reader did not set an ocean_model")

    def run_seed(self):
        """Seed drifters for model."""

        seed_kws = {
            "time": self.start_time.to_pydatetime(),
            "z": self.z,
        }

        if self.seed_flag == "elements":
            # add additional seed parameters
            seed_kws.update(
                {
                    "lon": self.lon,
                    "lat": self.lat,
                    "radius": self.radius,
                    "radius_type": self.radius_type,
                }
            )

            self.o.seed_elements(**seed_kws)

        elif self.seed_flag == "geojson":

            # geojson needs string representation of time
            seed_kws["time"] = self.start_time.isoformat()
            self.geojson["properties"] = seed_kws
            json_string_dumps = json.dumps(self.geojson)
            self.o.seed_from_geojson(json_string_dumps)

        else:
            raise ValueError(f"seed_flag {self.seed_flag} not recognized.")

        self.seed_kws = seed_kws
        self.initial_drifters = self.o.elements_scheduled

    def run_drifters(self):
        """Run the drifters!"""

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
        self.o.run(
            time_step=timedir * self.time_step,
            steps=self.steps,
            export_variables=self.export_variables,
            outfile=f"output-results_{datetime.datetime.utcnow():%Y-%m-%dT%H%M:%SZ}.nc",
        )

        self.o._config = full_config  # reinstate config

    @property
    def _config(self):
        """Surface the model configuration."""

        # save for reinstatement when running the drifters
        if not hasattr(self, "_config_orig"):
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
        return self.o.env.readers["roms native"].__dict__[key]

    @property
    def outfile_name(self):
        """Output file name."""

        return self.o.outfile_name
