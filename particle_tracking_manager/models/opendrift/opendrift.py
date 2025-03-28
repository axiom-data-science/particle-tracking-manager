"""Using OpenDrift for particle tracking."""
import json
import logging
from enum import Enum
from pathlib import Path

from opendrift.readers import reader_ROMS_native

# from ...config_ocean_model import _KNOWN_MODELS
from ...ocean_model_registry import ocean_model_registry
from .config_opendrift import open_drift_mapper
from ...the_manager import ParticleTrackingManager
from .plot import make_plots
from .utils import narrow_dataset_to_simulation_time, \
    apply_known_ocean_model_specific_changes, apply_user_input_ocean_model_specific_changes


logger = logging.getLogger()


class OpenDriftModel(ParticleTrackingManager):
    """OpenDrift particle tracking model.
    
    Parameters
    ----------
    drift_model : str
        Options: "OceanDrift", "LarvalFish", "OpenOil", "Leeway"
    export_variables : list
        List of variables to export, by default None. See PTM docs for options.
    radius : int
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
    stokes_drift : bool
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
    interpolator_filename : Optional[Union[pathlib.Path,str]], optional
        Filename to save interpolators to, by default None. The full path should be given, but no suffix.
        Use this to either read from an existing file at a non-default location or to save to a
        non-default location. If None and use_cache==True, the filename is set to a built-in name to an
        `appdirs` cache directory.
    plots : dict, optional
        Dictionary of plot names, their filetypes, and any kwargs to pass along, by default None.
        Available plot names are "spaghetti", "animation", "oil", "all".


    object_type: str = config_model["object_type"]["default"],
        Leeway object category for this simulation.

    diameter : float
        Seeding value of diameter. For LarvalFish simulation.
    neutral_buoyancy_salinity : float
        Seeding value of neutral_buoyancy_salinity. For LarvalFish simulation.
    stage_fraction : float
        Seeding value of stage_fraction. For LarvalFish simulation.
    hatched : float
        Seeding value of hatched. For LarvalFish simulation.
    length : float
        Seeding value of length. For LarvalFish simulation.
    weight : float
        Seeding value of weight. For LarvalFish simulation.

    oil_type : str
        Oil type to be used for the simulation, from the NOAA ADIOS database. For OpenOil simulation.
    m3_per_hour : float
        The amount (volume) of oil released per hour (or total amount if release is instantaneous). For OpenOil simulation.
    oil_film_thickness : float
        Seeding value of oil_film_thickness. For OpenOil simulation.
    droplet_size_distribution : str
        Droplet size distribution used for subsea release. For OpenOil simulation.
    droplet_diameter_mu : float
        The mean diameter of oil droplet for a subsea release, used in normal/lognormal distributions. For OpenOil simulation.
    droplet_diameter_sigma : float
        The standard deviation in diameter of oil droplet for a subsea release, used in normal/lognormal distributions. For OpenOil simulation.
    droplet_diameter_min_subsea : float
        The minimum diameter of oil droplet for a subsea release, used in uniform distribution. For OpenOil simulation.
    droplet_diameter_max_subsea : float
        The maximum diameter of oil droplet for a subsea release, used in uniform distribution. For OpenOil simulation.
    emulsification : bool
        Surface oil is emulsified, i.e. water droplets are mixed into oil due to wave mixing, with resulting increase of viscosity. For OpenOil simulation.
    dispersion : bool
        Oil is removed from simulation (dispersed), if entrained as very small droplets. For OpenOil simulation.
    evaporation : bool
        Surface oil is evaporated. For OpenOil simulation.
    update_oilfilm_thickness : bool
        Oil film thickness is calculated at each time step. The alternative is that oil film thickness is kept constant with value provided at seeding. For OpenOil simulation.
    biodegradation : bool
        Oil mass is biodegraded (eaten by bacteria). For OpenOil simulation.
    """

    def __init__(self, **kwargs):
        """Initialize OpenDriftModel."""

        # Initialize the parent class
        # This sets up the logger and ParticleTrackingState and SetupOutputFiles.
        super().__init__(**kwargs)
        
        # OpenDriftConfig is a subclass of TheManagerConfig so it knows about all the
        # TheManagerConfig parameters. TheManagerConfig is run with OpenDriftConfig.
        drift_model = kwargs.get("drift_model", "OceanDrift")
        if "drift_model" in kwargs:
            del kwargs["drift_model"]
        self.config = open_drift_mapper[drift_model](**kwargs)

        
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


    def _check_interpolator_filename_exists(self):
        if Path(self.config.interpolator_filename).exists():
            logger.info(f"Will load the interpolator from {self.config.interpolator_filename}.")
        else:
            logger.info(f"A new interpolator will be saved to {self.config.interpolator_filename}.")

    def _create_opendrift_model_object(self):
        # do this right away so I can query the object
        # we don't actually input output_format here because we first output to netcdf, then
        # resave as parquet after adding in extra config
        # TODO: should drift_model be instantiated in OpenDriftConfig or here?
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
        """Update OpenDrift's config values with OpenDriftConfig and TheManagerConfig.
        
        Update the default value in OpenDrift's config dict with the 
        config value from OpenDriftConfig, TheManagerConfig, OceanModelConfig, and SetupOutputFiles.
        
        This uses the metadata key "od_mapping" to map from the PTM parameter
        name to the OpenDrift parameter name.
        """
        
        base_models_to_check = [self.config, self.files, self.config.ocean_model_config]
        for base_model in base_models_to_check:
            for key in base_model.model_fields:
                if getattr(base_model.model_fields[key], "json_schema_extra") is not None:
                    if "od_mapping" in base_model.model_fields[key].json_schema_extra:
                        od_key = base_model.model_fields[key].json_schema_extra["od_mapping"]
                        if od_key in self.o._config:# and od_key is not None:
                            field_value = getattr(base_model, key)
                            if isinstance(field_value, Enum):
                                field_value = field_value.value
                            self.o._config[od_key]["value"] = field_value
                                
    def _modify_opendrift_model_object(self):
        if self.config.stokes_drift:
            self.o.set_config("drift:use_tabularised_stokes_drift", True)
            # self.o.set_config('drift:tabularised_stokes_drift_fetch', '25000')  # default
            # self.o.set_config('drift:stokes_drift_profile', 'Phillips')  # default

        # If 2D surface simulation (and not Leeway since not available), truncate model output below 0.5 m
        if not self.config.do3D and self.config.z == 0 and self.config.drift_model != "Leeway":
            self.o.set_config("drift:truncate_ocean_model_below_m", 0.5)
            logger.info("Truncating model output below 0.5 m.")


        # If 2D simulation (and not Leeway since not available), turn off vertical advection
        if not self.config.do3D and self.config.drift_model != "Leeway":
            self.o.set_config("drift:vertical_advection", False)
            logger.info("Disabling vertical advection.")

        # If 3D simulation, turn on vertical advection
        if self.config.do3D:
            self.o.set_config("drift:vertical_advection", True)
            logger.info("do3D is True so turning on vertical advection.")

    def _setup_for_simulation(self):

        self.logger_config.merge_with_opendrift_log()
        self._check_interpolator_filename_exists()
        self._create_opendrift_model_object()
        self._update_od_config_from_this_config()
        self._modify_opendrift_model_object()


    def _add_reader(
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
        """
        
        self._setup_for_simulation()
        
        # TODO: have standard_name_mapping as an initial input only with initial call to OpenDrift?
        # TODO: has ds as an initial input for user-input ds?
        # if (
        #     self.config.ocean_model_config.name not in ocean_model_registry.all()
        #     and self.config.ocean_model_config.name != "test"
        #     and ds is None
        # ):
        #     raise ValueError(
        #         "ocean_model must be a known model or user must input a Dataset."
        #     )

        # # user-input ds
        # if ds is not None:
        #     if name is None:
        #         self.config.ocean_model_config.name = "user_input"
        #     else:
        #         self.config.ocean_model_config.name = name

        # TODO: do I still need a pathway for ocean_model of "test"?
        # TODO: move tests from test_manager to other files
        if ds is None:
            ds = self.config.ocean_model_simulation.open_dataset(drop_vars=self.config.drop_vars)
        
        # don't need the following currently if using ocean_model_local since the kerchunk file is already 
        # narrowed to the simulation size
        if not self.config.ocean_model_local:
            ds = narrow_dataset_to_simulation_time(ds, self.config.start_time, self.config.end_time)
            logger.info("Narrowed model output to simulation time")
        
        ds = apply_known_ocean_model_specific_changes(ds, self.config.ocean_model_config.name, self.config.use_static_masks)
        
        if self.config.ocean_model_config.name not in ocean_model_registry.all() and self.config.ocean_model_config.name != "test":
            ds = apply_user_input_ocean_model_specific_changes(ds, self.config.use_static_masks)

        self.ds = ds

        reader = reader_ROMS_native.Reader(
            filename=ds,
            name=self.config.ocean_model_config.name,
            standard_name_mapping=self.config.ocean_model_config.standard_name_mapping,
            save_interpolator=self.config.save_interpolator,
            interpolator_filename=self.config.interpolator_filename,
        )

        self.o.add_reader([reader])
        self.reader = reader
        # can find reader at manager.o.env.readers[self.ocean_model.name]


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

        if self.config.start_time_end is not None:
            # time can be a list to start drifters linearly in time
            time = [
                self.config.start_time.to_pydatetime(),
                self.config.start_time_end.to_pydatetime(),
            ]
        elif self.config.start_time is not None:
            time = self.config.start_time
            # time = self.config.start_time.to_pydatetime()
        else:
            time = None

        _seed_kws = {
            "time": time,
            "z": self.config.z,
        }

        # update seed_kws with drift_model-specific seed parameters
        seedlist = {k: v["value"] for k, v in self.o.get_configspec(prefix="seed").items()}
        seedlist = {one.replace("seed:", ""): two for one, two in seedlist.items() if one not in already_there}
        _seed_kws.update(seedlist)

        if self.config.seed_flag == "elements":
            # add additional seed parameters
            _seed_kws.update(
                {
                    "lon": self.config.lon,
                    "lat": self.config.lat,
                    "radius": self.config.radius,
                    "radius_type": self.config.radius_type,
                }
            )

        elif self.config.seed_flag == "geojson":

            # geojson needs string representation of time
            _seed_kws["time"] = (
                self.config.start_time.isoformat() if self.config.start_time is not None else None
            )

        self._seed_kws = _seed_kws
        return self._seed_kws


    def _seed(self):
        """Actually seed drifters for model."""

        if self.config.seed_flag == "elements":
            self.o.seed_elements(**self.seed_kws)

        elif self.config.seed_flag == "geojson":

            # # geojson needs string representation of time
            # self.seed_kws["time"] = self.config.start_time.isoformat()
            self.config.geojson["properties"] = self.seed_kws
            json_string_dumps = json.dumps(self.config.geojson)
            self.o.seed_from_geojson(json_string_dumps)

        else:
            raise ValueError(f"seed_flag {self.config.seed_flag} not recognized.")

        self.initial_drifters = self.o.elements_scheduled

    def _run(self):
        """Run the drifters!"""
        
        # add input config to model config
        self.o.metadata_dict.update(self.config.model_dump())
        self.o.metadata_dict.update(self.files.model_dump())

        # actually run
        self.o.run(
            time_step=self.config.time_step,
            time_step_output=self.config.time_step_output,
            steps=self.config.steps,
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

    # def run_all(self):
    #     """Run all steps."""
    #     if not self.state.has_added_reader:
    #         self.add_reader()
    #     if not self.state.has_run_seeding:
    #         self.seed()
    #     if not self.state.has_run:
    #         self.run()

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

    def reader_metadata(self, key):
        """allow manager to query reader metadata."""

        if not self.state.has_added_reader:
            raise ValueError("reader has not been added yet.")
        return self.o.env.readers[self.config.ocean_model_config.name].__dict__[key]
