"""Uses OpenDrift for particle tracking model."""

# Standard library imports
import json
import logging

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

# Third-party imports
from opendrift.readers import reader_ROMS_native

from particle_tracking_manager.models.opendrift.enums.oil_types import OIL_ID_TO_NAME

from ...config_the_manager import OutputFormatEnum

# Local imports
from ...ocean_model_registry import ocean_model_registry
from ...the_manager import ParticleTrackingManager
from .config_opendrift import OpenDriftConfig, open_drift_mapper
from .plot import make_plots
from .utils import (
    apply_known_ocean_model_specific_changes,
    apply_user_input_ocean_model_specific_changes,
    narrow_dataset_to_simulation_time,
)


# Initialize logger
logger = logging.getLogger()


class OpenDriftModel(ParticleTrackingManager):
    """OpenDrift particle tracking model.

    Parameters
    ----------
    drift_model : str
        Options: "OceanDrift", "LarvalFish", "OpenOil", "Leeway", "HarmfulAlgalBloom". Default is "OceanDrift".
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

    def __init__(self, **kwargs: dict) -> None:
        """Initialize OpenDriftModel."""
        # Initialize the parent class
        # This sets up the logger and ParticleTrackingState and SetupOutputFiles.
        super().__init__(**kwargs)

        # OpenDriftConfig is a subclass of TheManagerConfig so it knows about all the
        # TheManagerConfig parameters. TheManagerConfig is run with OpenDriftConfig.
        drift_model = kwargs.get("drift_model", "OceanDrift")
        assert isinstance(drift_model, str)
        if "drift_model" in kwargs:
            del kwargs["drift_model"]
        self.config: OpenDriftConfig = open_drift_mapper[drift_model](**kwargs)

        # copy output_file from files to config. This isn't strictly necessary and it
        # seems like there is a better way to do this but do it for now to avoid
        # confusion between self.files.output_file and self.config.output_file otherwise
        # being different
        self.config.output_file = self.files.output_file

        # Note that you can see configuration possibilities for a given model with
        # o.list_configspec()
        # You can check the metadata for a given configuration with (min/max/default/type)
        # o.get_configspec('vertical_mixing:timestep')
        # You can check required variables for a model with
        # o.required_variables

        # TODO: streamline this
        self.checked_plot = False

    def _create_opendrift_model_object(self) -> None:
        """Create the OpenDrift model object."""

        # do this right away so I can query the object
        # we don't actually input output_format here because we first output to netcdf, then
        # resave as parquet after adding in extra config
        # TODO: should drift_model be instantiated in OpenDriftConfig or here?
        log_level = logger.level
        iomodule = self.config.output_format
        if iomodule == "both":
            iomodule = (
                OutputFormatEnum.netcdf
            )  # first output netcdf, then additionally output parquet at the end
        if self.config.drift_model == "Leeway":
            from opendrift.models.leeway import Leeway

            o = Leeway(loglevel=log_level, iomodule=iomodule)

        elif self.config.drift_model == "OceanDrift":
            from opendrift.models.oceandrift import OceanDrift

            o = OceanDrift(loglevel=log_level, iomodule=iomodule)

        elif self.config.drift_model == "LarvalFish":
            from opendrift.models.larvalfish import LarvalFish

            o = LarvalFish(loglevel=log_level, iomodule=iomodule)

        elif self.config.drift_model == "OpenOil":
            from opendrift.models.openoil import OpenOil

            o = OpenOil(loglevel=log_level, iomodule=iomodule, weathering_model="noaa")

        elif self.config.drift_model == "HarmfulAlgalBloom":
            from opendrift.models.harmfulalgalbloom import HarmfulAlgalBloom

            o = HarmfulAlgalBloom(loglevel=log_level, iomodule=iomodule)

        else:
            raise ValueError(
                f"Drifter model {self.config.drift_model} is not recognized."
            )

        self.o = o

    def _update_od_config_from_this_config(self) -> None:
        """Update OpenDrift's config values with OpenDriftConfig and TheManagerConfig.

        Update the default value in OpenDrift's config dict with the
        config value from OpenDriftConfig, TheManagerConfig, OceanModelConfig, and SetupOutputFiles.

        This uses the metadata key "od_mapping" to map from the PTM parameter
        name to the OpenDrift parameter name.
        """

        base_models_to_check = [self.config, self.files, self.config.ocean_model_config]
        for base_model in base_models_to_check:
            for field_name, field in base_model.model_fields.items():
                if (
                    field.json_schema_extra is None
                    or "od_mapping" not in field.json_schema_extra
                ):
                    continue

                od_key = field.json_schema_extra["od_mapping"]
                if od_key not in self.o._config:
                    continue

                # want the string representation of only this one used
                if od_key == "seed:oil_type":
                    # for oil_type, copy the oil name into the OpenDrift config
                    field_value = OIL_ID_TO_NAME[getattr(base_model, field_name)]
                # for others use value
                else:
                    field_value = getattr(base_model, field_name)
                    # if isinstance(field_value, Enum):
                    #     field_value = field_value.value

                self.o._config[od_key]["value"] = field_value

    def _modify_opendrift_model_object(self) -> None:
        """Modify the OpenDrift model object."""
        if self.config.stokes_drift:
            self.o.set_config("drift:use_tabularised_stokes_drift", True)
            # self.o.set_config('drift:tabularised_stokes_drift_fetch', '25000')  # default
            # self.o.set_config('drift:stokes_drift_profile', 'Phillips')  # default

        # If 2D surface simulation (and not Leeway since not available), truncate model output below 0.5 m
        if (
            not self.config.do3D
            and self.config.z == 0
            and self.config.drift_model != "Leeway"
        ):
            self.o.set_config("drift:truncate_ocean_model_below_m", 0.5)
            logger.debug("Truncating model output below 0.5 m.")

        # If 2D simulation (and not Leeway since not available), turn off vertical advection
        if not self.config.do3D and self.config.drift_model != "Leeway":
            self.o.set_config("drift:vertical_advection", False)
            logger.debug("Disabling vertical advection.")

        # If 3D simulation, turn on vertical advection
        if self.config.do3D:
            self.o.set_config("drift:vertical_advection", True)
            logger.debug("do3D is True so turning on vertical advection.")

        # # Assign oil_type for OpenOil simulation by id to be unique since oil names are not unique
        # # WARHING! o.set_oiltype_by_id doesn't appear to actually set the oil type (only the name, not the config)
        # if self.config.drift_model == "OpenOil":
        #     oil_type_id = OilTypeEnum(self.config.oil_type).name
        #     self.o.set_oiltype_by_id(oil_type_id)
        #     logger.debug(f"Setting oil type {self.config.oil_type} by id {oil_type_id}.")

        # TODO: what if I change the order of when config is updated with this method, then
        # does it impact that i sed oil type by id?

    def _setup_for_simulation(self) -> None:
        """Set up the simulation.
        This is run before the simulation starts.
        """

        self.logger_config.merge_with_opendrift_log()
        self._create_opendrift_model_object()
        self._update_od_config_from_this_config()
        self._modify_opendrift_model_object()

    def _add_reader(self, **kwargs: Any) -> None:
        """Add a reader to the OpenDrift model.

        Parameters
        ----------
        ds : xr.Dataset, optional
            Previously-opened Dataset containing ocean model output, if user wants to input
            unknown reader information.
        name : str, optional
            If ds is input, user can also input name of ocean model, otherwise will be called "user_input".
        """
        # Extract specific parameters from kwargs
        ds = kwargs.get("ds", None)
        name = kwargs.get("name", None)
        oceanmodel_lon0_360 = kwargs.get("oceanmodel_lon0_360", False)
        standard_name_mapping = kwargs.get("standard_name_mapping", None)

        if not self.state.has_run_setup:
            self._setup_for_simulation()

        if ds is None:
            ds = self.config.ocean_model_simulation.open_dataset(
                drop_vars=self.config.drop_vars
            )

        # don't need the following currently if using ocean_model_local since the kerchunk file is already
        # narrowed to the simulation size
        if not self.config.ocean_model_local:
            assert self.config.start_time is not None
            assert self.config.end_time is not None
            ds = narrow_dataset_to_simulation_time(
                ds, self.config.start_time, self.config.end_time
            )
            logger.debug("Narrowed model output to simulation time")

        ds = apply_known_ocean_model_specific_changes(
            ds, self.config.ocean_model_config.name, self.config.use_static_masks
        )

        if (
            self.config.ocean_model_config.name not in ocean_model_registry.all()
            and self.config.ocean_model_config.name != "test"
        ):
            ds = apply_user_input_ocean_model_specific_changes(
                ds, self.config.use_static_masks
            )

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
    def seed_kws(self) -> dict:
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

        time: float | datetime | list[float] | str | None
        if self.config.start_time_end is not None:
            # time can be a list to start drifters linearly in time
            time = [
                pd.Timestamp(self.config.start_time).to_pydatetime(),
                pd.Timestamp(self.config.start_time_end).to_pydatetime(),
            ]
        elif self.config.start_time is not None:
            time = self.config.start_time
            # time = self.config.start_time.to_pydatetime()
        else:
            time = None

        if self.config.seed_flag == "geojson":
            # geojson needs string representation of time
            time = (
                self.config.start_time.isoformat()
                if self.config.start_time is not None
                else None
            )

        _seed_kws = {
            "time": time,
            "z": self.config.z,
        }

        # update seed_kws with drift_model-specific seed parameters
        seedlist = {
            k: v["value"] for k, v in self.o.get_configspec(prefix="seed").items()
        }
        seedlist = {
            one.replace("seed:", ""): two
            for one, two in seedlist.items()
            if one not in already_there
        }
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
            # add additional seed parameters
            _seed_kws.update(
                {
                    "radius": self.config.radius,
                    "radius_type": self.config.radius_type,
                }
            )

        self._seed_kws = _seed_kws
        return self._seed_kws

    def _seed(self) -> None:
        """Actually seed drifters for model."""

        if self.config.seed_flag == "elements":
            self.o.seed_elements(**self.seed_kws)

        elif self.config.seed_flag == "geojson":

            # # geojson needs string representation of time
            # self.seed_kws["time"] = self.config.start_time.isoformat()
            self.config.geojson["properties"] = self.seed_kws  # type: ignore
            json_string_dumps = json.dumps(self.config.geojson)
            self.o.seed_from_geojson(json_string_dumps)

        else:
            raise ValueError(f"seed_flag {self.config.seed_flag} not recognized.")

        self.initial_drifters = self.o.elements_scheduled

    def _run(self) -> None:
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
            outfile=str(self.files.output_file),
        )

        # if output format is both, also save as parquet
        if self.config.output_format == "both":
            assert self.files.output_file is not None
            parquet_file = Path(self.files.output_file).with_suffix(".parquet")

            # Open the netCDF as an xarray.Dataset
            ds = xr.open_dataset(self.files.output_file)

            # Expect dims (trajectory, time)
            # Convert to a tidy DataFrame: columns = trajectory, time, variables
            df = ds.to_dataframe().reset_index()

            # # Optional: drop rows with NaN lon as “unseeded”/invalid
            # if "lon" in df.columns:
            #     df = df[~df["lon"].isna()]

            # Ensure time is datetime64[ns]
            df["time"] = pd.to_datetime(df["time"])

            # Save to parquet
            df.to_parquet(parquet_file, engine="fastparquet")
            logger.info(f"Also saved output to parquet file {parquet_file}.")

        # plot if requested
        if self.config.plots:
            assert isinstance(self.files.output_file, Path)
            # return plots because now contains the filenames for each plot
            self.config.plots = make_plots(
                self.config.plots,
                self.o,
                str(self.files.output_file).split(".")[0],
                self.config.drift_model,
            )

            # convert plots dict into string representation to save in output file attributes
            # https://github.com/pydata/xarray/issues/1307
            self.config.plots = repr(self.config.plots)

    def all_export_variables(self) -> list:
        """Output list of all possible export variables."""

        if not self.state.has_run_setup:
            self.setup_for_simulation()
            logger.debug(
                "Running setup for simulation so that full export variable list is available."
            )

        vars = (
            list(self.o.elements.variables.keys())
            + ["trajectory", "time"]
            + list(self.o.required_variables.keys())
        )

        return vars

    def export_variables(self) -> list:
        """Output list of all actual export variables."""

        return self.o.export_variables

    def reader_metadata(self, key: str) -> dict:
        """allow manager to query reader metadata."""

        if not self.state.has_added_reader:
            raise ValueError("reader has not been added yet.")
        return self.o.env.readers[self.config.ocean_model_config.name].__dict__[key]
