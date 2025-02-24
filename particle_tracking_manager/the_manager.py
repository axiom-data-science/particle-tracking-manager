from abc import ABC, abstractmethod
from pathlib import Path

from .config import PTMConfig, ParticleTrackingState, config_data
from .config_logging import LoggerConfig


class ParticleTrackingManager(ABC):
    """Manager class that controls particle tracking model.

    Parameters
    ----------
    model : str
        Name of Lagrangian model package to use for drifter tracking. Only option
        currently is "opendrift".
    lon : Optional[Union[int,float]], optional
        Longitude of center of initial drifter locations, by default None. Use with `seed_flag="elements"`.
    lat : Optional[Union[int,float]], optional
        Latitude of center of initial drifter locations, by default None. Use with `seed_flag="elements"`.
    geojson : Optional[dict], optional
        GeoJSON object defining polygon for seeding drifters, by default None. Use with `seed_flag="geojson"`.
    seed_flag : str, optional
        Flag for seeding drifters. Options are "elements", "geojson". Default is "elements".
    z : Union[int,float], optional
        Depth of initial drifter locations, by default 0 but taken from the
        default in the model. Values are overridden if
        ``surface_only==True`` to 0 and to the seabed if ``seed_seafloor`` is True.
        Depth is negative downward in OpenDrift.
    seed_seafloor : bool, optional
        Set to True to seed drifters vertically at the seabed, default is False. If True
        then value of z is set to None and ignored.
    number : int
        Number of drifters to simulate. Default is 100.
    start_time : Optional[str,datetime.datetime,pd.Timestamp], optional
        Start time of simulation, by default None
    start_time_end : Optional[str,datetime.datetime,pd.Timestamp], optional
        If not None, this creates a range of start times for drifters, starting with
        `start_time` and ending with `start_time_end`. Drifters will be initialized linearly
        between the two start times. Default None.
    run_forward : bool, optional
        True to run forward in time, False to run backward, by default True
    time_step : int, optional
        Time step in seconds, options >0, <86400 (1 day in seconds), by default 300.
    time_step_output : int, Timedelta, optional
        How often to output model output. Should be a multiple of time_step.
        By default 3600.
    steps : int, optional
        Number of time steps to run in simulation. Options >0.
        steps, end_time, or duration must be input by user. By default steps is 3 and
        duration and end_time are None. Only one of steps, end_time, or duration can be
        non-None at initialization time. If one of steps, end_time, or duration is input
        later, it will be used to overwrite the three parameters according to that newest
        parameter.
    duration : Optional[datetime.timedelta], optional
        Length of simulation to run, as positive-valued timedelta object, in hours,
        such as ``timedelta(hours=48)``.
        steps, end_time, or duration must be input by user. By default steps is 3 and
        duration and end_time are None. For CLI, input duration as a pandas Timedelta
        string like "48h" for 48 hours. Only one of steps, end_time, or duration can be
        non-None at initialization time. If one of steps, end_time, or duration is input
        later, it will be used to overwrite the three parameters according to that newest
        parameter.

    end_time : Optional[datetime], optional
        Datetime at which to end simulation, as positive-valued datetime object.
        steps, end_time, or duration must be input by user. By default steps is 3 and
        duration and end_time are None. Only one of steps, end_time, or duration can be
        non-None at initialization time. If one of steps, end_time, or duration is input
        later, it will be used to overwrite the three parameters according to that newest
        parameter.

    ocean_model : Optional[str], optional
        Name of ocean model to use for driving drifter simulation, by default None.
        Use None for testing and set up. Otherwise input a string.
        Options are: "NWGOA", "CIOFS", "CIOFSOP".
        Alternatively keep as None and set up a separate reader (see example in docs).
    ocean_model_local : Optional, bool
        Set to True to use local version of known `ocean_model` instead of remote version.
    surface_only : bool, optional
        Set to True to keep drifters at the surface, by default None.
        If this flag is set to not-None, it overrides do3D to False, vertical_mixing to False,
        and the z value(s) 0.
        If True, this flag also turns off reading model output below 0.5m if
        drift_model is not Leeway:
        ``o.set_config('drift:truncate_ocean_model_below_m', 0.5)`` to save time.
    do3D : bool, optional
        Set to True to run drifters in 3D, by default False. This is overridden if
        ``surface_only==True``. If True, vertical advection and mixing are turned on with
        options for setting ``diffusivitymodel``, ``background_diffusivity``,
        ``ocean_mixed_layer_thickness``, ``vertical_mixing_timestep``. If False,
        vertical motion is disabled.
    vertical_mixing : bool, optional
        Set to True to include vertical mixing, by default False. This is overridden if
        ``surface_only==True``.
    use_static_masks : bool, optional
        Set to True to use static masks ocean_model output when ROMS wetdry masks are available, by default False.
        This is relevant for all of the available known models. If you want to use static masks
        with a user-input ocean_model, you can drop the wetdry_mask_rho etc variables from the
        dataset before inputting to PTM. Setting this to True may save computation time but
        will be less accurate, especially in the tidal flat regions of the model.
    output_file : Optional[str], optional
        Name of output file to save, by default None. If None, default is set in the model. Without any suffix.
    output_format : str, default "netcdf"
        Name of input/output module type to use for writing Lagrangian model output. Default is "netcdf".
    use_cache : bool
        Set to True to use cache for saving interpolators, by default True.
    interpolator_filename : Optional[Union[pathlib.Path,str]], optional
        Filename to save interpolators to, by default None. The full path should be given, but no suffix.
        Use this to either read from an existing file at a non-default location or to save to a
        non-default location. If None and use_cache==True, the filename is set to a built-in name to an
        `appdirs` cache directory.
    wind_drift_factor : float
        Elements at surface are moved with this fraction of the wind vector, in addition to currents and Stokes drift.
    stokes_drift : bool, optional
        Set to True to turn on Stokes drift, by default True.
    horizontal_diffusivity : float
        Horizontal diffusivity is None by default but will be set to a grid-dependent value for known ocean_model values. This is calculated as 0.1 m/s sub-gridscale velocity that is missing from the model output and multiplied by an estimate of the horizontal grid resolution. This leads to a larger value for NWGOA which has a larger value for mean horizontal grid resolution (lower resolution). If the user inputs their own ocean_model information, they can input their own horizontal_diffusivity value. A user can use a known ocean_model and then overwrite the horizontal_diffusivity value to some value.
    log : str, optional
        Options are "low" and "high" verbosity for log, by default "low"

    Notes
    -----
    Configuration happens at initialization time for the child model. There is currently
    no separate configuration step.
    """
    
    # TODO: update docs and tests to not demonstrate doing things in steps
    # since won't be able to anymore

    def __init__(self, **kwargs):

        # output_file is processed in setup_logger() so it is put into kwargs so it can be subsequently
        # used in PTMConfig. This is not ideal to have one configuration parameter dealt with first but makes it so 
        # that the logger can be set up before the rest of the configuration is processed and used during configuration.
        self.logger, self.output_file = LoggerConfig().setup_logger(output_file=kwargs.get("output_file", config_data["output_file"]["default"]), 
                                                                         log_level=kwargs.get("log_level", config_data["log_level"]["default"]))

        self.logfile_name = Path(self.logger.handlers[0].baseFilename).name
        self.state = ParticleTrackingState()
        
        # TODO: alphabetize config files
        # TODO: update docstrings
    

    @abstractmethod
    def add_reader(self, **kwargs):
        """Here is where the model output is opened.
        
        Subclasses must implement this method and:
        
        * Set `self.has_added_reader = True` at the end of the method.
        """
        pass

    # TODO: change methods to _methods as appropriate

    @abstractmethod
    def seed(self, lon=None, lat=None, z=None):
        """Initialize the drifters in space and time.
        
        Subclasses must implement this method and:
        
        * Raise a ValueError if not self.has_added_reader
        * Set `self.has_run_seeding = True` at the end of the method.
        """
        pass

    @abstractmethod
    def run(self):
        """Call model run function.
        
        Subclasses must implement this method and:
        
        * Raise a ValueError if not self.has_run_seeding
        * Set `self.has_run = True` at the end of the method.
        * Should close the file handler and logger.
        """
        pass

    def run_all(self):
        """Run all steps."""
        if not self.state.has_added_reader:
            self.add_reader()
        if not self.state.has_run_seeding:
            self.seed()
        if not self.state.has_run:
            self.run()

    # def output(self):
    #     """Hold for future output function."""
    #     pass

    @abstractmethod
    def _config(self):
        """Model should have its own version which returns variable config."""
        pass

    # @abstractmethod
    # def _add_ptm_config(self):
    #     """Have this in the model class to modify config."""
    #     pass

    # @abstractmethod
    # def _add_model_config(self):
    #     """Have this in the model class to modify config."""
    #     pass

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

    # @abstractmethod
    # def query_reader(self):
    #     """Define in child class."""
    #     pass

    @abstractmethod
    def all_export_variables(self):
        """Output list of all possible export variables."""
        pass

    @abstractmethod
    def export_variables(self):
        """Output list of all actual export variables."""
        pass

    # this is fully handled by the field output_file in PTMConfig
    # @property
    # @abstractmethod
    # def outfile_name(self):
    #     """Output file name."""
    #     pass