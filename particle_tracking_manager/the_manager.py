"""Defines the ParticleTrackingManager class which is the base class for any Lagrangian model to inherit from."""

# Standard library imports
import logging

from abc import ABC, abstractmethod

# Third-party imports
from typing import Any, Self

# Local imports
from .config_logging import LoggerConfig
from .config_misc import ParticleTrackingState, SetupOutputFiles
from .config_the_manager import TheManagerConfig


logger = logging.getLogger()


class ParticleTrackingManager(ABC):
    """Manager class that controls particle tracking model.

    Parameters
    ----------
    model : str
        Name of Lagrangian model package to use for drifter tracking. Only option
        currently is "opendrift".
    lon : Optional[Union[int,float]], optional
        Central longitude for seeding drifters. If this is set, `lat` should also be set, and `geojson` should be None.
    lat : Optional[Union[int,float]], optional
        Central latitude for seeding drifters. If this is set, `lon` should also be set, and `geojson` should be None.
    geojson : Optional[dict], optional
        GeoJSON describing a polygon within which to seed drifters; must contain "geometry". If this is set, `lon` and `lat` should be None.
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
    do3D : bool, optional
        Set to True to run drifters in 3D, by default False. This is overridden if
        ``surface_only==True``. If True, vertical advection and mixing are turned on with
        options for setting ``diffusivitymodel``, ``background_diffusivity``,
        ``ocean_mixed_layer_thickness``, ``vertical_mixing_timestep``. If False,
        vertical motion is disabled.
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
    horizontal_diffusivity : float
        Horizontal diffusivity is None by default but will be set to a grid-dependent value for known ocean_model values. This is calculated as 0.1 m/s sub-gridscale velocity that is missing from the model output and multiplied by an estimate of the horizontal grid resolution. This leads to a larger value for NWGOA which has a larger value for mean horizontal grid resolution (lower resolution). If the user inputs their own ocean_model information, they can input their own horizontal_diffusivity value. A user can use a known ocean_model and then overwrite the horizontal_diffusivity value to some value.
    log_level : str, optional
        Options are the logging input options. By default "INFO"

    """

    config: TheManagerConfig

    def __init__(self, **kwargs: dict) -> None:
        """Initialize the ParticleTrackingManager."""

        # Set up strings for the output files, which will be used in Logger setup and for all other output files.
        inputs = {
            key: kwargs[key]
            for key in ["output_file", "output_format"]
            if key in kwargs
        }
        self.files = SetupOutputFiles(**inputs)

        # Setup logging, this also contains the log_level parameter
        inputs = {key: kwargs[key] for key in ["log_level"] if key in kwargs}
        self.logger_config = LoggerConfig(**inputs)
        self.logger_config.setup_logger(logfile_name=self.files.logfile_name)
        self.state = ParticleTrackingState()

    @classmethod
    def from_config(cls, config: TheManagerConfig) -> Self:
        """Create an OpenDriftModel from a config.

        Not currently working.
        """
        return cls(**config.dict())

    def setup_for_simulation(self) -> None:
        """Set up the simulation.

        This may not be necessary to separate out but is sometimes necessary.
        """

        self._setup_for_simulation()

        # Set up state
        self.state.has_run_setup = True

    def add_reader(self, **kwargs: dict) -> None:
        """Add reader to model class."""
        self._add_reader(**kwargs)

        self.state.has_added_reader = True

    def seed(self) -> None:
        """Seed drifters."""

        if not self.state.has_added_reader:
            raise ValueError("first add reader with `manager.add_reader(**kwargs)`.")

        # run seeding function in model class
        self._seed()  # in child class

        self.state.has_run_seeding = True

    def run(self) -> None:
        """Call model run_drifters function.

        Also run some other items.
        """

        if not self.state.has_run_seeding:
            raise ValueError("first run seeding with `manager.seed()`.")

        logger.info(
            f"start_time: {self.config.start_time}, end_time: {self.config.end_time}, steps: {self.config.steps}, duration: {self.config.duration}"
        )

        self._run()  # in child class

        self.logger_config.close_loggers(logger)
        self.state.has_run = True

    def run_all(self) -> None:
        """Run all steps."""
        if not self.state.has_added_reader:
            self.add_reader()
        if not self.state.has_run_seeding:
            self.seed()
        if not self.state.has_run:
            self.run()

    @abstractmethod
    def _setup_for_simulation(self) -> None:
        """Steps to setup for specific model's simulation."""
        raise NotImplementedError("This should be implemented in the model class.")

    @abstractmethod
    def _add_reader(self, **kwargs: Any) -> None:
        """Add reader to model class."""
        raise NotImplementedError("This should be implemented in the model class.")

    @abstractmethod
    def _seed(self) -> None:
        """Seed drifters in model class."""
        raise NotImplementedError("This should be implemented in the model class.")

    @abstractmethod
    def _run(self) -> None:
        """Run drifter simulation in model class."""
        raise NotImplementedError("This should be implemented in the model class.")
