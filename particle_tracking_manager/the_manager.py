"""Contains logic for configuring particle tracking simulations."""


# from docstring_inheritance import NumpyDocstringInheritanceMeta
import datetime
import json
import logging
import pathlib

from typing import Optional, Union

import pandas as pd

from .cli import is_None


_KNOWN_MODELS = [
    "NWGOA",
    "CIOFS",
    "CIOFSFRESH",
    "CIOFSOP",
]

# Read PTM configuration information

loc = pathlib.Path(__file__).parent / pathlib.Path("the_manager_config.json")
with open(loc, "r") as f:
    # Load the JSON file into a Python object
    config_ptm = json.load(f)

# convert "None"s to Nones
for key in config_ptm.keys():
    if is_None(config_ptm[key]["default"]):
        config_ptm[key]["default"] = None


ciofs_operational_start_time = datetime.datetime(2021, 8, 31, 19, 0, 0)
ciofs_operational_end_time = (pd.Timestamp.now() + pd.Timedelta("48H")).to_pydatetime()
ciofs_end_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
nwgoa_end_time = datetime.datetime(2009, 1, 1, 0, 0, 0)
overall_start_time = datetime.datetime(1999, 1, 1, 0, 0, 0)
overall_end_time = ciofs_operational_end_time


class ParticleTrackingManager:
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
        Name of output file to save, by default None. If None, default is set in the model. With ".nc" suffix.

    Notes
    -----
    Configuration happens at initialization time for the child model. There is currently
    no separate configuration step.
    """

    logger: logging.Logger
    ocean_model: str
    lon: Union[int, float]
    lat: Union[int, float]
    surface_only: Optional[bool]
    z: Optional[Union[int, float]]
    start_time: Optional[Union[str, datetime.datetime, pd.Timestamp]]
    start_time_end: Optional[Union[str, datetime.datetime, pd.Timestamp]]
    steps: Optional[int]
    time_step: int
    duration: Optional[datetime.timedelta]
    end_time: Optional[datetime.datetime]
    timedir: int
    config_ptm: dict
    config_model: Optional[dict]
    seed_seafloor: bool

    def __init__(
        self,
        model: str,
        lon: Optional[Union[int, float]] = None,
        lat: Optional[Union[int, float]] = None,
        geojson: Optional[dict] = None,
        seed_flag: str = config_ptm["seed_flag"]["default"],
        z: Optional[Union[int, float]] = config_ptm["z"]["default"],
        seed_seafloor: bool = config_ptm["seed_seafloor"]["default"],
        number: int = config_ptm["number"]["default"],
        start_time: Optional[Union[str, datetime.datetime, pd.Timestamp]] = None,
        start_time_end: Optional[Union[str, datetime.datetime, pd.Timestamp]] = None,
        run_forward: bool = config_ptm["run_forward"]["default"],
        time_step: int = config_ptm["time_step"]["default"],
        time_step_output: Optional[int] = config_ptm["time_step_output"]["default"],
        steps: Optional[int] = config_ptm["steps"]["default"],
        duration: Optional[datetime.timedelta] = config_ptm["duration"]["default"],
        end_time: Optional[datetime.datetime] = config_ptm["end_time"]["default"],
        # universal inputs
        ocean_model: Optional[str] = config_ptm["ocean_model"]["default"],
        ocean_model_local: Optional[bool] = config_ptm["ocean_model_local"]["default"],
        surface_only: Optional[bool] = config_ptm["surface_only"]["default"],
        do3D: bool = config_ptm["do3D"]["default"],
        vertical_mixing: bool = config_ptm["vertical_mixing"]["default"],
        use_static_masks: bool = config_ptm["use_static_masks"]["default"],
        output_file: Optional[str] = config_ptm["output_file"]["default"],
        **kw,
    ) -> None:
        """Inputs necessary for any particle tracking."""

        # get all named parameters input to ParticleTrackingManager class
        from inspect import signature

        sig = signature(ParticleTrackingManager)

        self.__dict__["config_ptm"] = config_ptm
        # self.__dict__["config_model"] = None  # previously defined before manager is initialized
        self.__dict__["_config_orig"] = None

        # check this here for initialization since later they will be set
        if steps is not None:
            assert duration is None and end_time is None
        if duration is not None:
            assert steps is None and end_time is None
        if end_time is not None:
            assert steps is None and duration is None

        # initialize all class attributes to None without triggering the __setattr__ method
        # which does a bunch more stuff
        for key in sig.parameters.keys():
            self.__dict__[key] = None

        # mode flags
        self.__dict__["has_added_reader"] = False
        self.__dict__["has_run_seeding"] = False
        self.__dict__["has_run"] = False

        # Set all attributes which will trigger some checks and changes in __setattr__
        # these will also update "value" in the config dict
        for key in sig.parameters.keys():
            # no need to run through for init if value is None (already set to None)
            if locals()[key] is not None:
                self.__setattr__(key, locals()[key])

        self.kw = kw

        if self.__dict__["output_file"] is None:
            self.__dict__[
                "output_file"
            ] = f"output-results_{datetime.datetime.now():%Y-%m-%dT%H%M:%SZ}.nc"

        ## set up log for this simulation
        # Create a file handler
        assert self.__dict__["output_file"] is not None
        logfile_name = self.__dict__["output_file"].replace(".nc", ".log")
        self.file_handler = logging.FileHandler(logfile_name)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(self.file_handler)

        self.logger.info(f"filename: {logfile_name}")
        ##

    def __setattr_model__(self, name: str, value) -> None:
        """Implement this in model class to add specific __setattr__ there too."""
        pass

    # calculate other simulation-length parameters when one is input
    # this way whichever parameter is input last overwrites the other parameters
    # that could have been input earlier
    # also have to check for the special case that start_time is being updated to be the
    # initial model output when the reader is set and in that case also need to update
    # end_time based on whichever of steps or duration is available.
    def calc_end_time(self, changed_variable):
        """Calculate end time based on other simulation length parameters."""

        if changed_variable == "steps" or (
            self.steps is not None and changed_variable == "start_time"
        ):
            return self.start_time + self.timedir * self.steps * datetime.timedelta(
                seconds=self.time_step
            )
        elif changed_variable == "duration" or (
            self.duration is not None and changed_variable == "start_time"
        ):
            return self.start_time + self.timedir * self.duration
        else:
            return self.end_time
        # if self.start_time is not None and self.steps is not None:
        #     return self.start_time + self.timedir * self.steps * datetime.timedelta(
        #         seconds=self.time_step
        #     )
        # elif self.start_time is not None and self.duration is not None:
        #     return self.start_time + self.timedir * self.duration
        # else:
        #     return self.end_time

    def calc_duration(self):
        """Calculate duration based on end_time and start_time."""
        if self.end_time is not None and self.start_time is not None:
            return abs(self.end_time - self.start_time)
        else:
            return self.duration

    def calc_steps(self):
        """Calculate steps based on duration and time_step."""
        if self.duration is not None and self.start_time is not None:
            return self.duration / datetime.timedelta(seconds=self.time_step)
        else:
            return self.steps

    def __setattr__(self, name: str, value) -> None:
        """Implement my own __setattr__ to enforce subsequent actions."""

        # create/update class attribute
        self.__dict__[name] = value

        # create/update "value" keyword in config to keep it up to date
        if name in self.config_ptm.keys():
            self.config_ptm[name]["value"] = value

        # create/update "value" keyword in model config to keep it up to date
        if self.config_model is not None:  # can't run this until init in model class
            self.__setattr_model__(name, value)

        # None of the following checks occur if value is None
        if value is not None:

            # check longitude when it is set
            if name == "lon":
                assert (
                    -180 <= value <= 180
                ), "Longitude needs to be between -180 and 180 degrees."

            if name == "lat":
                assert (
                    -90 <= value <= 90
                ), "Latitude needs to be between -90 and 90 degrees."

            if name == "start_time":
                if isinstance(value, (str, datetime.datetime, pd.Timestamp)):
                    self.__dict__[name] = pd.Timestamp(value)
                    self.config_ptm[name]["value"] = pd.Timestamp(value)
                else:
                    raise TypeError(
                        "start_time must be a string, datetime, or Timestamp."
                    )

            # # make sure ocean_model name uppercase
            # if name == "ocean_model":
            #     self.__dict__[name] = value.upper()
            #     self.config_ptm["ocean_model"]["value"] = value.upper()

            # check start_time relative to ocean_model selection
            if name in ["ocean_model", "start_time"]:
                if self.start_time is not None and self.ocean_model is not None:
                    assert isinstance(self.start_time, pd.Timestamp)
                    if self.ocean_model == "NWGOA":
                        assert overall_start_time <= self.start_time <= nwgoa_end_time
                    elif self.ocean_model == "CIOFS":
                        assert overall_start_time <= self.start_time <= ciofs_end_time
                    elif self.ocean_model == "CIOFSOP":
                        assert (
                            ciofs_operational_start_time
                            <= self.start_time
                            <= ciofs_operational_end_time
                        )

            # deal with if input longitudes need to be shifted due to model
            if name == "oceanmodel_lon0_360" and value:
                if self.ocean_model != "test" and self.lon is not None:
                    # move longitude to be 0 to 360 for this model
                    # this is not a user-defined option
                    if -180 < self.lon < 0:
                        self.__dict__["lon"] += 360
                        self.config_ptm["lon"]["value"] += 360  # this isn't really used

            if name == "surface_only" and value:
                self.logger.info(
                    "Overriding values for do3D, z, and vertical_mixing because surface_only is True (to False, 0, False)."
                )
                self.do3D = False
                self.z = 0
                self.vertical_mixing = False

            # z should be a negative value, representing below the surface
            if name == "z" and value > 0:
                raise ValueError(
                    "z should be a negative value, representing below the surface."
                )

            # in case any of these are reset by user after surface_only is already set
            # if surface_only is True, do3D must be False
            if name == "do3D" and value and self.surface_only:
                self.logger.info(
                    "do3D must be False because surface_only is True. Setting do3D to False."
                )
                self.__dict__["do3D"] = False
                self.config_ptm["do3D"]["value"] = False

            # if surface_only is True, z must be 0
            if name == "z" and value != 0 and self.surface_only:
                self.logger.info(
                    "z must be 0 because surface_only is True. Setting z to 0."
                )
                self.__dict__["z"] = 0
                self.config_ptm["z"]["value"] = 0

            # if surface_only is True, vertical_mixing must be False
            if name == "vertical_mixing" and value and self.surface_only:
                self.logger.info(
                    "vertical_mixing must be False because surface_only is True. Setting vertical_mixing to False."
                )
                self.__dict__["vertical_mixing"] = False
                self.config_ptm["vertical_mixing"]["value"] = False

            # if not 3D turn off vertical_mixing
            if name in ["do3D", "vertical_mixing"]:
                if not self.do3D and self.vertical_mixing:
                    self.logger.info("Turning off vertical_mixing since do3D is False")
                    self.__dict__["vertical_mixing"] = False
                    self.config_ptm["vertical_mixing"]["value"] = False

            # set z to None if seed_seafloor is True
            if name == "seed_seafloor" and value:
                self.logger.info("setting z to None since being seeded at seafloor")
                self.z = None

            # in case z is changed back after initialization
            if (
                name == "z" and self.seed_seafloor
            ):  # already checked that value is not None
                self.logger.info(
                    "setting `seed_seafloor` from True to False since now setting a non-None z value"
                )
                self.seed_seafloor = False

            # if reader, lon, and lat set, check inputs
            if (
                name == "has_added_reader"
                and value
                and self.lon is not None
                and self.lat is not None
                or name in ["lon", "lat"]
                and self.has_added_reader
                and self.lon is not None
                and self.lat is not None
            ):

                if self.ocean_model != "test":
                    rlon = self.reader_metadata("lon")
                    assert rlon.min() < self.lon < rlon.max()
                    rlat = self.reader_metadata("lat")
                    assert rlat.min() < self.lat < rlat.max()

            # if reader, lon, and lat set, check inputs
            if name == "has_added_reader" and value and self.start_time is not None:

                if self.ocean_model != "test":
                    assert self.reader_metadata("start_time") <= self.start_time

            # if reader, lon, and lat set, check inputs
            if name == "has_added_reader" and value:
                assert self.ocean_model is not None

            # define time direction
            if name == "run_forward":
                if value:
                    self.__dict__["timedir"] = 1
                else:
                    self.__dict__["timedir"] = -1

            if (
                name in ["start_time", "end_time", "steps", "duration"]
                and self.start_time is not None
            ):
                # the behavior in calc_end_time changes depending on which variable has been updated
                self.__dict__["end_time"] = self.calc_end_time(name)
                self.config_ptm["end_time"]["value"] = self.calc_end_time(name)
                # duration and steps are always updated now that start_time and end_time are set
                self.__dict__["duration"] = self.calc_duration()
                self.config_ptm["duration"]["value"] = self.calc_duration()
                self.__dict__["steps"] = self.calc_steps()
                self.config_ptm["steps"]["value"] = self.calc_steps()

            if name == "ocean_model" and value not in _KNOWN_MODELS:
                self.logger.info(f"ocean_model is not one of {_KNOWN_MODELS}.")

    def add_reader(self, **kwargs):
        """Here is where the model output is opened."""

        self.run_add_reader(**kwargs)

        self.has_added_reader = True

    def seed(self, lon=None, lat=None, z=None):
        """Initialize the drifters in space and time

        ... and with any special properties.
        """

        for key in [lon, lat, z]:
            if key is not None:
                self.__setattr__(self, f"{key}", key)

        # if self.ocean_model != "test" and not self.has_added_reader:
        #     raise ValueError("first add reader with `manager.add_reader(**kwargs)`.")

        # have this check here so that all parameters aren't required until seeding
        if self.seed_flag == "elements" and self.lon is None and self.lat is None:
            msg = f"""lon and lat need non-None values if using `seed_flag="elements"`.
                    Update them with e.g. `self.lon=-151` or input to `seed`."""
            raise KeyError(msg)
        elif self.seed_flag == "geojson" and self.geojson is None:
            msg = f"""geojson need non-None value if using `seed_flag="geojson"`."""
            raise KeyError(msg)

        msg = f"""z needs a non-None value.
                  Please update it with e.g. `self.z=-10` or input to `seed`."""
        if not self.seed_seafloor:
            assert self.z is not None, msg

        if self.ocean_model is not None and not self.has_added_reader:
            self.add_reader()

        if self.start_time is None:
            raise KeyError(
                "first add reader with `manager.add_reader(**kwargs)` or input a start_time."
            )

        self.run_seed()
        self.has_run_seeding = True

    def run(self):
        """Call model run function."""

        if not self.has_run_seeding:
            raise KeyError("first run seeding with `manager.seed()`.")

        self.logger.info(
            f"start_time: {self.start_time}, end_time: {self.end_time}, steps: {self.steps}, duration: {self.duration}"
        )

        # need end time info
        assert (
            self.steps is not None
            or self.duration is not None
            or self.end_time is not None
        )

        self.run_drifters()

        # Remove the handler at the end of the loop
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

        self.has_run = True

    def run_all(self):
        """Run all steps."""

        if not self.has_added_reader:
            self.add_reader()

        if not self.has_run_seeding:
            self.seed()

        if not self.has_run:
            self.run()

    def output(self):
        """Hold for future output function."""
        pass

    def _config(self):
        """Model should have its own version which returns variable config"""
        pass

    def _add_ptm_config(self):
        """Have this in the model class to modify config"""
        pass

    def _add_model_config(self):
        """Have this in the model class to modify config"""
        pass

    def _update_config(self) -> None:
        """Update configuration between model, PTM additions, and model additions."""

        # Modify config with PTM config
        self._add_ptm_config()

        # Modify config with model-specific config
        self._add_model_config()

    def show_config_model(self):
        """define in child class"""
        pass

    def show_config(self, **kwargs) -> dict:
        """Show parameter configuration across both model and PTM."""

        self._update_config()

        # Filter config
        config = self.show_config_model(**kwargs)

        return config

    def reader_metadata(self, key):
        """define in child class"""
        pass

    def query_reader(self):
        """define in child class."""
        pass

    def all_export_variables(self):
        """Output list of all possible export variables.

        define in child class.
        """
        pass

    def export_variables(self):
        """Output list of all actual export variables.

        define in child class.
        """
        pass

    @property
    def outfile_name(self):
        """Output file name.

        define in child class.
        """
        pass
