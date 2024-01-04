"""Contains logic for configuring OpenDrift simulations."""


from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
import xarray as xr

import pandas as pd
import cmocean.cm as cmo
from docstring_inheritance import NumpyDocstringInheritanceMeta
import datetime


class ParticleTrackingManager(metaclass=NumpyDocstringInheritanceMeta):
    """Manager class that controls particle tracking model."""
    
    def __init__(self, 
                 model,
                 lon: Optional[Union[int,float]] = None,
                 lat: Optional[Union[int,float]] = None,
                 z: Union[int,float,str] = 0,
                 start_time: Optional[datetime.datetime] = None,
                 run_forward: bool = True,
                 time_step: int = 3600,  # s
                 time_step_output: Optional[int] = None,
                 
                 steps: Optional[int] = 3,
                 duration: Optional[datetime.timedelta] = None,
                 end_time: Optional[datetime.datetime] = None,
                              
                 # universal inputs
                 log: str = "low",
                 oceanmodel: Optional[str] = None,
                 surface_only: Optional[bool] = None,
                 do3D: bool = False,
                 vertical_mixing: bool = False,
                 coastline_action: str = "previous",
                 stokes_drift: bool = True,
                 **kw) -> None:
        """Inputs necessary for any particle tracking.

        Parameters
        ----------
        model : str
            Name of Lagrangian model to use for drifter tracking
        lon : Optional[Union[int,float]], optional
            Longitude of center of initial drifter locations, by default None
        lat : Optional[Union[int,float]], optional
            Latitude of center of initial drifter locations, by default None
        z : Union[int,float,str], optional
            Depth of initial drifter locations, by default 0. Values are overridden if 
            ``surface_only==True``.
        start_time : Optional[datetime], optional
            Start time of simulation, as a datetime object, by default None
        run_forward : bool, optional
            True to run forward in time, False to run backward, by default True
        time_step : int, optional
            Time step in seconds, options >0, <86400 (1 day in seconds), by default 3600
        time_step_output : int, optional
            How often to output model output, in seconds. Should be a multiple of time_step. 
            By default will take the value of time_step.
        steps : int, optional
            Number of time steps to run in simulation. Options >0. 
            steps, end_time, or duration must be input by user. By default steps is 3 and 
            duration and end_time are None.
        duration : Optional[datetime.timedelta], optional
            Length of simulation to run, as positive-valued timedelta object.
            steps, end_time, or duration must be input by user. By default steps is 3 and 
            duration and end_time are None.
        end_time : Optional[datetime], optional
            Datetime at which to end simulation, as positive-valued datetime object.
            steps, end_time, or duration must be input by user. By default steps is 3 and 
            duration and end_time are None.
        log : str, optional
            Options are "low" and "high" verbosity for log, by default "low"
        oceanmodel : Optional[str], optional
            Name of ocean model to use for driving drifter simulation, by default None.
            Use None for testing and set up. Otherwise input a string. 
            Options are: "NWGOA", "CIOFS", "CIOFS (operational)".
            Alternatively keep as None and set up a separate reader (see example in docs).
        surface_only : bool, optional
            Set to True to keep drifters at the surface, by default None. 
            If this flag is set to not-None, it overrides do3D to False, vertical_mixing to False, 
            and the z value(s) 0. 
            If True, this flag also turns off reading model output below 0.5m if 
            driftmodel is not Leeway:
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
        coastline_action : str, optional
            Action to perform if a drifter hits the coastline, by default "previous". Options
            are 'stranding', 'previous'.
        stokes_drift : bool, optional
            Set to True to turn on Stokes drift, by default True.
        """
        
        self.lon = lon
        self.lat = lat

        self.surface_only = surface_only
        self.do3D = do3D
        self.z = z
        self.vertical_mixing = vertical_mixing

        # checks are in __setattr__

        self.run_forward = run_forward
        self.time_step = time_step
        self.time_step_output = time_step_output
        
        self.start_time = start_time
        self.steps = steps
        self.duration = duration
        self.end_time = end_time
            

        self.model = model
        
        self.log = log
        self.oceanmodel = oceanmodel
        self.vertical_mixing = vertical_mixing
        self.coastline_action = coastline_action
        self.stokes_drift = stokes_drift

        # mode flags
        self.has_run_config = False
        self.has_added_reader = False
        self.has_run_seeding = False
        self.has_run = False

        self.kw = kw
        
        
    def __setattr__(self, name: str, value) -> None:
        self.__dict__[name] = value
        
        # check longitude when it is set
        if value is not None and name == "lon":
            assert -180 <= value <= 180, "Longitude needs to be between -180 and 180 degrees."

        # deal with if input longitudes need to be shifted due to model
        if name == "oceanmodel_lon0_360" and value:
            if self.oceanmodel is not None and self.lon is not None:
                # move longitude to be 0 to 360 for this model
                # this is not a user-defined option
                if -180 < self.lon < 0:
                    self.lon += 360

        # set output time step to match time_step if None
        if name == "time_step_output":
            self.__dict__[name] = value or self.time_step

        # If time_step updated, also update time_step_output
        if name == "time_step" and hasattr(self, "time_step_output"):
            print("updating time_step_output to match time_step.")
            self.__dict__["time_step_output"] = value

        if name == "surface_only" and value: 
            print("overriding values for `do3D`, `z`, and `vertical_mixing` because `surface_only` True")
            self.__dict__["do3D"] = False
            self.__dict__["z"] = 0
            self.__dict__["vertical_mixing"] = False

        # in case any of these are reset by user after surface_only is already set
        if name in ["do3D", "z", "vertical_mixing"]:
            if self.surface_only:
                print("overriding values for `do3D`, `z`, and `vertical_mixing` because `surface_only` True")
                self.__dict__["do3D"] = False
                self.__dict__["z"] = 0
                self.__dict__["vertical_mixing"] = False
                

        # # forward/backward
        # if name in ["time_step", "run_forward"]:
        #     if self.run_forward:
        #         self.__dict__["time_step"] = abs(self.time_step)
        #     else:
        #         self.__dict__["time_step"] = -1*abs(self.time_step)

        
    # these should all be implemented in drifter_class object
    def config(self):
        """Configuration for a simulation."""
        
        self.run_config()
        self.has_run_config = True
    
    def add_reader(self, **kwargs):
        """Here is where the model output is opened."""
        
        if not self.has_run_config:
            raise KeyError("first run configuration with `manager.config()`.")

        self.run_add_reader(**kwargs)

        self.has_added_reader = True

    def seed(self):
        """Initialize the drifters in space and time
        
        ... and with any special properties.
        """
        
        if not self.has_run_config:
            raise KeyError("first run configuration with `manager.config()`.")

        msg = f"""lon, lat, and z all need non-None values. 
                  Please update them with e.g. `self.lon=-151`."""
        assert self.lon is not None and self.lat is not None and self.z is not None, msg

        # use reader start time if not otherwise input
        if self.start_time is None and self.has_added_reader:
            self.start_time = self.o.env.readers['roms native'].start_time
        elif self.start_time is None and not self.has_added_reader:
            msg = f"""`start_time` is required or first setup reader and then will use 
                      that start_time by default."""
            raise KeyError(msg)

        # MOVE THIS TO CHECKS?
        # if reader already set, check inputs
        if self.oceanmodel is not None and self.has_added_reader:

            rlon = self.o.env.readers['roms native'].lon
            assert rlon.min() < self.lon < rlon.max()
            rlat = self.o.env.readers['roms native'].lat
            assert rlat.min() < self.lat < rlat.max()
            assert self.o.env.readers['roms native'].start_time <= self.start_time
        else:
            print("did not check seeding inputs for location or time")
        
        self.run_seed()
        self.has_run_seeding = True
    
    def run(self):
        
        if not self.has_run_seeding:
            raise KeyError("first run configuration with `manager.seed()`.")

        if not self.has_added_reader:
            raise KeyError("first run configuration with `manager.add_reader()`.")

        # need end time info
        assert self.steps is not None or self.duration is not None or self.end_time is not None
        
        if self.run_forward:
            timedir = 1
        else:
            timedir = -1
        
        if self.steps is not None:
            self.end_time = self.start_time + timedir*self.steps*datetime.timedelta(seconds=self.time_step)
            self.duration = abs(self.end_time - self.start_time)
        elif self.duration is not None:
            self.end_time = self.start_time + timedir*self.duration
            self.steps = self.duration/datetime.timedelta(seconds=self.time_step)
        elif self.end_time is not None:
            self.duration = abs(self.end_time - self.start_time)
            self.steps = self.duration/datetime.timedelta(seconds=self.time_step)
 
        self.run_drifters()
        self.has_run = True
    
    def run_all(self):
        """Run all steps."""
        
        if not self.has_run_config:
            self.config()
        
        if not self.has_added_reader:
            self.add_reader()

        if not self.has_run_seeding:
            self.seed()

        if not self.has_run:
            self.run()

    
    def output(self):
        pass