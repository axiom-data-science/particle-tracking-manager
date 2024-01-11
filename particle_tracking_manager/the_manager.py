"""Contains logic for configuring particle tracking simulations."""


from pathlib import Path
import pathlib
from typing import Optional, Union
import warnings

import numpy as np
import xarray as xr

import pandas as pd
import cmocean.cm as cmo
# from docstring_inheritance import NumpyDocstringInheritanceMeta
import datetime
import yaml

from .cli import is_None


# Read PTM configuration information
loc = pathlib.Path(__file__).parent / pathlib.Path("the_manager_config.yaml")
with open(loc, "r") as f:
    # Load the YAML file into a Python object
    config_ptm = yaml.safe_load(f)

# convert "None"s to Nones
for key in config_ptm.keys():
    if is_None(config_ptm[key]["default"]):
        config_ptm[key]["default"] = None


class ParticleTrackingManager():
    """Manager class that controls particle tracking model."""
    
    def __init__(self,
                 model: str,
                 lon: Optional[Union[int,float]] = None,
                 lat: Optional[Union[int,float]] = None,
                 z: Union[int,float] = config_ptm["z"]["default"],
                 seed_seafloor: bool = config_ptm["seed_seafloor"]["default"],
                 number: int = config_ptm["number"]["default"],
                 start_time: Optional[datetime.datetime] = None,
                 run_forward: bool = config_ptm["run_forward"]["default"],
                 time_step: int = config_ptm["time_step"]["default"],
                 time_step_output: Optional[int] = config_ptm["time_step_output"]["default"],
                 
                 steps: Optional[int] = config_ptm["steps"]["default"],
                 duration: Optional[datetime.timedelta] = config_ptm["duration"]["default"],
                 end_time: Optional[datetime.datetime] = config_ptm["end_time"]["default"],
                              
                 # universal inputs
                 log: str = config_ptm["log"]["default"],
                 ocean_model: Optional[str] = config_ptm["ocean_model"]["default"],
                 surface_only: Optional[bool] = config_ptm["surface_only"]["default"],
                 do3D: bool = config_ptm["do3D"]["default"],
                 vertical_mixing: bool = config_ptm["vertical_mixing"]["default"],
                 **kw) -> None:
        """Inputs necessary for any particle tracking.

        Parameters
        ----------
        model : str
            Name of Lagrangian model to use for drifter tracking. Only option
            currently is "opendrift".
        lon : Optional[Union[int,float]], optional
            Longitude of center of initial drifter locations, by default None
        lat : Optional[Union[int,float]], optional
            Latitude of center of initial drifter locations, by default None
        z : Union[int,float], optional
            Depth of initial drifter locations, by default 0 but taken from the 
            default in the model. Values are overridden if 
            ``surface_only==True`` to 0 and to the seabed if ``seed_seafloor`` is True.
        seed_seafloor : bool, optional
            Set to True to seed drifters vertically at the seabed, default is False. If True
            then value of z is set to None and ignored.
        number : int
            Number of drifters to simulate. Default is 100.
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
            Length of simulation to run, as positive-valued timedelta object, in hours,
            such as ``timedelta(hours=48)``.
            steps, end_time, or duration must be input by user. By default steps is 3 and 
            duration and end_time are None.
        end_time : Optional[datetime], optional
            Datetime at which to end simulation, as positive-valued datetime object.
            steps, end_time, or duration must be input by user. By default steps is 3 and 
            duration and end_time are None.
        log : str, optional
            Options are "low" and "high" verbosity for log, by default "low"
        ocean_model : Optional[str], optional
            Name of ocean model to use for driving drifter simulation, by default None.
            Use None for testing and set up. Otherwise input a string. 
            Options are: "NWGOA", "CIOFS", "CIOFS_now".
            Alternatively keep as None and set up a separate reader (see example in docs).
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
        """
        
        # get all named parameters input to ParticleTrackingManager class
        from inspect import signature
        sig = signature(ParticleTrackingManager)
        
        self.config_ptm = config_ptm

        # Set all attributes which will trigger some checks and changes in __setattr__
        # these will also update "value" in the config dict
        for key in sig.parameters.keys():
            self.__setattr__(key, locals()[key])
        
        # # Update config for all named input arguments
        # for key in sig.parameters.keys():
        #     if key != "kw":
        #         config_ptm[f"{key}"]["value"] = locals()[key]
        
        
        
        # # CHECK KWARGS HERE AND UPDATE CONFIG ACCORDINGLY!
        
        # # config_ptm["steps"]["value"] = steps
        
        # # update value in 
        # from inspect import signature
        # sig = signature(ParticleTrackingManager)
        # for key in sig.parameters.keys():
        #     config_ptm[f"{key}"]["value"] = key
        # import pdb; pdb.set_trace()
        
        # self.lon = lon
        # self.lat = lat

        # self.surface_only = surface_only
        # self.do3D = do3D
        # self.z = z
        # self.seed_seafloor = seed_seafloor
        # self.vertical_mixing = vertical_mixing

        # # checks are in __setattr__

        # self.run_forward = run_forward
        # self.time_step = time_step
        # self.time_step_output = time_step_output
        
        # self.start_time = start_time
        # self.steps = steps
        # self.duration = duration
        # self.end_time = end_time
            

        # self.model = model
        
        # self.log = log
        # self.ocean_model = ocean_model
        # self.vertical_mixing = vertical_mixing
        # self.coastline_action = coastline_action
        # self.stokes_drift = stokes_drift

        # mode flags
        self.has_run_config = False
        self.has_added_reader = False
        self.has_run_seeding = False
        self.has_run = False

        self.kw = kw

    def __setattr_model__(self, name: str, value) -> None:
        """Implement this in model class."""
        pass        
        
    def __setattr__(self, name: str, value) -> None:
        # print(name)
        # import pdb; pdb.set_trace()
        
        # create/update class attribute
        self.__dict__[name] = value
        
        # create/update "value" keyword in config to keep it up to date
        if name != "config_ptm" and name in self.config_ptm.keys():
            self.config_ptm[name]["value"] = value
        
        # create/update "value" keyword in model config to keep it up to date
        self.__setattr_model__(name, value)
        
        # check longitude when it is set
        if value is not None and name == "lon":
            assert -180 <= value <= 180, "Longitude needs to be between -180 and 180 degrees."
        
        if value is not None and name == "lat":
            assert -90 <= value <= 90, "Latitude needs to be between -90 and 90 degrees."

        # make sure ocean_model name uppercase
        if name == "ocean_model" and value is not None:
            self.__dict__[name] = value.upper()
            
        # deal with if input longitudes need to be shifted due to model
        if name == "oceanmodel_lon0_360" and value:
            if self.ocean_model is not "test" and self.lon is not None:
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
            if hasattr(self, "surface_only") and self.surface_only:
                print("overriding values for `do3D`, `z`, and `vertical_mixing` because `surface_only` True")
                self.__dict__["do3D"] = False
                self.__dict__["z"] = 0
                self.__dict__["vertical_mixing"] = False

            # if not 3D turn off vertical_mixing
            if hasattr(self, "do3D") and not self.do3D:
                print("turning off vertical_mixing since do3D is False")
                self.__dict__["vertical_mixing"] = False
            
        
        # set z to None if seed_seafloor is True
        if name == "seed_seafloor" and value:
            print("setting z to None since being seeded at seafloor")
            self.__dict__["z"] = None
        
        # in case z is changed back after initialization
        if name == "z" and value is not None and hasattr(self, "seed_seafloor"):
            print("setting `seed_seafloor` to False since now setting a non-None z value")
            self.__dict__["seed_seafloor"] = False

        # if reader, lon, and lat set, check inputs
        if name == "has_added_reader" and value and self.lon is not None and self.lat is not None \
            or name in ["lon","lat"] and hasattr(self, "has_added_reader") and self.has_added_reader and self.lon is not None and self.lat is not None:
                
            if self.ocean_model != "TEST":
                rlon = self.reader_metadata("lon")
                assert rlon.min() < self.lon < rlon.max()
                rlat = self.reader_metadata("lat")
                assert rlat.min() < self.lat < rlat.max()

        # use reader start time if not otherwise input
        if name == "has_added_reader" and value and self.start_time is None:
            print("setting reader start_time as simulation start_time")
            self.__dict__["start_time"] = self.reader_metadata("start_time")

        # if reader, lon, and lat set, check inputs
        if name == "has_added_reader" and value and self.start_time is not None:
                
            if self.ocean_model != "TEST":
                assert self.reader_metadata("start_time") <= self.start_time

        # if reader, lon, and lat set, check inputs
        if name == "has_added_reader" and value:
            assert self.ocean_model is not None


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

    def seed(self, lon=None, lat=None, z=None):
        """Initialize the drifters in space and time
        
        ... and with any special properties.
        """
        
        for key in [lon, lat, z]:
            if key is not None:
                self.__setattr__(self, f"{key}", key)
        
        # if self.ocean_model != "TEST" and not self.has_added_reader:
        #     raise ValueError("first add reader with `manager.add_reader(**kwargs)`.")

        msg = f"""lon and lat need non-None values. 
                  Update them with e.g. `self.lon=-151` or input to `seed`."""
        assert self.lon is not None and self.lat is not None

        msg = f"""z needs a non-None value. 
                  Please update it with e.g. `self.z=-10` or input to `seed`."""
        if not self.seed_seafloor:
            assert self.z is not None, msg

        self.run_seed()
        self.has_run_seeding = True
    
    def run(self):
        
        if not self.has_run_seeding:
            raise KeyError("first run seeding with `manager.seed()`.")

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
    
    def _config(self):
        """Model should have its own version which returns variable config"""
        pass
    
    def _add_ptm_config(self):
        """Have this in the model class to modify config"""
        pass
    
    def _add_model_config(self):
        """Have this in the model class to modify config"""
        pass
    
    # def _ptm_config(self):
    #     """Configuration metadata for PTM parameters."""

    #     # Modify model parameter config with PTM config
    #     # self._ptm_config_model()
    #     self._add_ptm_config()

    #     # combine model config and PTM config updates
    #     self._config.update(self.config_ptm)

    def show_config(self, **kwargs) -> dict:
        """Show parameter configuration across both model and PTM."""
        
        if not self.has_run_config:
            print("running config for you...")
            self.config()
        
        # # add in PTM-specific info to base model config
        # self._ptm_config()
        # Modify model parameter config with PTM config
        # self._ptm_config_model()
        self._add_ptm_config()
        
        # add in additional model config to base model config
        self._add_model_config()
        
        # Filter config
        config = self.show_config_model(**kwargs)

        return config
    
    def query_reader(self):
        """Overwrite method in model."""
        pass