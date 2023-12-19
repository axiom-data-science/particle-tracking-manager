"""Contains logic for configuring OpenDrift simulations."""


from pathlib import Path
import warnings

import numpy as np
import xarray as xr

import pandas as pd
import cmocean.cm as cmo


class ParticleTrackingManager(object):
    
    def __init__(self, 
                 model,
                 lon=None,
                 lat=None,
                 z=0,
                 start_time=None,
                 end_time=None,
                 runForward=True,
                 time_step=3600,  # s
                 time_step_output=None,
                 nsteps=4,    
                              
                 # universal inputs
                 log="low",
                 oceanmodel=None,
                 surface_only=True,
                 do3D=False,
                 vertical_mixing=False,
                 coastline_action="previous",
                 stokes_drift=True,
                 **kwargs) -> None:
        """
        
        opendrift-specific inputs can be input without being named and they will be passed 
        on through kw."""
        
        time_step_output = time_step_output or time_step
        
        self.lon = lon
        self.lat = lat
        self.z = z
        self.start_time = start_time
        self.end_time = end_time
        self.runForward = runForward
        self.time_step = time_step
        self.time_step_output = time_step_output
        self.nsteps = nsteps
        
        self.model = model
        
        
        self.log = log
        self.oceanmodel = oceanmodel
        self.surface_only = surface_only
        self.do3D = do3D
        self.vertical_mixing = vertical_mixing
        self.coastline_action = coastline_action
        self.stokes_drift = stokes_drift

        # mode flags
        self.has_run_config = False
        self.has_added_reader = False
        self.has_run_seeding = False
        self.has_run = False

        self.kw = kwargs
        
        
        
    # these should all be implemented in drifter_class object
    def config(self):
        """Configuration for a simulation."""
        
        self.run_config()
        self.has_run_config = True
    
    def add_reader(self):
        """Here is where the model output is opened."""
        
        if not self.has_run_config:
            raise KeyError("first run configuration with `manager.config()`.")

        self.run_add_reader()
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
        self.run_seed()
        self.has_run_seeding = True
    
    def run(self):
        
        if not self.has_run_seeding:
            raise KeyError("first run configuration with `manager.config()`.")

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