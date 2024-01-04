"""Using OpenDrift for particle tracking."""
import datetime
from typing import Optional, Union
import pandas as pd
import opendrift
from opendrift.readers import reader_ROMS_native, reader_netCDF_CF_generic, reader_global_landmask
from opendrift.models.oceandrift import OceanDrift

from opendrift.models.larvalfish import LarvalFish
from opendrift.models.leeway import Leeway
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.oceandrift import Lagrangian3DArray
from opendrift.models.openoil import OpenOil

from .utils import copydocstring

from .the_manager import ParticleTrackingManager


ciofs_operational_start_time = datetime.datetime(2021,8,31,19,0,0)
ciofs_operational_end_time = (pd.Timestamp.now() + pd.Timedelta("48H")).to_pydatetime()
ciofs_end_time = datetime.datetime(2023,1,1,0,0,0)
nwgoa_end_time = datetime.datetime(2009,1,1,0,0,0)
overall_start_time = datetime.datetime(1999,1,1,0,0,0)
overall_end_time = ciofs_operational_end_time


# @copydocstring( ParticleTrackingManager )
class OpenDrift(ParticleTrackingManager):
    """Open drift particle tracking model."""
    
    def __init__(self, model: str="opendrift", driftmodel: str=None, 
                 lon: Optional[Union[int,float]] = None,
                 lat: Optional[Union[int,float]] = None,
                 z: Optional[Union[int,float,str]] = None,
                 start_time: Optional[datetime.datetime] = None,
                 run_forward: Optional[bool] = None,
                 time_step: Optional[int] = None,  # s
                 time_step_output: Optional[int] = None,
                 
                 steps: Optional[int] = None,
                 duration: Optional[datetime.timedelta] = None,
                 end_time: Optional[datetime.datetime] = None,
                              
                 # universal inputs
                 log: Optional[str] = None,
                 oceanmodel: Optional[str] = None,
                 surface_only: Optional[bool] = None,
                 do3D: Optional[bool] = None,
                 vertical_mixing: Optional[bool] = None,
                 coastline_action: Optional[str] = None,
                 stokes_drift: Optional[bool] = None,
                 **kw) -> None:
        """Inputs for OpenDrift model.

        Parameters
        ----------
        model : str
            Name of Lagrangian model to use for drifter tracking, in this case: "opendrift"
        driftmodel : str, optional
            Options: "OceanDrift", "LarvalFish", "OpenOil", "Leeway", by default "OceanDrift"
        use_auto_landmask : bool
            Set as True to use general landmask instead of that from oceanmodel. 
            Use for testing primarily. Default is False.
        number : int
            Number of drifters to simulate. Default is 100.
        INCLUDE OTHER OPTIONAL PARAMETERS?
        """
        driftmodel = driftmodel or "OceanDrift"
        self.driftmodel = driftmodel
        # self.kwargs = kwargs
        # import pdb; pdb.set_trace()

        # # # Calling general constructor of parent class
        # super(OpenDrift, self).__init__(model, lon, lat, z, 
        #                                 start_time, run_forward, 
        #                                 time_step, time_step_output,
        #                                 steps, duration, end_time,
        #                                 log, oceanmodel, surface_only,
        #                                 do3D, vertical_mixing,
        #                                 coastline_action, stokes_drift,
        #                                 **kwargs)
        # super().__init__()
        super().__init__(model, lon, lat, z, 
                            start_time, run_forward, 
                            time_step, time_step_output,
                            steps, duration, end_time,
                            log, oceanmodel, surface_only,
                            do3D, vertical_mixing,
                            coastline_action, stokes_drift,
                         **kw)
        
        if self.log == "low":
            self.loglevel = 20
        elif self.log == "high":
            self.loglevel = 0

    def run_config(self):

        # do this right away so I can query the object
        if self.driftmodel == "Leeway":
            o = Leeway(loglevel=self.loglevel)

        elif self.driftmodel == "OceanDrift":
            o = OceanDrift(loglevel=self.loglevel)

        elif self.driftmodel == "LarvalFish":
            o = LarvalFish(loglevel=self.loglevel)

        elif self.driftmodel == "OpenOil":
            o = OpenOil(loglevel=self.loglevel, weathering_model='noaa')

        else:
            raise ValueError(f"Drifter model {self.driftmodel} is not recognized.")

        # Note that you can see configuration possibilities for a given model with
        # o.list_configspec()
        # You can check the metadata for a given configuration with (min/max/default/type)
        # o.get_configspec('vertical_mixing:timestep')
        # You can check required variables for a model with
        # o.required_variables
        
        # defaults that might be overridden by incoming kwargs
        kw = {
            "emulsification": True,
            "dispersion": True,
            "evaporation": True,
            "wave_entrainment": True,
            "update_oilfilm_thickness": True,
            "biodegradation": True,
            "diffusivitymodel": 'windspeed_Large1994',
            "mixed_layer_depth": 30,
            "max_speed": 5,
            "oil_type": 'GENERIC MEDIUM CRUDE',
            "use_auto_landmask": False,
            }
        # use some built in defaults when I don't care to override for my own default
        key = 'seed:object_type'
        kw.update({key.split(":")[1]: Leeway(loglevel=50).get_configspec(key)[key]["default"],})

        oo = OpenOil(loglevel=50)
        key = 'seed:oil_type'
        kw.update({key.split(":")[1]: oo.get_configspec(key)[key]["default"],})
        key = 'seed:m3_per_hour'
        kw.update({key.split(":")[1]: oo.get_configspec(key)[key]["default"],})
        
        lf = LarvalFish(loglevel=50)
        key = 'seed:diameter'
        kw.update({key.split(":")[1]: lf.get_configspec(key)[key]["default"],})
        key = 'seed:neutral_buoyancy_salinity'
        kw.update({key.split(":")[1]: lf.get_configspec(key)[key]["default"],})
        key = 'seed:stage_fraction'
        kw.update({key.split(":")[1]: lf.get_configspec(key)[key]["default"],})
        key = 'seed:hatched'
        kw.update({key.split(":")[1]: lf.get_configspec(key)[key]["default"],})
        key = 'seed:length'
        kw.update({key.split(":")[1]: lf.get_configspec(key)[key]["default"],})
        key = 'seed:weight'
        kw.update({key.split(":")[1]: lf.get_configspec(key)[key]["default"],})
                                
        od = OceanDrift(loglevel=50)
        key = 'wave_entrainment:droplet_size_distribution'
        kw.update({"wave_entrainment_droplet_size_distribution": oo.get_configspec(key)[key]["default"],})
        key = 'drift:wind_drift_depth'
        kw.update({key.split(":")[1]: od.get_configspec(key)[key]["default"],})
        key = 'vertical_mixing:timestep'
        kw.update({key.split(":")[1]: od.get_configspec(key)[key]["default"],})
        key = 'drift:advection_scheme'
        kw.update({key.split(":")[1]: od.get_configspec(key)[key]["default"],})
        key = 'drift:horizontal_diffusivity'
        kw.update({key.split(":")[1]: od.get_configspec(key)[key]["default"],})

        # add defaults for seeding
        kw.update({"radius": 1000,
                   "number": 100,
        })

        # use defaults when I don't want to override
        key = 'wind_drift_factor'
        kw.update({key: Lagrangian3DArray().variables[key]["default"],})
                
        kw.update(self.kw)


        if self.driftmodel == "Leeway":
            o.set_config('seed:object_type', kw["object_type"])

        elif self.driftmodel == "OceanDrift":
            pass

        elif self.driftmodel == "LarvalFish":
            pass

        elif self.driftmodel == "OpenOil":
            o.set_config('processes:emulsification',  kw["emulsification"])
            o.set_config('processes:dispersion', kw["dispersion"])
            o.set_config('processes:evaporation', kw["evaporation"])
            if kw["wave_entrainment"]:
                o.set_config('wave_entrainment:entrainment_rate', 'Li et al. (2017)')  # only option
            o.set_config('wave_entrainment:droplet_size_distribution', kw["wave_entrainment_droplet_size_distribution"])
            o.set_config('processes:update_oilfilm_thickness', kw["update_oilfilm_thickness"])
            o.set_config('processes:biodegradation', kw["biodegradation"])

        else:
            raise ValueError(f"Drifter model {self.driftmodel} is not recognized.")

        if self.driftmodel != "Leeway" and self.driftmodel != "LarvalFish":
            o.set_config('drift:wind_drift_depth', kw["wind_drift_depth"])


        # Leeway model doesn't have this option built in
        if self.surface_only and self.driftmodel != "Leeway":
            o.set_config('drift:truncate_ocean_model_below_m', 0.5)

        # 2D
        # Leeway doesn't have this option available
        if not self.do3D and self.driftmodel != "Leeway":
            o.disable_vertical_motion()
        elif self.do3D:
            o.set_config('drift:vertical_advection', True)
            o.set_config('drift:vertical_mixing', self.vertical_mixing)
            if self.vertical_mixing:
                o.set_config('vertical_mixing:diffusivitymodel', kw["diffusivitymodel"])
                o.set_config('vertical_mixing:background_diffusivity', 1.2e-5)  # default 1.2e-5
                o.set_config('environment:fallback:ocean_mixed_layer_thickness', kw["mixed_layer_depth"])
                o.set_config('vertical_mixing:timestep', kw["vertical_mixing_timestep"]) # seconds


        if self.z == "seabed":
            o.set_config('seed:seafloor', True)


        o.set_config('drift:advection_scheme', kw["advection_scheme"])
        o.set_config('drift:max_speed', kw["max_speed"])
        o.set_config('general:use_auto_landmask', kw["use_auto_landmask"])  # use ocean model landmask instead of generic coastline data
        o.set_config('drift:horizontal_diffusivity', kw["horizontal_diffusivity"])  # m2/s
        o.set_config('general:coastline_action', self.coastline_action)

        if self.driftmodel != "Leeway":
            o.set_config('drift:stokes_drift', self.stokes_drift)
            if self.stokes_drift:
                o.set_config('drift:use_tabularised_stokes_drift', True)
                o.set_config('drift:tabularised_stokes_drift_fetch', '25000')  # default
                o.set_config('drift:stokes_drift_profile', 'Phillips')  # default
        
        self.o = o
        
        # update kw
        self.kw.update(kw)


    def run_add_reader(self, loc=None, kwargs_xarray=None, oceanmodel_lon0_360=False, ):
        """Might need to cache this if its still slow locally.
        
        Parameters
        ----------
        oceanmodel_lon0_360 : bool
            True if ocean model longitudes span 0 to 360 instead of -180 to 180.
        """
        
        oceanmodel = self.oceanmodel
        kwargs_xarray = kwargs_xarray or {}
        
        if loc is not None:
            self.oceanmodel = "user_input"

        if oceanmodel is not None or loc is not None:
            if oceanmodel == "NWGOA":
                oceanmodel_lon0_360 = True
                loc = "http://xpublish-nwgoa.srv.axds.co/datasets/nwgoa_all/zarr/"
                kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
            elif oceanmodel == "CIOFS":
                oceanmodel_lon0_360 = False
                loc = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"
                kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
                reader = reader_ROMS_native.Reader(loc, kwargs_xarray=kwargs_xarray)
            elif oceanmodel == "CIOFS (operational)":
                pass
                # loc = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"
                # kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
                # reader = reader_ROMS_native.Reader(loc, kwargs_xarray=kwargs_xarray) 
            
            reader = reader_ROMS_native.Reader(loc, kwargs_xarray=kwargs_xarray)
            self.o.add_reader([reader])
            self.reader = reader
            # can find reader at manager.o.env.readers['roms native']            
            
            self.oceanmodel_lon0_360 = oceanmodel_lon0_360


    def run_seed(self):
        
        if self.start_time is None and hasattr(self.o, "reader"):
            self.start_time = self.o.reader.start_time

        seed_kws = dict(lon=self.lon,
                        lat=self.lat,
                        radius=self.kw["radius"],
                        number=self.kw["number"],
                        time=self.start_time,
                    )
        
        if "radius_type" in self.kw:
            seed_kws.update({"radius_type": self.kw["radius_type"]})
            self.kw.pop("radius_type")

        if self.z != "seabed":
            seed_kws.update({"z": self.z})

        # Include model-specific inputs

        # can vary by drifter, but leave that for future possible work
        if self.driftmodel != "LarvalFish":
            seed_kws.update({"wind_drift_factor": self.kw["wind_drift_factor"]})

        if self.driftmodel == "LarvalFish":
            seed_kws.update(dict(diameter=self.kw["diameter"], 
                                neutral_buoyancy_salinity=self.kw["neutral_buoyancy_salinity"],
                                stage_fraction=self.kw["stage_fraction"],
                                hatched=self.kw["hatched"],
                                length=self.kw["length"],
                                weight=self.kw["weight"],))

        elif self.driftmodel == "OpenOil":
            seed_kws.update(dict(
                oil_type=self.kw["oil_type"],
                m3_per_hour=self.kw["m3_per_hour"],
                                ))

        self.o.seed_kws = seed_kws
        
        self.o.seed_elements(**seed_kws)
        
        self.initial_drifters = self.o.elements_scheduled


    def run_drifters(self):

        if self.run_forward:
            timedir = 1
        else:
            timedir = -1
        
        self.o.run(
            time_step=timedir*self.time_step,
            steps=self.steps,
            outfile=f'output-results_{datetime.utcnow():%Y-%m-%dT%H%M:%SZ}.nc'
        )
