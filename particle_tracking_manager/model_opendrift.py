"""Using OpenDrift for particle tracking."""
from datetime import datetime
import opendrift
from opendrift.readers import reader_ROMS_native, reader_netCDF_CF_generic, reader_global_landmask
from opendrift.models.oceandrift import OceanDrift

from opendrift.models.larvalfish import LarvalFish
from opendrift.models.leeway import Leeway
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.oceandrift import Lagrangian3DArray
from opendrift.models.openoil import OpenOil

from the_manager import ParticleTrackingManager


class OpenDrift(ParticleTrackingManager):
    
    def __init__(self, model="opendrift", driftmodel=None, **kwargs) -> None:
        driftmodel = driftmodel or "OceanDrift"
        self.driftmodel = driftmodel
        # self.kwargs = kwargs


        # # Calling general constructor of parent class
        # import pdb; pdb.set_trace()
        super(OpenDrift, self).__init__(model, **kwargs)
        # super().__init__()
        
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
                   "ndrifters": 100,
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


    def run_add_reader(self):
        """Might need to cache this if its still slow locally."""
        
        oceanmodel = self.oceanmodel
        
        if oceanmodel is not None:
            if oceanmodel == "NWGOA":
                loc = "http://xpublish-nwgoa.srv.axds.co/datasets/nwgoa_all/zarr/"
                kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
                reader = reader_ROMS_native.Reader(loc, kwargs_xarray=kwargs_xarray)
            elif oceanmodel == "CIOFS":
                loc = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"
                kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
                reader = reader_ROMS_native.Reader(loc, kwargs_xarray=kwargs_xarray)
            elif oceanmodel == "CIOFS (operational)":
                pass
                # loc = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"
                # kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
                # reader = reader_ROMS_native.Reader(loc, kwargs_xarray=kwargs_xarray) 
            # elif oceanmodel == "fake":
            
            self.o.add_reader([reader])
            self.reader = reader


    def run_seed(self):
        
        if self.start_time is None and hasattr(self.o, "reader"):
            self.start_time = self.o.reader.start_time

        seed_kws = dict(lon=self.lon,
                        lat=self.lat,
                        radius=self.kw["radius"],
                        number=self.kw["ndrifters"],
                        time=self.start_time,
                    )

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

        self.o.seed_elements(**seed_kws)


    def run_drifters(self):
        
        self.o.run(
            time_step=self.time_step,
            steps=self.nsteps,
            outfile=f'output-results_{datetime.utcnow():%Y-%m-%dT%H%M:%SZ}.nc'
        )
