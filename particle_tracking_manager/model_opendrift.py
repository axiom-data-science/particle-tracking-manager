"""Using OpenDrift for particle tracking."""
import copy
import datetime
import pathlib
from typing import Optional, Union
import pandas as pd
import yaml
import opendrift
from opendrift.readers import reader_ROMS_native, reader_netCDF_CF_generic, reader_global_landmask
from opendrift.models.oceandrift import OceanDrift

from opendrift.models.larvalfish import LarvalFish
from opendrift.models.leeway import Leeway
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.oceandrift import Lagrangian3DArray
from opendrift.models.openoil import OpenOil

# from .utils import copydocstring
from .cli import is_None
from .the_manager import ParticleTrackingManager


# Read PTM configuration information
loc = pathlib.Path(__file__).parent / pathlib.Path("opendrift_config.yaml")
with open(loc, "r") as f:
    # Load the YAML file into a Python object
    config_model = yaml.safe_load(f)

# convert "None"s to Nones
for key in config_model.keys():
    if is_None(config_model[key]["default"]):
        config_model[key]["default"] = None


ciofs_operational_start_time = datetime.datetime(2021,8,31,19,0,0)
ciofs_operational_end_time = (pd.Timestamp.now() + pd.Timedelta("48H")).to_pydatetime()
ciofs_end_time = datetime.datetime(2023,1,1,0,0,0)
nwgoa_end_time = datetime.datetime(2009,1,1,0,0,0)
overall_start_time = datetime.datetime(1999,1,1,0,0,0)
overall_end_time = ciofs_operational_end_time


# @copydocstring( ParticleTrackingManager )
class OpenDrift(ParticleTrackingManager):
    """Open drift particle tracking model."""
    
    def __init__(self,
                 driftmodel: str=None,
                 emulsification: bool = config_model["emulsification"]["default"],
                 **kw) -> None:
        """Inputs for OpenDrift model.

        Parameters
        ----------
        driftmodel : str, optional
            Options: "OceanDrift", "LarvalFish", "OpenOil", "Leeway", by default "OceanDrift"
        use_auto_landmask : bool
            Set as True to use general landmask instead of that from ocean_model. 
            Use for testing primarily. Default is False.
        number : int
            Number of drifters to simulate. Default is 100.
        INCLUDE OTHER OPTIONAL PARAMETERS?
        
        Notes
        -----
        Docs available for more initialization options with ``ptm.ParticleTrackingManager?``
        
        """
        
        model = "opendrift"

        super().__init__(model, **kw)
        
        # get all named parameters input to ParticleTrackingManager class
        from inspect import signature
        sig = signature(OpenDrift)
        # ADD SETATTR ADDITION HERE TO ADD TO MANAGER
        self.config_model = config_model

        # Set all attributes which will trigger some checks and changes in __setattr__
        # these will also update "value" in the config dict
        for key in sig.parameters.keys():
            self.__setattr__(key, locals()[key])
        
        # ## OpenOil ##
        # self.emulsification = emulsification
        
        driftmodel = driftmodel or "OceanDrift"
        self.driftmodel = driftmodel

        if self.log == "low":
            self.loglevel = 20
        elif self.log == "high":
            self.loglevel = 0

        
    def __setattr_model__(self, name: str, value) -> None:
        
        # create/update "value" keyword in config to keep it up to date
        if hasattr(self, "config_model") and name != "config_model" and name != "config_ptm" and name in self.config_model.keys():
            self.config_model[name]["value"] = value
        

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
            # "emulsification": True,
            # "dispersion": True,
            # "evaporation": True,
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
            # o.set_config('processes:emulsification',  kw["emulsification"])
            # o.set_config('processes:dispersion', kw["dispersion"])
            # o.set_config('processes:evaporation', kw["evaporation"])
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


        if self.seed_seafloor:
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
        
        # ocean_model = self.ocean_model
        kwargs_xarray = kwargs_xarray or {}
        
        if loc is not None and self.ocean_model is None:
            self.ocean_model = "user_input"

        if self.ocean_model.upper() == "TEST":
            oceanmodel_lon0_360 = True
            loc = "test"
            kwargs_xarray = dict()

        elif self.ocean_model is not None or loc is not None:
            if self.ocean_model == "NWGOA":
                oceanmodel_lon0_360 = True
                loc = "http://xpublish-nwgoa.srv.axds.co/datasets/nwgoa_all/zarr/"
                kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
            elif self.ocean_model == "CIOFS":
                oceanmodel_lon0_360 = False
                loc = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"
                kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
                reader = reader_ROMS_native.Reader(loc, kwargs_xarray=kwargs_xarray)
            elif self.ocean_model == "CIOFS_now":
                pass
                # loc = "http://xpublish-ciofs.srv.axds.co/datasets/ciofs_hindcast/zarr/"
                # kwargs_xarray = dict(engine="zarr", chunks={"ocean_time":1})
                # reader = reader_ROMS_native.Reader(loc, kwargs_xarray=kwargs_xarray) 
            
            reader = reader_ROMS_native.Reader(loc, kwargs_xarray=kwargs_xarray)
            self.o.add_reader([reader])
            self.reader = reader
            # can find reader at manager.o.env.readers['roms native']      
            
            self.oceanmodel_lon0_360 = oceanmodel_lon0_360
            
        else:
            raise ValueError("reader did not set an ocean_model")



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

        if not self.seed_seafloor:
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
            outfile=f'output-results_{datetime.datetime.utcnow():%Y-%m-%dT%H%M:%SZ}.nc'
        )
    
    @property
    def _config(self):
        """Surface the model configuration."""
        
        return self.o._config
    
    # def _ptm_config_model(self):
    #     """Add PTM configuration into opendrift config."""
        
    #     # combine PTM config with opendrift metadata for some parameters        
    #     od_config_to_add_ptm_config_to = {key: self.show_config_model(key=self.config_ptm[key]["od_mapping"]) for key in self.config_ptm.keys() if "od_mapping" in self.config_ptm[key].keys()}

    #     # otherwise things get combined too much        
    #     od_config_to_add_ptm_config_to = copy.deepcopy(od_config_to_add_ptm_config_to)
        
    #     for key in od_config_to_add_ptm_config_to.keys():
    #         # self.config_ptm[key].update(od_config_to_add_ptm_config_to[key])
    #         od_config_to_add_ptm_config_to[key].update(self.config_ptm[key])

    #     self.config_ptm.update(od_config_to_add_ptm_config_to)

        
    #     # dict1 = copy.deepcopy(self._config)
    #     # dict2 = copy.deepcopy(self.config_ptm)
        
    #     # dictB = {k: {**v, **dict2[k]} if k in dict2.keys() else {**v} for k, v in dict1.items()}
        
    #     # dict2.update(dictB)
    
    def _add_ptm_config(self):        
        
        dict1 = copy.deepcopy(self._config)
        dict2 = copy.deepcopy(self.config_ptm)

        # dictB has the od_mapping version of the keys of dict2
        # e.g.  'processes:emulsification' instead of 'emulsification'
        # dictB values are the OpenDrift config parameters with config_od parameters added on top
        dictB = {v["od_mapping"]: ({**dict1[v["od_mapping"]], **v} if "od_mapping" in v and v["od_mapping"] in dict1.keys() else {**v}) for k, v in dict2.items() if "od_mapping" in v}

        # dictC is the same as dictB except the names are the PTM/OpenDrift names instead of the 
        # original OpenDrift names
        dictC = {k: {**dict1[v["od_mapping"]], **v} if "od_mapping" in v and v["od_mapping"] in dict1.keys() else {**v} for k, v in dict2.items()}
        
        # this step brings config overrides from config_ptm into the overall config
        self._config.update(dictB)
        # this step brings other model config (plus additions from mapped parameter config) into the overall config
        self._config.update(dictC)
        # # this step brings other model config into the overall config
        # self._config.update(dict2)
    
    def _add_model_config(self): 
        """Goal is to combine the config both directions:
        
        * override OpenDrift config defaults with those from opendrift_config as well
          as include extra information like ptm_level
        * bring OpenDrift config parameter metadata into config_model so application 
          could query it to get the ranges, options, etc.
        """       
        
        dict1 = copy.deepcopy(self._config)
        dict2 = copy.deepcopy(self.config_model)

        # dictB has the od_mapping version of the keys of dict2
        # e.g.  'processes:emulsification' instead of 'emulsification'
        # dictB values are the OpenDrift config parameters with config_od parameters added on top
        dictB = {v["od_mapping"]: {**dict1[v["od_mapping"]], **v} if "od_mapping" in v and v["od_mapping"] in dict1.keys() else {**v} for k, v in dict2.items()}

        # dictC is the same as dictB except the names are the PTM/OpenDrift names instead of the 
        # original OpenDrift names
        dictC = {k: {**dict1[v["od_mapping"]], **v} if "od_mapping" in v and v["od_mapping"] in dict1.keys() else {**v} for k, v in dict2.items()}


        # this step brings config overrides from config_model into the overall config
        self._config.update(dictB)
        # this step brings other model config (plus additions from mapped parameter config) into the overall config
        self._config.update(dictC)
        # self._config.update(dict2)
        # import pdb; pdb.set_trace()
    
    # def _add_od_config_old(self):
        
    #     dict1 = copy.deepcopy(self._config)
    #     dict2 = copy.deepcopy(config_model)
        
    #     # dictB is dict1 but overwrite and combine any keys from dict2 that match
    #     # combine and as relevant prioritize key/values from dict2 into dict1
    #     # dictB keys are identical to dict1        
    #     dictB = {k: {**v, **dict2[k]} if k in dict2.keys() else {**v} for k, v in dict1.items()}
    #     # import pdb; pdb.set_trace()
        
    #     # effect of this step is to have fully combined dict1 and dict2 
    #     # including nested dicts
    #     dict2.update(dictB)
    #     self._config.update(dict2)

    def get_configspec(self, prefix, substring, level,
                       ptm_level):
        """Copied from OpenDrift, then modified."""

        if not isinstance(level, list) and level is not None:
            level = [level]
        if not isinstance(ptm_level, list) and ptm_level is not None:
            ptm_level = [ptm_level]

        # check for prefix or substring comparison
        configspec = {
            k: v
            for (k, v) in self._config.items()
            if k.startswith(prefix) and substring in k
        }

        if level is not None:
            # check for levels (if present)
            configspec = {
                k: v
                for (k, v) in configspec.items()
                if "level" in configspec[k] and configspec[k]['level'] in level
            }

        if ptm_level is not None:
            # check for ptm_levels (if present)
            configspec = {
                k: v
                for (k, v) in configspec.items()
                if "ptm_level" in configspec[k] and configspec[k]['ptm_level'] in ptm_level
            }

        return configspec

    def show_config_model(self, key=None, prefix='', level=None, 
                          ptm_level=None, substring='') -> dict:
        """Show configuring for the drift model selected in configuration.
        
        Runs configuration for you if it hasn't yet been run.
        
        Parameters
        ----------
        key : str, optional
            If input, show configuration for just that key.
        prefix : str, optional
            prefix to search config for, only for OpenDrift parameters (not PTM).
        level : int, list, optional
            Limit search by level:
            * CONFIG_LEVEL_ESSENTIAL = 1
            * CONFIG_LEVEL_BASIC = 2
            * CONFIG_LEVEL_ADVANCED = 3
            e.g. 1, [1,2], [1,2,3]
        ptm_level : int, list, optional
            Limit search by level:
            * Surface to user = 1
            * Medium surface to user = 2
            * Surface but bury = 3
            e.g. 1, [1,2], [1,2,3]. To access all PTM parameters search for 
            `ptm_level=[1,2,3]`.

        
        Examples
        --------
        Show all possible configuration for the previously-selected drift model:
        
        >>> manager.show_config()
        
        Show configuration with a specific prefix:
        
        >>> manager.show_config(prefix="seed")

        Show configuration matching a substring:
        
        >>> manager.show_config(substring="stokes")
        
        Show configuration at a specific level (from OpenDrift):
        
        >>> manager.show_config(level=1)

        Show all OpenDrift configuration:
        
        >>> manager.show_config(level=[1,2,3])
        
        Show configuration for only PTM-specified parameters:
        
        >>> manager.show_config(ptm_level=[1,2,3])
        
        Show configuration for a specific PTM level:
        
        >>> manager.show_config(ptm_level=2)
        
        Show configuration for a single key:
        
        >>> manager.show_config("seed:oil_type")
        
        Show configuration for parameters that are both OpenDrift and PTM-modified:
        
        >>> m.show_config(ptm_level=[1,2,3], level=[1,2,3])
        
        """

        if key is not None:
            prefix = key
            
        output = self.get_configspec(prefix=prefix, level=level, 
                                    ptm_level=ptm_level, substring=substring)
        
        if key is not None:
            if key in output:
                return output[key]
            else:
                return output
        else:
            return output
    
    def reader_metadata(self, key):
        """allow manager to query reader metadata."""
        
        if not self.has_added_reader:
            raise ValueError("reader has not been added yet.")
        return self.o.env.readers['roms native'].__dict__[key]


# class LeewayModel(ParticleTrackingManager, OpenDrift):
#     def __init__(self)