---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Ocean Model Configuration

Here we describe how to user existing ocean model configurations as well as how to add your own.

## Existing Configurations (Known Ocean Models)

Some ocean models are built into PTM and can be accessed by the input parameter `ocean_model`.

The built-in ocean models are:
* **CIOFS3** (1999–2024) across Cook Inlet, Alaska, the newest hindcast version of NOAA's CIOFS model. (Thyng, K. M., C. Liu, 2025. Cook Inlet Circulation Modeling - Long-term Hindcast with Improved Freshwater Forcing and Other Attributes, Final Report to the National Oceanic Atmospheric Administration National Centers for Coastal Ocean Science Kasitsna Bay Lab, Axiom Data Science, Anchorage, AK.)
* **CIOFS** (1999–2022) across Cook Inlet, Alaska, a hindcast version of NOAA's CIOFS model. (Thyng, K. M., C. Liu, M. Feen, E. L. Dobbins, 2023. Cook Inlet Circulation Modeling, Final Report to Oil Spill Recovery Institute, Axiom Data Science, Anchorage, AK.)
* **CIOFSOP** (mid-2021 through 48 hours from present time) which is the nowcast/forecast version of the CIOFS model. (Shi, L., L. Lanerolle, Y. Chen, D. Cao, R. Patchen, A. Zhang,
and E. P. Myers, 2020. NOS Cook Inlet Operational Forecast System: Model development and hindcast skill assessment, NOAA Technical Report NOS CS 40, Silver Spring, Maryland, September 2020.)
* **NWGOA** (1999–2008) over the Northwest Gulf of Alaska (Danielson, S. L., K. S. Hedstrom, E. Curchitser,	2016. Cook Inlet Model Calculations, Final Report to Bureau of Ocean Energy Management,	M14AC00014,	OCS	Study BOEM 2015-050, University	of Alaska Fairbanks, Fairbanks,	AK,	149 pp.)


### Show available ocean models

Show all available ocean_models as list of strings:

```{code-cell} ipython3
from particle_tracking_manager.ocean_model_registry import ocean_model_registry

ocean_model_registry.all()
```

### Show a specific ocean model

Show each individual ocean_model:

```{code-cell} ipython3
ocean_model_registry.show("CIOFS3")
```

```{code-cell} ipython3
ocean_model_registry.show("CIOFS")
```

```{code-cell} ipython3
ocean_model_registry.show("CIOFSOP")
```

```{code-cell} ipython3
ocean_model_registry.show("NWGOA")
```

### Return ocean model object

To instead return the ocean model config object, use `get`:

```{code-cell} ipython3
ocean_model_registry.get("CIOFSOP")
```


## User Configurations

(user_templates)=
### User Templates

User templates are defined by default in the `particle_tracking_manager/user_ocean_models` directory, and they can also be placed in a directory defined by environmental variable `PTM_CONFIG_DIR`. An example user configuration file is available to use as a template which is also used to defined example model output for use in docs and tests. The example is called "TXLA" and looks like:

```{code-cell} ipython3
ocean_model_registry.show("TXLA")
```

If you want to set up your own ocean model configuration file, start from the TXLA file and save your own version defining a different model and either place it in `user_ocean_models` or a directory defined in the `PTM_CONFIG_DIR` variable path. `ocean_model_registry` will find any *.yaml file placed in either location.

***Note:***
If you are going to run a simulation with "TXLA" you need to run with `ocean_model_local=False`. Also in the background the package downloads the necessary model output file using `pooch` if you use this model.


### On-the-fly Configurations

You can also use an xarray Dataset that is in memory as input to PTM. To do this, start by defining your dataset:

```{code-cell} ipython3
import xarray as xr
import numpy as np

ds = xr.Dataset(
    data_vars={
        "u": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "v": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "w": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "salt": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "temp": (("ocean_time", "Z", "Y", "X"), np.zeros((2, 3, 2, 3))),
        "wetdry_mask_rho": (("ocean_time", "Y", "X"), np.ones((2, 2, 3))),
        "mask_rho": (("Y", "X"), np.ones((2, 3))),
        "angle": (("Y", "X"), np.zeros((2, 3))),
        "Uwind": (("ocean_time", "Y", "X"), np.zeros((2, 2, 3))),
        "Vwind": (("ocean_time", "Y", "X"), np.zeros((2, 2, 3))),
        "Cs_r": (("Z"), np.linspace(-1, 0, 3)),
        "hc": 16,
    },
    coords={
        "ocean_time": ("ocean_time", [0, 3600], {"units": "seconds since 1970-01-01 00:00:00", "calendar": "gregorian"}),
        "s_rho": (("Z"), np.linspace(-1, 0, 3)),
        "lon_rho": (("Y", "X"), np.array([[1, 2, 3], [1, 2, 3]])),
        "lat_rho": (("Y", "X"), np.array([[1, 1, 1], [2, 2, 2]])),
    },
)
```

Next set up a dictionary defining any of the OceanModelConfig parameters you want to define as part of your ocean model/Dataset:


```{code-cell} ipython3
ds_info = dict(temporal_resolution_str="PT1H", lon_min=1, lon_max=3, lat_min=1, lat_max=2, start_time_model="1970-01-01T00:00:00", end_time_fixed="1970-01-01T01:00:00")
```

Then register your dataset:

```{code-cell} ipython3
import particle_tracking_manager as ptm

ptm.config_ocean_model.register_on_the_fly(ds_info)
```

Check that everything made it in there ok with:
```{code-cell} ipython3

ptm.ocean_model_registry.ocean_model_registry.get("ONTHEFLY")
```

At this point, until you close this kernel, you can use "ONTHEFLY" as your ocean model and have it defined as you do here. For example:

```{code-cell} ipython3
m = ptm.OpenDriftModel(ocean_model="ONTHEFLY", lon=2, lat=1.5, start_time="1970-01-01", duration="10m", horizontal_diffusivity=0)
m.add_reader(ds=ds)
```

You can subsequently run the simulation with

```{code-cell} ipython3
m.run_all()
```


## Configuration Details

### Local or remote access

Currently you can use remote or local (default) access of the built-in ocean models (`ocean_model_local=True`). This only works if you are running on Axiom servers. Local access uses `kerchunk` to set up a `kerchunk` representation of the model output required to run your particle tracking simulation. Remote access is through something like a THREDDS server or opendap link.


### Wet/dry vs. Static Masks

The built-in models in PTM have wet/dry masks from ROMS so they have had to be specially handled, requiring some new development in `OpenDrift`. There are two options:

* (DEFAULT) Use the typical, static, ROMS masks (`mask_rho`, `mask_u`, `mask_v`). For ROMS simulations run in [wet/dry mode](https://www.myroms.org/wiki/WET_DRY), grid cells in `mask_rho` are 0 if they are permanently dry and 1 if they are ever wet. This saves some computational time but is inconsistent with the ROMS output files in some places since the drifters may be allowed (due to the static mask) to enter a cell they wouldn't otherwise. However, it doesn't make much of a difference for simulations that aren't in the tidal flats.
* Use the time-varying wet/dry masks (`wetdry_mask_rho`, `wetdry_mask_u`, `wetdry_mask_v`). This costs some more computational time but is fully consistent with the ROMS output files. This option should be selected if drifters are expected to run in the tidal flats.
