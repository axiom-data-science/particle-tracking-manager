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

# Tutorial

Particle Tracking Manager (PTM) is a wrapper around particle tracking codes to easily run particle simulations in select (or user-input) ocean models. Currently, `OpenDrift` is included. In this tutorial we demonstrate using the four wrapped drift models from `OpenDrift` along with some high level configuration changes.

```{code-cell} ipython3
import xarray as xr
import particle_tracking_manager as ptm
import xroms
import cmocean.cm as cmo
import pprint
```

After a drift simulation is run, results can be found in file with name `m.config.output_file`. Detailed information on ocean models is available {doc}`ocean_models`. The example below use the user-defined "TXLA" ocean model.

## OceanDrift (Physics)

This model can run in 2D or 3D with or without horizontal or vertical mixing, wind drift, Stokes drift, etc.

### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(lon=-90, lat=28.7, number=10, steps=40,
                       z=-5, do3D=True, horizontal_diffusivity=100,
                       ocean_model="TXLA",
                       ocean_model_local=False,
                       start_time="2009-11-19T12:00",
                       plots={'spaghetti': {'linecolor': 'z', 'cmap': 'cmo.deep'}})
```

The configuration parameters for this simulation are:

```{code-cell} ipython3
pprint.pprint(m.config.model_dump())
```

### Run

```{code-cell} ipython3
m.run_all()
```


## Harmful Algal Bloom

The goal of this scenario is to examine the transport of an existing harmful algal bloom or determine where an existing bloom originated, respecting temperature and salinity bounds for the species.

### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="HarmfulAlgalBloom", lon = -89.8, lat = 29.08,
                       number=10, steps=40,
                       ocean_model="TXLA",
                       start_time="2009-11-19T12:00",
                       ocean_model_local=False,
                       species_type="Pseudo nitzschia",
                       plots={'spaghetti': {}})
```

The currently available species are:

```{code-cell} ipython3
ptm.HarmfulAlgalBloomModelConfig.model_json_schema()["$defs"]["HABSpeciesTypeEnum"]["enum"]
```

where `custom` is an option to allow the user to input all necessary parameters to represent a species of their choice.

There are parameters available just for the HAB model:

```{code-cell} ipython3
import json
print(json.dumps(ptm.models.opendrift.enums.species_types.HABParameters.model_json_schema(), indent=2))
```


The special parameters for each available species are:

```{code-cell} ipython3
species = ptm.HarmfulAlgalBloomModelConfig.model_json_schema()["$defs"]["HABSpeciesTypeEnum"]["enum"]
species.remove('custom')
for specie in species:
    print(f"{specie}: {ptm.models.opendrift.enums.species_types.SPECIES_HAB_DEFAULTS[specie]}")
```

The regular parameters that are set for each available species are:

```{code-cell} ipython3
for specie in species:
    print(f"{specie}: {ptm.models.opendrift.enums.species_types.SPECIES_HAB_MANAGER_DEFAULTS[specie]}")
```

The configuration parameters for this simulation are:

```{code-cell} ipython3
pprint.pprint(m.config.model_dump())
```

### Run

```{code-cell} ipython3
m.run_all()
```


## Leeway (Search and Rescue)

These are simulations of objects that stay at the surface and are transported by both the wind and ocean currents at rates that depend on how much the object sticks up out of and down into the water. The constants to use for those rates have been experimentally determined by the coastguard and are used in this model.

### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="Leeway", lon = -89.8, lat = 29.08,
                       number=10, steps=40,
                       ocean_model="TXLA",
                       start_time="2009-11-19T12:00",
                       ocean_model_local=False,
                       object_type="Fishing vessel, general (mean values)",
                       plots={'spaghetti': {}})

# setup opendrift object (need to do this to set the wind in the next step)
m.setup_for_simulation()

# This drift model requires wind data to be set which isn't present in model output
m.o.set_config('environment:constant:x_wind', -1)
m.o.set_config('environment:constant:y_wind', 1)
```

The objects that can be modeled are:

```{code-cell} ipython3
ptm.LeewayModelConfig.model_json_schema()["$defs"]["ObjectTypeEnum"]["enum"]
```

You can run the previous command without initializing a class instance. If you don't want to remember which scenario object to check and don't mind initializating, you can instead run from your manager instance:

```{code-cell} ipython3
m.config.model_json_schema()["$defs"]["ObjectTypeEnum"]["enum"]
```

The configuration parameters for this simulation are:

```{code-cell} ipython3
pprint.pprint(m.config.model_dump())
```

### Run

```{code-cell} ipython3
m.run_all()
```

## LarvalFish

This model simulates eggs and larvae that move in 3D with the currents and some basic behavior and vertical movement. It also simulates some basic growth of the larvae.

There are specific seeding options for this model:
* 'diameter'
* 'neutral_buoyancy_salinity'
* 'stage_fraction'
* 'hatched'
* 'length'
* 'weight'

### Eggs

An optional general flag is to initialize the drifters at the seabed, which might make sense for eggs and is demonstrated here.

#### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="LarvalFish", lon=-89.85, lat=28.8, number=10, steps=45,
                       z=None,
                       do3D=True, seed_seafloor=True,
                       ocean_model="TXLA",
                       start_time="2009-11-19T12:00",
                       ocean_model_local=False,
                       plots={'spaghetti': {'linecolor': 'z', 'cmap': 'cmo.deep'},
                              'property1': {'variable': 'length'},
                              'property2': {'variable': 'weight'},
                              'property3': {'variable': 'diameter'},
                              'property4': {'variable': 'stage_fraction'}})
```

The configuration parameters for this simulation are:

```{code-cell} ipython3
pprint.pprint(m.config.model_dump())
```

#### Run

```{code-cell} ipython3
m.run_all()
```

Output from the simulation can be viewed in the history or elements, or from the output file.

```{code-cell} ipython3
m.config.output_file
```

```{code-cell} ipython3
m.o.result["z"]
```

```{code-cell} ipython3
m.o.elements
```

### Hatched!

#### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="LarvalFish", lon=-89.85, lat=28.8, number=10, steps=45,
                       do3D=True, seed_seafloor=True, hatched=1, stage_fraction=None,
                       z=None,
                       ocean_model="TXLA",
                       start_time="2009-11-19T12:00",
                       ocean_model_local=False,
                       plots={'spaghetti': {'linecolor': 'z', 'cmap': 'cmo.deep'},
                              'property1': {'variable': 'length'},
                              'property2': {'variable': 'weight'},
                              'property3': {'variable': 'diameter'},
                              'property4': {'variable': 'stage_fraction'}})
```

The configuration parameters for this simulation are:

```{code-cell} ipython3
pprint.pprint(m.config.model_dump())
```

#### Run

```{code-cell} ipython3
m.run_all()
```


## OpenOil

This model simulates the transport of oil. Processes optionally modeled (which are included in PTM by default) include:
* "emulsification"
* "dispersion"
* "evaporation"
* "update_oilfilm_thickness"
* "biodegradation"

There are also specific seeding options for this model:
* "m3_per_hour"
* "oil_film_thickness"
* "droplet_size_distribution"
* "droplet_diameter_mu"
* "droplet_diameter_sigma"
* "droplet_diameter_min_subsea"
* "droplet_diameter_max_subsea"

### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="OpenOil", lon=-89.85, lat=28.08, number=10, steps=45,
                       z=-10, do3D=True, oil_type="EC00561",
                       start_time="2009-11-19T12:00", ocean_model="TXLA",
                       ocean_model_local=False,
                       )
m.setup_for_simulation()
m.o.set_config('environment:constant:x_wind', -1)
m.o.set_config('environment:constant:y_wind', 1)
```

Note that `oil_type` was input by its oil id, in order to disambiguate the oils in ADIOS.

List of available oil types can be found in the
1. NOAA's ADIOS database https://adios.orr.noaa.gov/oils
2. or in the library's `OIL_ID_TO_NAME` dictionary:
    ```{code-cell} ipython3
    ptm.models.opendrift.enums.oil_types.OIL_ID_TO_NAME
    ```

You can also find the oil IDs by name in said [database](https://adios.orr.noaa.gov/oils)
or in the library's `NAME_TO_OIL_ID` dictionary. E.g.:

```{code-cell} ipython3
ptm.models.opendrift.enums.oil_types.NAME_TO_OIL_ID["ALASKA NORTH SLOPE"]
```

Keep in mind that some oil types share the same name which is why `NAME_TO_OIL_ID`
contains lists of IDs.

The configuration parameters for this simulation are:

```{code-cell} ipython3
pprint.pprint(m.config.model_dump())
```


### Run

```{code-cell} ipython3
m.run_all()
```


Run the plots after the simulation has finished:
```{code-cell} ipython3
import particle_tracking_manager.models.opendrift.plot as plot
plots = plot.make_plots_after_simulation(m.config.output_file,
                                 plots={'spaghetti': {'linecolor': 'z', 'cmap': 'cmo.deep'},
                                        'oil': {'show_wind_and_current': True}})
```

To show the second plot:

```{code-cell} ipython3
from IPython.display import Image

image_filename = plots["oil"]["filename"]
Image(filename=image_filename)
```
