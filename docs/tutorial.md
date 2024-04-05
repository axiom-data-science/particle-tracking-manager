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
```

## Ocean Models

### Known Models

Three ocean models are built into PTM and can be accessed by name `ocean_model=` "NWGOA", "CIOFS", and "CIOFSOP", and either accessed remotely or locally if run on an internal server (at Axiom) (with `ocean_model_local=True`).

### Wet/dry vs. Static Masks

The known models in PTM have wet/dry masks from ROMS so they have had to be specially handled, requiring some new development in `OpenDrift`. There are two options:

* (DEFAULT) Use the typical, static, ROMS masks (`mask_rho`, `mask_u`, `mask_v`). For ROMS simulations run in [wet/dry mode](https://www.myroms.org/wiki/WET_DRY), grid cells in `mask_rho` are 0 if they are permanently dry and 1 if they are ever wet. This saves some computational time but is inconsistent with the ROMS output files in some places since the drifters may be allowed (due to the static mask) to enter a cell they wouldn't otherwise. However, it doesn't make much of a difference for simulations that aren't in the tidal flats.
* Use the time-varying wet/dry masks (`wetdry_mask_rho`, `wetdry_mask_u`, `wetdry_mask_v`). This costs some more computational time but is fully consistent with the ROMS output files. This option should be selected if drifters are expected to run in the tidal flats.

### User-input Models

As opposed to known models, a user can input their own xarray Dataset, which we will do for this tutorial. When you input your own Dataset, you have to add the reader by hand as opposed to being able to input the `ocean_model` name in the initial call.

```{code-cell} ipython3
url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
ds = xr.open_dataset(url, decode_times=False)
```

## Drift Models

After a drift simulation is run, results can be found in file with name `m.outfile_name`.

### OceanDrift (Physics)

This model can in 2D or 3D with or without horizontal or vertical mixing, wind drift, Stokes drift, etc. By default this would be run at the surface in 2D but we can input selections to run in 3D starting at depth.

#### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(lon=-90, lat=28.7, number=10, steps=40,
                       z=-5, do3D=True, horizontal_diffusivity=100,)
```

The drift_model-specific parameters chosen by the user and PTM for this simulation are:

```{code-cell} ipython3
m.drift_model_config()
```

You can also find the full PTM and `OpenDrift` configuration information with:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
m.show_config()
```

#### Add reader and run

```{code-cell} ipython3
m.add_reader(ds=ds)
m.run_all()
```

#### Plot

```{code-cell} ipython3
m.o.plot(linecolor="z", fast=True, cmap=cmo.deep)
```

### Leeway (Search and Rescue)

These are simulations of objects that stay at the surface and are transported by both the wind and ocean currents at rates that depend on how much the object sticks up out of and down into the water. The constants to use for those rates have been experimentally determined by the coastguard and are used in this model.

#### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="Leeway", lon = -89.8, lat = 29.08,
                       number=10, steps=40,
                       object_type="Fishing vessel, general (mean values)")

# This drift model requires wind data to be set which isn't present in model output
m.o.set_config('environment:constant:x_wind', -1)
m.o.set_config('environment:constant:y_wind', 1)
```

The objects that can be modeled are:

```{code-cell} ipython3
m.show_config(key="seed:object_type")["enum"]
```

The drift_model-specific parameters chosen by the user and PTM for this simulation are:

```{code-cell} ipython3
m.drift_model_config()
```

#### Add reader and run

```{code-cell} ipython3
m.add_reader(ds=ds)
m.run_all()
```

#### Plot

```{code-cell} ipython3
m.o.plot(fast=True)
```

### LarvalFish

This model simulates eggs and larvae that move in 3D with the currents and some basic behavior and vertical movement. It also simulates some basic growth of the larvae.

There are specific seeding options for this model:
* 'diameter'
* 'neutral_buoyancy_salinity'
* 'stage_fraction'
* 'hatched'
* 'length'
* 'weight'

#### Eggs

An optional general flag is to initialize the drifters at the seabed, which might make sense for eggs and is demonstrated here.

##### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="LarvalFish", lon=-89.85, lat=28.8, number=10, steps=45,
                       do3D=True, seed_seafloor=True)
```

The drift_model-specific parameters chosen by the user and PTM for this simulation are:

```{code-cell} ipython3
m.drift_model_config()
```

##### Add reader and run

```{code-cell} ipython3
m.add_reader(ds=ds)
m.run_all()
```

##### Plot

```{code-cell} ipython3
m.o.plot(linecolor="z", fast=True, cmap=cmo.deep_r)
```

```{code-cell} ipython3
m.o.plot_property('length')
m.o.plot_property('weight')
m.o.plot_property('diameter')
m.o.plot_property('stage_fraction')
```

Output from the simulation can be viewed in the history or elements, or from the output file.

```{code-cell} ipython3
m.outfile_name
```

```{code-cell} ipython3
m.o.history["z"].data
```

```{code-cell} ipython3
m.o.elements
```

#### Hatched!

##### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="LarvalFish", lon=-89.85, lat=28.8, number=10, steps=45,
                       do3D=True, seed_seafloor=True, hatched=1)
```

The drift_model-specific parameters chosen by the user and PTM for this simulation are:

```{code-cell} ipython3
m.drift_model_config()
```

##### Add reader and run

```{code-cell} ipython3
m.add_reader(ds=ds)
m.run_all()
```

##### Plot

```{code-cell} ipython3
m.o.plot(linecolor="z", fast=True, cmap=cmo.deep_r)
```

```{code-cell} ipython3
m.o.plot_property('length')
m.o.plot_property('weight')
m.o.plot_property('diameter')
m.o.plot_property('stage_fraction')
```

### OpenOil

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

#### Initialize manager `m`

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="OpenOil", lon=-89.85, lat=28.08, number=10, steps=45,
                       z=-10, do3D=True, oil_type='GENERIC BUNKER C')
m.o.set_config('environment:constant:x_wind', -1)
m.o.set_config('environment:constant:y_wind', 1)
```

List available oil types from NOAA's ADIOS database:

```{code-cell} ipython3
m.show_config(key="seed:oil_type")
```

The drift_model-specific parameters chosen by the user and PTM for this simulation are:

```{code-cell} ipython3
m.drift_model_config()
```

#### Add reader and run

```{code-cell} ipython3
m.add_reader(ds=ds)
m.run_all()
```

#### Plot

```{code-cell} ipython3
m.o.plot(linecolor="z", fast=True, cmap=cmo.deep_r)
```

Plot the oil budget.

```{code-cell} ipython3
m.o.plot_oil_budget(show_wind_and_current=False)
```

```{code-cell} ipython3

```
