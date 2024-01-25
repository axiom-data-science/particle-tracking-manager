---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3.12.0 ('ptm')
  language: python
  name: python3
---

# Quick Start Guide

+++

The simplest way to run `particle-tracking-manager` is to choose a built-in ocean model and select a location to initialize drifters, then use the built-in defaults for everything else (including start time which defaults to the first time step in the model output). You can do this interacting with the software as a Python library or using a command line interface.

Alternatively, you can run the package with new model output by inputting the necessary information into the `Manager`.

Details about what setup and configuration are available in {doc}`configuration`.

+++

## Python Package

Run directly from the Lagrangian model you want to use, which will inherit from the manager class. For now there is one option of `OpenDriftModel`.

```
import particle_tracking_manager as ptm

m = ptm.OpenDriftModel(ocean_model="NWGOA", lon=-151, lat=59)
m.run_all()
```

Then find results in file `m.outfile_name`.

+++

## Command Line Interface

The equivalent for the set up above for using the command line is:

```
ptm lon=-151 lat=59 ocean_model=NWGOA
```

`m.outfile_name` is printed to the screen after the command has been run. `ptm` is installed as an entry point with `particle-tracking-manager`.

+++

(new_reader)=
## Python package with local model output

This demo will run using easily-available ROMS model output from `xroms`.

```{code-cell} ipython3

import particle_tracking_manager as ptm
import xroms


m = ptm.OpenDriftModel(lon = -90, lat = 28.7, number=1, steps=2)

url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
reader_kwargs = dict(loc=url, kwargs_xarray={})
m.add_reader(**reader_kwargs)

# m.run_all() or the following
m.seed()
m.run()
```

## Idealized simulation

To run an idealized scenario, no reader should be added but configuration parameters can be manually changed, for example:

```{code-cell} ipython3
import particle_tracking_manager as ptm
from datetime import datetime
m = ptm.OpenDriftModel(lon=4.0, lat=60.0, start_time=datetime(2015, 9, 22, 6),
                       use_auto_landmask=True,)

# idealized simulation, provide a fake current
m.o.set_config('environment:fallback:y_sea_water_velocity', 1)

# seed
m.seed()

# run simulation
m.run()
```

## Ways to Get Information

Check drifter initialization properties:

```
m.initial_drifters
```

Look at reader/ocean model properties:

```
m.reader
```

Get reader/ocean model properties (gathered metadata about model):

```
m.reader_metadata(key)
```

Show configuration details — many more details on this in {doc}`configuration`:

```
m.show_config()
```
