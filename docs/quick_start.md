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

Alternatively, you can run the package with new model output by either setting up a user-input ocean model configuration yaml file or setting one up on the fly. See information for both built-in and user-input ocean models in {doc}`ocean_models`.

Details about what setup and configuration are available in {doc}`configuration`.

+++

## Python Package

Run directly from the Lagrangian model you want to use, which will inherit from the manager class. For now there is one option of `OpenDriftModel`.

```
import particle_tracking_manager as ptm

m = ptm.OpenDriftModel(steps=1)
# Can modify `m` between these steps, or look at `OpenDrift` config with `m.drift_model_config()`
m.run_all()
```

This example uses defaults to fill in important information including the `ocean_model` ("CIOFSOP"), the `start_time` (something during CIOFSOP's time range), lon/lat (a location in Cook Inlet, AK); `steps`, `duration`, or `end_time` is required with `start_time`. Most users will need to add the `ocean_model_local=False` flag, if not running on Axiom servers directly.

Then find results in file `m.config.output_file`.

+++

## Command Line Interface

The equivalent for the set up above for using the command line is:

```
ptm steps=1
```

To just initialize the simulation and print the `OpenDrift` configuration to screen without running the simulation, add the `--dry-run` flag:

```
ptm steps=1 --dry-run
```

You can choose to output one or more plots with the `plots` keyword. For example, the following will output a spaghetti plot made from the track file, using OpenDrift's plotting capabilities (also running with other inputs):

```
ptm lon=-151.2 lat=59.1 start_time=2006-02-02T00:00 ocean_model=NWGOA duration="1h" plots="{'spaghetti': {}}"
```

You can instead run your simulation and then later make plots with:

```
ptm output_file=[path for outfile including suffix] plots="{'spaghetti': {}}"
```

`m.config.output_file` is printed to the screen after the command has been run. `ptm` is installed as an entry point with `particle-tracking-manager`.

Note that each plot option should be input in a dictionary but then within a string to be correctly interpreted by the CLI. More information on plot options in PTM is available in {ref}`plots`. Many options are available, including animations (see [OpenDrift docs for more information](https://opendrift.github.io/)).


+++

(new_reader)=
## Python package with local model output

There is a short example of ROMS ocean model output available through `xroms` that we will use for demonstration purposes. A configuration file for it is included in this package under the name "TXLA". We will use this example here, but also the configuration file acts as an example template for users who want to set up their own ocean model configuration files. More information on this template {ref}`here<user_templates>`.

To use the "TXLA" ocean model you need to set `ocean_model_local=False` to access the file correctly.

```{code-cell} ipython3

import particle_tracking_manager as ptm
import ast

m = ptm.OpenDriftModel(lon=-90, lat=28.7, number=10, steps=20,
                       start_time="2009-11-19T13:00",
                       use_static_masks=True, plots={'spaghetti': {}},
                       ocean_model="TXLA", ocean_model_local=False)

m.run_all()
```

You can access the plot name as follows (note you need to use `ast.literal_eval()` because `plots` is stored as a string in the file).

```{code-cell} ipython3
ast.literal_eval(m.config.plots)["spaghetti"]["filename"]
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
m.reader_metadata(<key>)
```

Show schema details — many more details on this in {doc}`configuration`:

```
m.config.model_json_schema()
```

Show configuration values for your model:

```
m.config.model_dump()
```
