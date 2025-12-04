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

# Configuration and Setup Options

## Configuration Classes

A handful of `pydantic` BaseModels make up the configuration for PTM. This allows for straight-forward parameter definitions, including min/maxes, which can be shared via JSON schema, and validation, all without even touching the main modules.

The main configuration classes are:
1. `TheManagerConfig`
1. `OpenDriftConfig` and instances `LarvalFishModelConfig`, `LeewayModelConfig`, `OceanDriftModelConfig`, `OpenOilModelConfig`, `HarmfulAlgalBloomModelConfig`

Other configuration classes are:
1. `OceanModelConfig`
1. `OceanModelRegistry`
1. `OceanModelSimulation`
1. `LoggerConfig`
1. `ParticleTrackingState`
1. `SetupOutputFiles`

### Retrieve JSON schemas

To retrieve the JSON schema for most parameters related to a PTM run, you can access through one of the scenario classes:

#### OceanDriftModelConfig

```{code-cell} ipython3
import particle_tracking_manager as ptm
import pprint
import json

schema = ptm.OceanDriftModelConfig.model_json_schema()
print(json.dumps(schema, indent=2))
```

#### LarvalFishModelConfig

```{code-cell} ipython3
schema = ptm.LarvalFishModelConfig.model_json_schema()
print(json.dumps(schema, indent=2))
```

#### LeewayModelConfig

```{code-cell} ipython3
schema = ptm.LeewayModelConfig.model_json_schema()
print(json.dumps(schema, indent=2))
```

#### OpenOilModelConfig

```{code-cell} ipython3
schema = ptm.OpenOilModelConfig.model_json_schema()
print(json.dumps(schema, indent=2))
```

#### HarmfulAlgalBloomModelConfig

```{code-cell} ipython3
schema = ptm.HarmfulAlgalBloomModelConfig.model_json_schema()
print(json.dumps(schema, indent=2))
```

#### TheManagerConfig

You can also examine the schema for `TheManagerConfig` directly, which is a subset of the parameters in the scenario classes (which inherit from `TheManagerConfig`).

```{code-cell} ipython3
schema = ptm.TheManagerConfig.model_json_schema()
print(json.dumps(schema, indent=2))
```


## Specific Configuration Options

This section is split into two: first options that are available to all models (thus are handled in the Manager) and those for `OpenDriftModel` (the only model option currently).

This is not currently a comprehensive list but a place where extra details are included that might not be clear or available elsewhere. For more information look at the configuration information (previous section) and the docstrings for each class in the API.

### OpenDriftModel options

#### Drift model

Though `OpenDrift` has more models available, the currently wrapped `drift_model` options in PTM are:

* OceanDrift: physics-only scenario (default)
* Leeway: scenario for Search and Rescue of various objects at the surface
* OpenOil: oil spill scenarios
* LarvalFish: scenario for fish eggs and larvae that can grow
* HarmfulAlgalBloom: scenario for modeling harmful algal blooms once they exist to see where they travel or where they came from

Set these with e.g.:

```
m = ptm.OpenDriftModel(drift_model="OpenOil")
```

This selection sets some of the configuration details and export variables that are relevant for the simulation.

(config:export_variables)=
#### Export Variables

All possible variables will be exported by default into the outfiles and available in memory (`m.o.result` and `m.o.result_metadata` or `m.o.get_property(<key>)` for `OpenDriftModel`).

```
m.all_export_variables()
```

To limit the variables saved in the export file, input a list of just the variables that you want to save, keeping in mind that `['lon', 'lat', 'ID', 'status','z']` will also be included regardless. For example:
```
m = ptm.OpenDriftModel(export_variables=[])
```

The default list of `export_variables` is set in `config_model` but is modified depending on the `drift_model` set and the `export_variables` input by the user.

The export variables available for each model is reflected in each class's JSON schema.


#### How to modify details for Stokes Drift

Turn on (on by default, drift model-dependent):

```
m = ptm.OpenDriftModel(stokes_drift=True)
```

If Stokes drift is on, the following is also turned on in OpenDriftModel:

```
m.o.set_config('drift:use_tabularised_stokes_drift', True)
```

or this could be overridden with

```
m.o.set_config('drift:use_tabularised_stokes_drift', False)
```

The defaults beyond that are set but they can be modified with:

```
m.o.set_config('drift:tabularised_stokes_drift_fetch', '25000')  # default
m.o.set_config('drift:stokes_drift_profile', 'Phillips')  # default
```


#### Implicit Mixing

##### Vertical Mixing

The user can change the background diffusivity with

```
m.o.set_config('vertical_mixing:background_diffusivity', 1.2e-5)  # default 1.2e-5
```


##### Horizontal Diffusivity

The user can add horizontal diffusivity which is time-step independent diffusion. In `PTM` (not `OpenDrift`) this is calculated as an estimated 0.1 m/s sub-gridscale velocity that is missing from the model output and multiplied by an estimate of the horizontal grid resolution. This leads to a larger value for NWGOA which has a larger value for mean horizontal grid resolution (lower resolution). If the user inputs their own ocean_model information, they can input their own `horizontal_diffusivity` value. Also a user can use a built-in ocean_model and the overwrite the `horizontal_diffusivity` value to 0.


##### Additional Uncertainty

One can also add time-step-dependent uncertainty to the currents and winds with `current_uncertainty` and `wind_uncertainty`, respectively.
