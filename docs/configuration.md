# Configuration and Setup Options

## Configuration Overview

Configuration parameters are shown in `m.show_config()` for:

* the specified `drift_model` (from `m._config` which for OpenDriftModel points to `m.o._config`)
* configuration added from `ParticleTrackingManager`  (`config_ptm`)
* configuration added from `OpenDriftModel` (as would also be true with any model included in the future) (`config_model`). This configuration includes parameters that point to configuration parameters from a different `drift_model` than a given instance of `OpenDriftModel` was initialized with, which leads to these parameters being present in `m.show_config()`. The following example shows a `Manager` `m` initialized with `drift_model=="OceanDrift"` but when queried for emulsification, the translated PTM parameter name for the OpenDrift parameter `emulsification`, or `processes:emulsification` itself, they both show the configuration information added from OpenDriftModel:

```
m.show_config(key="emulsification")
{'default': True, 'od_mapping': 'processes:emulsification', 'ptm_level': 2, 'value': True}

m.show_config(key="processes:emulsification")
{'default': True, 'od_mapping': 'processes:emulsification', 'ptm_level': 2, 'value': True}
```

For comparison, if `drift_model=="OpenOil"` this would look like the following, in which the parameters contain more config information which came in from OpenDrift itself:

```
m = ptm.OpenDriftModel(drift_model="OpenOil")

m.show_config(key="emulsification")

{'type': 'bool',
 'default': True,
 'description': 'Surface oil is emulsified, i.e. water droplets are mixed into oil due to wave mixing, with resulting increase of viscosity.',
 'level': 2,
 'value': True,
 'od_mapping': 'processes:emulsification',
 'ptm_level': 2}

m.show_config(key="processes:emulsification")

{'type': 'bool',
 'default': True,
 'description': 'Surface oil is emulsified, i.e. water droplets are mixed into oil due to wave mixing, with resulting increase of viscosity.',
 'level': 2,
 'value': True,
 'od_mapping': 'processes:emulsification',
 'ptm_level': 2}

```


### Show different sources of config

PTM-level config:

```
m.config_ptm
```

Model-level config:

```
m.config_model
```

Show OpenDrift config only. This is tricky because the configurations get mixed up together to keep all information consistent across parameters. The kludge way to show these is since all OpenDrift config parameter names have “:” in the name:

```
m.show_config(substring=":")
```

All config:

```
m.show_config()
```


### Showing Configuration Parameter Details

Show seed parameters that are in OpenDrift for `drift_model`:

```
m.show_config(prefix="seed", level=[1,2,3]).keys()
```

Show all possible configuration for the previously-selected `drift_model` (parameters that are not options will be included but will not have full config information):

```
m.show_config()
```

Show configuration with a specific prefix:

```
m.show_config(prefix="seed")
```

Show configuration matching a substring:

```
m.show_config(substring="stokes")
```

Show configuration at a specific level (from OpenDrift):

```
m.show_config(level=1)
```

Show all OpenDrift configuration:

```
m.show_config(level=[1,2,3])
```

Show configuration for only PTM-specified parameters:

```
m.show_config(ptm_level=[1,2,3])
```

Show configuration for a specific PTM level:

```
m.show_config(ptm_level=2)
```

Show configuration for a single key:

```
m.show_config("seed:oil_type")
```


## Specific Configuration Options

### Ocean model

Setting up an ocean model is also referred to as `add_reader()`.

```
m.show_config(key="ocean_model")
```

The built-in ocean models are:
* NWGOA (1999–2008) over the Northwest Gulf of Alaska (Danielson, S. L., K. S. Hedstrom, E. Curchitser,	2016. Cook Inlet Model Calculations, Final Report to Bureau of Ocean Energy Management,	M14AC00014,	OCS	Study BOEM 2015-050, University	of Alaska Fairbanks, Fairbanks,	AK,	149 pp.)
* CIOFS (1999–2022) across Cook Inlet, Alaska, a hindcast version of NOAA's CIOFS model. (Thyng, K. M., C. Liu, M. Feen, E. L. Dobbins, 2023. Cook Inlet Circulation Modeling, Final Report to Oil Spill Recovery Institute, Axiom Data Science, Anchorage, AK.)
* CIOFS_NOW (mid-2021 through 48 hours from present time) which is the nowcast/forecast version of the CIOFS model. (Shi, L., L. Lanerolle, Y. Chen, D. Cao, R. Patchen, A. Zhang,
and E. P. Myers, 2020. NOS Cook Inlet Operational Forecast System: Model development and hindcast skill assessment, NOAA Technical Report NOS CS 40, Silver Spring, Maryland, September 2020.)

An alternative ocean model can be used instead by initializing the `Manager` then setting up the reader manually, as shown in a {ref}`Quick Start<new_reader>` example:

```
import particle_tracking_manager as ptm
import xroms

m = ptm.OpenDriftModel(lon=-90, lat=28.7, number=1, steps=2)
url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
reader_kwargs = dict(loc=url, kwargs_xarray={})
m.add_reader(**reader_kwargs)
m.run_all()
```


To run an idealized scenario, no reader should be added (`ocean_model` should be left as None), then fallback configuration parameters (which are not surfaced specifically in `particle-tracking-manager`) can be manually changed, for example:

```
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

For testing purposes, all steps can be run (including added a "reader") with the selections above plus including `ocean_model="test"`.

```
from datetime import datetime
m = ptm.OpenDriftModel(lon=4.0, lat=60.0, start_time=datetime(2015, 9, 22, 6),
                       use_auto_landmask=True, ocean_model="test")

m.run_all()
```


### How to modify details for Stokes Drift

Turn on (on by default):

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

Find the options with e.g.

```
m.show_config(key='drift:tabularised_stokes_drift_fetch')
```


### Vertical Mixing

The user can change the background diffusivity with

```
m.o.set_config('vertical_mixing:background_diffusivity', 1.2e-5)  # default 1.2e-5
```
