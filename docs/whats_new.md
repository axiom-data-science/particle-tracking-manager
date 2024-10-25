# What's New

## v0.9.1 (October 25, 2024)

* Added local model option of CIOFS Fresh for which kerchunk files also can be generated on the fly

## v0.9.0 (July 26, 2024)

* Added utilities to generate kerchunk files on the fly for the time period of the simulation length for CIOFS and NWGOA. This has majorly sped up CIOFS simulations and modestly sped up NWGOA simulations.
* depth z should be negative! Fixed this in tests.
* added `start_time_end`, which adds OpenDrift capability for starting drifters over linear time frame
* fixed so unique log file is output for each simulation even if run in a script, and has the same name as `output_file`.
* small fix to histogram plot

## v0.8.4 (April 24, 2024)

* updated the `ptm_level` of a bunch of config parameters

## v0.8.3 (April 23, 2024)

* removed `Dcrit` because realized it is not necessary
* improved log handling for CLI
* changed `OpenDrift` default handling so they are now changed to None

## v0.8.2 (April 10, 2024)

* updated docs
* improved `drift_model_config()`
* updated tests
* now include PTM metadata with output file

## v0.8.1 (April 5, 2024)

* updated docs

## v0.8.0 (April 2, 2024)

* `time_step_output` behavior has changed — 1 hour by default
* `time_step` is now 5 min by default
* added `Dcrit` parameter for accurately finding where drifters are stranded in tidal flats
* `vertical_mixing` is True by default now
* added seafloor_action option
* fixed some Leeway/3D handling and log messaging
* export_variables are specific to drift_model as needed
* do not drop zeta anymore since used in opendrift
* output_file is now an option


## v0.7.1 (February 21, 2024)

* Small fix to some attributes to be less verbose
* Fix setup.cfg to have correct config path since name changed


## v0.7.0 (February 21, 2024)

* Now initialize all class attributes with None and removed usage of `hasattr` which simplifies and clarifies some code.
* Improved handling of `start_time`, `end_time`, `duration`, and `steps` in `manager.py` which fixed a bug in which users couldn't input `start_time` and have the simulation run successfully.
* simplified handling of `horizontal_diffusivity` in `opendrift` model.
* user can change `end_time`, `duration`, and `steps` and have the others update accordingly. Tests added to check this.
* changed known model "CIOFS_now" to "CIOFSOP" to avoid upper/lower issues and include "OP" for "operational".
* many more tests and improved behavior for attribute checks and updates


## v0.6.0 (February 15, 2024)

* is set up to tell `opendrift` ROMS reader to save the interpolator to a cache that is set up the first time it is run. This only works with the newest dev version of `opendrift` at the moment, and the files saved are hundreds of MB, but it speeds up the simulations pretty well (12 to 30 seconds).
* reworked which variables are dropped in which scenarios for `opendrift` and integrated with using wetdry vs static masks.
* added package `appdirs` to manage the cache for storing interpolator pickles.
* fix to CLI so duration input is formatted correctly.
* can now input `name` to accompany user-input `xarray Dataset` for `ocean_model`.
* added `ocean_model` "CIOFS_now" local and remote links.


## v0.5.0 (February 12, 2024)

* updated to using version of `opendrift` in which you can input an xarray Dataset directly
* added new parameter for built-in ocean_models to specify whether to look locally or remote for the output (`ocean_model_local`)
* added local model output information for known models using parquet files for kerchunk access to model output
* changed `max_speed` parameter, which controls buffer size in `opendrift`, to 2 from 5.
* improved handling of "steps", "duration", and "end_time" parameters.
* improved reader interaction and speed with `opendrift` by dropping unnecessary variables from ocean_model Dataset, separating out the `standard_name` mapping input to the ROMS reader in `opendrift`, added option for whether or not to use wet/dry masks in ocean_model output if available


## v0.4.0 (January 25, 2024)

* modified level of surfacing for some configuration parameters
* made `ptm` an entry point
* finished removing WKT code, which hadn't been working
* added “excludestring” as an option for filtering configuration parameters
* updated checks for necessary `drift_model=="Leeway"` and parameter combinations.
* updated docs according to software updates
