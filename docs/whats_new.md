# What's New

## v2.1.2 (December 19, 2025)

* Fixed bug so that CIOFSOP `end_time` is updated before the simulation run `end_time` gets validated
  against it.


## v2.1.1 (December 17, 2025)

* Fixed bug when using geojson particle seeding. Now `PTM` will properly assign `radius` and `radius_type` when using a geojson `Point`, `LineString`, or `Polygon`.

## v2.1.0 (December 15, 2025)

* Fixed bug so now CIOFSOP `end_time` is updated each time a new instance of `OpenDriftModel` is created.
* fixed logging issue so `OpenDrift` logs show up more fully again.
* `seed_flag` is no longer in the schema and whether to input particle locations is determined by whether `lon` and `lat` are input or a `geojson` input is input. If neither are input, default lon/lat are available that work with the default ocean model.

## v2.0.6 (December 11, 2025)

* `output_format="parquet"` is still present and will work with the newest version of `OpenDrift` once a PR is merged.
* A new option has been added: `output_format="both"`. This saves `OpenDrift` output to a netCDF file but then after the simulation is over also saves a parquet file, not using `OpenDrift` for that step. This allows PTM to produce both a netCDF file and parquet file for the same `OpenDrift` simulation, which is helpful because the same metadata is not currently available in an output parquet file as is a netCDF file, so you can't go between without losing information.

## v2.0.5 (December 10, 2025)

* Changed the schema for `HarmfulAlgalBloom` so that "PN", for example, is input, though a nice label is also provided through "title" in the schema.

## v2.0.4 (Decemeber 9, 2025)

* Fixed a log file path bug where the log file path was just the filename only.
  Because of that it would get created in the current working directory and not
  in the outputs directory next to the simulation output.
* Refactored the `SetupOutputFiles` model to simplify things a bit and update tests.


## v2.0.3 (Decemeber 5, 2025)

* Fixed docs.


## v2.0.2 (Decemeber 5, 2025)

* Changed "name" of `HarmfulAlgalBloom` `species_type` enum to be an abbreviation "PN" and the value for the initial species to be "Pseudo nitzschia" instead of "Pseudo_nitzschia".
* Added to the `HarmfulAlgalBloom` schema to have a "oneOf" output containing the preset values associated with a `species_type` in the description.


## v2.0.0 (December 4, 2025)

* Added a new model scenario: `HarmfalAlgalBloom`.
* Fix a bug for running backward in time.
* Ocean model configuration files now need to include `chunks`


## v0.13.2 (November 20, 2025)

* If timezone-aware `start_time`, `start_time_end`, or `end_time` is input, the timezone is now removed and it is assumed they are in the same timezone as the model times.


## v0.13.1 (November 19, 2025)

* Fixed kerchunk JSON file filtering.
* Added CIOFS3 as ocean model option.
* Refactored `OilTypeEnum` so that it's value is only the oil ID. Also add the
  "oneOf" json schema mapping directly to the field's `json_schema_extra` so that
  it is part of the schema by default.


## v0.12.2 (April 15, 2025)

* fixed issue when setting `vertical_mixing` False for `OpenOil` was not passed through correctly
* now `do3d=False` and `vertical_mixing=True` can be set at the same time.
* updated some config descriptions
* found error in `OilTypeEnum` leading to oil_types with duplicate labels not being separated out in the Enum. This is fixed. Now oil should be input with either the ID from ADIOS or with the syntax `(ID, label)`. Once in the configuration, it will be presented as only `label` since that is how `OpenDrift` accepts `oil_type`.
* `OilTypeEnum` was reordered so the first 5 oils are most relevant to Alaska work.
* Reverted `time_step` units back to seconds from minutes. It was being mapped unexpectedly in `OpenDrift` so using seconds is better.
* Updated list of options for `object_type` and it is now fully consistent with that in `OpenDrift`.
* Now enforce that for Leeway model if `hatched==1` then `stage_fraction=None`, for seeding.
* Updated descriptions in a lot of parameters in `config_the_manager.py` and `config_opendrift.py`.
* Updated defaults in config classes.


## v0.12.1 (April 8, 2025)

* Correction to `interpolator_filename` handling and improvement in testing.


## v0.12.0 (April 9, 2025)

* Major refactoring:
    * Removed `surface_only` flag.
    * Changed default value for `vertical_mixing` to False to match `do3D` default of False.
    * Moved some configuration parameters between configuration objects.
    * Major improvements of log handling.
    * `time_step` and `time_step_output` changed to minutes from seconds.
    * ocean_model_registry for known and user-input models
    * changed some logger statement from "info" to "debug"
    * now pinned to `opendrift` v1.13.0 and `kerchunk` v0.2.7 because not ready for zarr v3
    * can run plots from a parquet or netcdf output file now
    * can't run idealized simulations using `OpenDrift` directly in PTM anymore but this could be added back in if needed
    * updated docs
    * property plot now requires keyword "variable" instead of "prop" (this change is from `OpenDrift`)
    * most configuration parameters are under `m.config` now instead of just `m`, if `m` represents a Manager instance.
    * a geojson dict can be input but it is not fully working at the moment
    * `oil_type` can be input as the name or the id from the adios database.
    * `.model_json_schema()` is overridden in `OpenDriftConfig` to include a bit of custom code to modify the `oil_type` output in the JSON schema "properties" area. This shouldn't affect anything other than it being available if people want that.


## v0.11.2 (February 6, 2025)

* Suffix for parquet file format is now ".parquet" instead of ".parq".
* Added a method to run plots from saved OpenDrift output file; also available in CLI. Updated docs.


## v0.11.1 (February 4, 2025)

* Move known model hard-wired model times into the class so they are refreshed each time the library is read.
* Add dockerfile for running PTM in a container.

## v0.10.1 (January 30, 2025)

* Added built-in way to create plots for simulation using OpenDrift. Details available in {ref}`plots`.
* User can now input a location to both save and read the interpolator, which avoids using the built-in cache location.

## v0.9.6 (November 15, 2024)

* made caching directory creation and saving to cache optional with input option `use_cache`.

## v0.9.5 (November 14, 2024)

* fixed error in output file

## v0.9.4 (November 14, 2024)

* Updated locations for local model output.

## v0.9.3 (November 13, 2024)

* Moved `output_format` parameter to manager config from model config
* Changed source location for CIOFSOP local model output

## v0.9.2 (November 11, 2024)

* Added ability to save output files as parquet instead of netcdf.
* Partially updated docs

## v0.9.1 (October 25, 2024)

* Added local model option of CIOFS Fresh for which kerchunk files also can be generated on the fly.

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
