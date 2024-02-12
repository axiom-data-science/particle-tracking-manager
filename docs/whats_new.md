# What's New

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
