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

(plots)=
# Detailed Built-In Plots Demonstration

PTM allows access to OpenDrift's built-in plotting capabilities with the use of an input dictionary. You can browse the types of plots from OpenDrift in their [gallery](https://opendrift.github.io/gallery/index.html); this page demonstrates how to access a variety of those plots through PTM.

The available plot types are:
* spaghetti
* property
* oil
* animation
* animation profile

To create a type of plot, the listed words must be in the plot dictionary key and it will work by word-matching. Each plot key must also be distinct. A basic version of each available plot (except "property", which requires an input property to plot) will be plotted for "all". Some examples will be demonstrated here, as well as how to get the plot whether using PTM as a Python package or through the CLI. Note that for the CLI, the `--plots` input needs to come before or after all other inputs besides `--dry-run`.

Examples will be shown with code run at the bottom of this page. First we show options.

To make two or more plots or animations of the same type for the same simulation, input multiple plot dictionaries. Just be sure that each plot key is distinct.

A plot or animation option that uses a nondefault variable must also have access to that variable in the output file, which you accomplish by requesting it as an `export_variable`.

## Spaghetti

### Basic plot

This example plots the particle tracks. Input:

```
plots={'spaghetti': {}}
```

To run this with the CLI, the plots section would be
```
--plots="{'spaghetti': {}}"
```

### Options

You can choose the filetype by including `"filetype": "jpg"` or other filetypes that `matplotlib` can handle in the "spaghetti" plot dictionary. The default is png.

You can also color the lines with a parameter from the simulations, for example, "z" to plot the depth. You can plot any variables that are in the export variables. Get more information about export variables in {ref}`config:export_variables`. This is input as `"linecolor": "z"` and you can add a colormap if you want with "cmap".

You can control other parameters for the plot too, see OpenDrift options [here](https://github.com/OpenDrift/opendrift/blob/9a7f3cbc3a08bf09dd4e02fd208f76288cf49551/opendrift/models/basemodel/__init__.py#L3297-L3339).

### Multiple plots

To make two or more spaghetti plots for the same simulation, input multiple plot dictionaries. Just be sure that each plot key includes the word "spaghetti" and isn't the same as any other plot key value.

Input:
```
import cmocean.cm as cmo

...

..., plots={'spaghetti': {},
        'spaghetti2': {'linecolor': 'z', 'cmap': cmo.deep}}, ...
```

For CLI use
```
--plots="{'spaghetti': {}, 'spaghetti2': {'linecolor': 'z', 'cmap': cmo.deep}}"
```


## Animation

### Basic

You can run a basic particle animation with

```
plots={'animation': {}}
```

To run this with the CLI, the plots section would be
```
--plots="{'animation': {}}"
```

### Options

You can choose the filetype with `"filetype": "mp4"` in the input plot dictionary. The default is "gif".

You can change the background of the animation and colormap, along with the frames per second, such as, `'animation': {'background': 'sea_surface_height', 'filetype': '.mp4', 'fps': 2, 'cmap': 'cmo.deep_r'`.

Other variables to use for background include

* sea_floor_depth_below_sea_level
* land_binary_mask
* zeta
* sea_surface_height

Other options for animations include:
* bgalpha=1
* markersize='mass_oil'
* markersize_scaling=80

Other options available from OpenDrift are found [here](https://github.com/OpenDrift/opendrift/blob/9a7f3cbc3a08bf09dd4e02fd208f76288cf49551/opendrift/models/basemodel/__init__.py#L2556-L2598).


## Oil Budget

### Basic


You can make a basic oil budget plot with

```
plots={'oil': {}}
```

To run this with the CLI, the plots section would be
```
--plots="{'oil': {}}"
```

### Options

Options are:
* `show_wind_and_current=True`
* `show_watercontent_and_viscosity=True`


## Animation Profile

This shows an animation from the side, with depth on the y-axis.


### Basic

You can make a profile animation with

```
plots={'animation_profile': {}}
```

To run this with the CLI, the plots section would be
```
--plots="{'animation_profile': {}}"
```

### Options

Useful options include:
* `markersize='mass_oil'`
* `markersize_scaling=80`
* `color='z'`
* `alpha=.5`


## Property Plot

This plots a single property over time, either for every track or the average over all the tracks.

### Basic

You can make a property plot with the following — you must input a "prop" to plot!

```
plots={'property': {'prop': 'z'}}
```

To run this with the CLI, the plots section would be
```
--plots="{'property': {'prop': 'z'}}"
```

### Options

* `'mean': True`


## Demonstration

You can mix and match the plot and animation options above.

### Demo for "all" option


```{code-cell} ipython3
import particle_tracking_manager as ptm
import xroms
import xarray as xr
import cmocean.cm as cmo
import pandas as pd

m = ptm.OpenDriftModel(lon=-90, lat=28.7, number=10, duration=pd.Timedelta("3h"),
                       do3D=True, use_static_masks=True,
                       plots={'all': {},
})

url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
ds = xr.open_dataset(url, decode_times=False)
m.add_reader(ds=ds)

m.run_all()
```

To show the animations:

```{code-cell} ipython3
from IPython.display import Image
import ast

gif_filename = ast.literal_eval(m.plots)["animation"]["filename"]
Image(filename=gif_filename)
```

```{code-cell} ipython3
gif_filename = ast.literal_eval(m.plots)["animation_profile"]["filename"]
Image(filename=gif_filename)
```


### Demo with larval fish scenario

This example plots particle tracks and additionally plots tracks colored by `sea_surface_height`.

```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="LarvalFish", lon=-90, lat=28.7, number=10, duration=pd.Timedelta("3h"),
                       do3D=True, use_static_masks=True,
                       hatched=1,
                       seed_seafloor=True,
                       export_variables=["sea_floor_depth_below_sea_level", 'sea_water_temperature'],
                       plots={'spaghetti': {},
                              'spaghetti2': {'linecolor': 'sea_water_temperature', 'cmap': 'cmo.thermal'},
                              'animation': {},
                              'animation2': {'background': 'sea_floor_depth_below_sea_level',
                                             'cmap': 'cmo.deep'},
                              'animation_profile': {},
                              'animation_profile2': {'markersize_scaling': 80, 'cmap': 'cmo.amp',
                                                     'color': 'weight', 'fps': 4},
                              'property': {'prop': 'z'},
                              'propertymean': {'prop': 'z', 'mean': True},
})


url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
ds = xr.open_dataset(url, decode_times=False)
m.add_reader(ds=ds)

m.run_all()
```

To show the animations:

```{code-cell} ipython3
from IPython.display import Image
import ast

gif_filename = ast.literal_eval(m.plots)["animation"]["filename"]
Image(filename=gif_filename)
```

```{code-cell} ipython3
gif_filename = ast.literal_eval(m.plots)["animation2"]["filename"]
Image(filename=gif_filename)
```

```{code-cell} ipython3
gif_filename = ast.literal_eval(m.plots)["animation_profile"]["filename"]
Image(filename=gif_filename)
```

```{code-cell} ipython3
gif_filename = ast.literal_eval(m.plots)["animation_profile2"]["filename"]
Image(filename=gif_filename)

```



### Demo with oil spill scenario


```{code-cell} ipython3
m = ptm.OpenDriftModel(drift_model="OpenOil", lon=-90, lat=28.7, number=10, duration=pd.Timedelta("3h"),
                       do3D=False, use_static_masks=True, z=0,
                       export_variables=["sea_floor_depth_below_sea_level", "sea_water_salinity"],
                       plots={'spaghetti': {},
                              'spaghetti2': {'linecolor': 'viscosity', 'cmap': 'cmo.speed'},
                              'animation': {},
                              'animation2': {'background': 'sea_floor_depth_below_sea_level',
                                             'cmap': 'cmo.deep'},
                              'animation_profile': {},
                              'animation_profile2': {'markersize_scaling': 80, 'cmap': 'cmo.amp',
                                                     'color': 'mass_oil', 'fps': 4},
                              'property': {'prop': 'sea_water_salinity'},
                              'propertymean': {'prop': 'sea_water_salinity', 'mean': True},
                              'oil': {},
})
m.o.set_config('environment:constant:x_wind', 1)
m.o.set_config('environment:constant:y_wind', -1)


url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
ds = xr.open_dataset(url, decode_times=False)
m.add_reader(ds=ds)

m.run_all()
```

To show the animations:

```{code-cell} ipython3
from IPython.display import Image
import ast

gif_filename = ast.literal_eval(m.plots)["animation"]["filename"]
Image(filename=gif_filename)
```

```{code-cell} ipython3
gif_filename = ast.literal_eval(m.plots)["animation2"]["filename"]
Image(filename=gif_filename)
```

```{code-cell} ipython3
gif_filename = ast.literal_eval(m.plots)["animation_profile"]["filename"]
Image(filename=gif_filename)
```

```{code-cell} ipython3
gif_filename = ast.literal_eval(m.plots)["animation_profile2"]["filename"]
Image(filename=gif_filename)

```
