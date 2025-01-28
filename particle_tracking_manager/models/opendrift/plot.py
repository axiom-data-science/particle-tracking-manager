import logging

import pandas as pd


model = "opendrift"
logger = logging.getLogger(model)


def check_plots(which_plots, export_variables, drift_model):
    """Check that input plot options are valid,

    particularly that the necessary export variables are included.

    Parameters
    ----------
    which_plots : dict
        Dictionary of plot options.
    export_variables : list
        Variables to be saved to file.
    drift_model : str
        Which OpenDrift model is being used.

    Raises
    ------
    ValueError
        If "all" is specified with other plot options.
    ValueError
        If necessary export variables are missing.
    ValueError
        If oil budget plot is requested with a drift model other than OpenOil.
    """

    if (
        len([k for k in which_plots.keys() if "oil" in k]) > 0
        and drift_model != "OpenOil"
    ):
        raise ValueError("Oil budget plot only available for OpenOil drift model")

    # if "all", make sure it is the only plot option
    # since it will create its own dictionary of plot options
    if "all" in which_plots and len(which_plots) > 1:
        raise ValueError(
            "If 'all' is specified for plots, it must be the only plot option."
        )

    # flatten dict
    which_plots_flat = pd.json_normalize(which_plots).to_dict(orient="records")

    if len(which_plots_flat) == 0:
        return

    which_plots_flat = which_plots_flat[0]

    # check for cases that require an export variable
    plot_options_to_check = ["linecolor", "color", "background", "prop", "markersize"]
    missing_variables = []
    for plot_option in plot_options_to_check:
        variables = [
            v
            for k, v in which_plots_flat.items()
            if plot_option == k.split(".")[1] and v not in export_variables
        ]
        missing_variables.extend(variables)

    # handle oil budget separately
    vars_needed = ["x_wind", "y_wind", "x_sea_water_velocity", "y_sea_water_velocity"]
    variables = [
        list(set(vars_needed) - set(export_variables))
        for k, v in which_plots_flat.items()
        if "show_wind_and_current" in k
        and v
        and (len(set(vars_needed) - set(export_variables)) > 0)
    ]
    if len(variables) > 0:
        variables = variables[0]
        missing_variables.extend(variables)

    if len(missing_variables) > 0:
        raise ValueError(
            f"Missing export variables for the following plot options: {list(set(missing_variables))}"
        )


def make_filename_string(plot_name, filename, kwargs):
    """Create a filename string based on the kwargs."""

    # modify filename with plot kwargs
    if len(kwargs) > 0:
        filename += "_" + "_".join(
            [
                f"{key}_{value}"
                for key, value in kwargs.items()
                if key not in ["filetype"]
            ]
        )

    # include filetype suffix
    if "spaghetti" in plot_name or "oil" in plot_name or "property" in plot_name:
        if "filetype" not in kwargs:
            filename += ".png"
        else:
            filename += "." + kwargs["filetype"]
            kwargs.pop("filetype")

    elif "animation" in plot_name:
        if "filetype" not in kwargs:
            filename += ".gif"
        else:
            filename += "." + kwargs["filetype"]
            kwargs.pop("filetype")

    else:
        raise ValueError(f"Invalid plot name: {plot_name}")

    return filename


def plot(plot_name, input_kwargs, o, filename, drift_model):
    """Create a plot based on the plot name and input kwargs.

    Parameters
    ----------
    plot_name : str
        Must contain key word to identify the plot type.
    input_kwargs : dict
        Optional plot kwargs.
    o : _type_
        OpenDrift simulation object.
    filename : str
        Filename substring to save the plot.
    drift_model : str
        Which OpenDrift model is being used.
    """

    if "spaghetti" in plot_name or plot_name == "all":

        filename += "_spaghetti"

        # add input plot input_kwargs to the default kwargs
        kwargs = {"fast": True}
        kwargs.update(input_kwargs)

        # modify filename with plot kwargs
        filename = make_filename_string(plot_name, filename, kwargs)

        o.plot(filename=filename, **kwargs)

        logger.info(f"Saved spaghetti plot to {filename}")

    elif (
        "animation" in plot_name and "profile" not in plot_name
    ) or plot_name == "all":

        # add input plot kwargs to the default kwargs
        kwargs = {"fast": True, "fps": 4}
        kwargs.update(input_kwargs)

        # modify filename with plot kwargs
        filename = make_filename_string(plot_name, filename, kwargs)

        o.animation(filename=filename, **kwargs)

    elif ("animation" in plot_name and "profile" in plot_name) or plot_name == "all":

        filename += "_profile"

        # add input plot kwargs to the default kwargs
        kwargs = {"fps": 4}
        kwargs.update(input_kwargs)

        # modify filename with plot kwargs
        filename = make_filename_string(plot_name, filename, kwargs)

        o.animation_profile(filename=filename, **kwargs)

    elif "oil" in plot_name or plot_name == "all":

        filename += "_oil"
        plot_name = "oil"

        # add input plot kwargs to the default kwargs
        kwargs = {}
        kwargs = {
            "show_wind_and_current": False,
            "show_watercontent_and_viscosity": True,
        }
        kwargs.update(input_kwargs)

        # modify filename with plot kwargs
        filename = make_filename_string(plot_name, filename, kwargs)

        if drift_model == "OpenOil":
            o.plot_oil_budget(filename=filename, **kwargs)
        else:
            raise ValueError("Oil budget plot only available for OpenOil drift model")

    elif "property" in plot_name:  # or plot_name == "all":

        if not "prop" in input_kwargs:
            raise ValueError(
                "Property plot must be specified as 'prop' in the plots dictionary."
            )

        filename += "_property"
        plot_name = "property"

        # add input plot kwargs to the default kwargs
        kwargs = {}
        kwargs.update(input_kwargs)

        # modify filename with plot kwargs
        filename = make_filename_string(plot_name, filename, kwargs)

        o.plot_property(filename=filename, **kwargs)

    return filename


def make_plots(which_plots, o, filename, drift_model):
    """Run through each plot key and make the plot.

    This extra level of abstraction is necessary to allow for multiple
    types of each plot to be made, for example, a spaghetti plot and a
    spaghetti plot with tracks colored.
    """

    if "all" in which_plots:
        which_plots = {
            "spaghetti": {},
            "animation": {},
            "animation_profile": {},
        }
        if drift_model == "OpenOil":
            which_plots["oil"] = {}

    for which_plot in which_plots:
        filename_out = plot(
            which_plot, which_plots[which_plot], o, filename, drift_model
        )
        # save filename for plot
        which_plots[which_plot]["filename"] = filename_out
    return which_plots
