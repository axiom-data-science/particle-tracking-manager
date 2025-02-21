"""General utilities."""


def calc_known_horizontal_diffusivity(ocean_model):
    """Calculate horizontal diffusivity based on known ocean_model."""

    # dx: approximate horizontal grid resolution (meters), used to calculate horizontal diffusivity
    if ocean_model == "NWGOA":
        dx = 1500
    elif "CIOFS" in ocean_model:
        dx = 100

    # horizontal diffusivity is calculated based on the mean horizontal grid resolution
    # for the model being used.
    # 0.1 is a guess for the magnitude of velocity being missed in the models, the sub-gridscale velocity
    sub_gridscale_velocity = 0.1
    horizontal_diffusivity = sub_gridscale_velocity * dx
    return horizontal_diffusivity
