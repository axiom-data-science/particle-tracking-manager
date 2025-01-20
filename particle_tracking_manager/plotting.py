"""Some extra plotting code."""

import opendrift


def plot_dest(o, filename=None, **kwargs):
    """This is copied from an opendrift example.

    ...and subsequently modified.

    Parameters
    ----------
    kwargs : dict
        Additional keyword arguments to pass to the plotting function.
    """

    import cmocean

    cmap = cmocean.tools.crop_by_percent(cmocean.cm.amp, 20, which="max", N=None)

    od = opendrift.open_xarray(o.outfile_name)

    density = od.get_histogram(pixelsize_m=5000).isel(time=-1).isel(origin_marker=0)
    density = density.where(density > 0)
    density = density / density.sum() * 100
    vmax = density.max()

    return o.plot(
        background=density,
        clabel="Probability of final location [%]",
        markersize=0.5,
        lalpha=0.02,
        vmin=0,
        vmax=vmax,
        cmap=cmap,
        fast=True,
        filename=filename,
        **kwargs
    )
