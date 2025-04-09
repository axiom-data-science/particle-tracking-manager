"""Command line interface to get inputs from web application."""

# Standard library imports
import argparse
import ast

from datetime import datetime

# Third-party imports
import pandas as pd

# Local imports
import particle_tracking_manager as ptm


def is_int(s: str) -> bool:
    """Check if string is actually int."""
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def is_float(s: str) -> bool:
    """Check if string is actually float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def is_None(s: str) -> bool:
    """Check if string is actually None."""
    if s == "None":
        return True
    else:
        return False


def is_datestr(s: str) -> bool:
    """Check if string is actually a datestring."""

    try:
        out = pd.Timestamp(s)
        assert not pd.isnull(out)
        return True
    except (ValueError, AssertionError, TypeError):
        return False


def is_deltastr(s: str) -> bool:
    """Check if string is actually a Timedelta."""

    try:
        out = pd.Timedelta(s)
        assert not pd.isnull(out)
        return True
    except (ValueError, AssertionError):
        return False


# https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    """With can user can input dicts on CLI."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        """With can user can input dicts on CLI."""
        setattr(namespace, self.dest, dict())
        for value in values:
            # maxsplit helps in case righthand side of input has = in it, like filenames can have
            key, value = value.split("=", maxsplit=1)
            # catch list case
            if value.startswith("[") and value.endswith("]"):
                # if "[" in value and "]" in value:
                value = value.strip("][").split(",")
            # change numbers to numbers but with attention to decimals and negative numbers
            if is_int(value):
                value = int(value)
            elif is_float(value):
                value = float(value)
            elif is_None(value):
                value = None
            elif is_deltastr(value):
                value = pd.Timedelta(value).isoformat()
            elif is_datestr(value):
                value = pd.Timestamp(value)
            getattr(namespace, self.dest)[key] = value


def main():
    """Parser method.

    Include all inputs

    Example
    -------

    >>> python cli.py lon=-151 lat=59 use_auto_landmask=True start_time=2000-1-1
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "kwargs",
        nargs="*",
        action=ParseKwargs,
        help="Input keyword arguments for running PTM. Available options are specific to the `catalog_type`. Dictionary-style input, e.g. `case='Oil'`. Format for list items is e.g. standard_names='[sea_water_practical_salinity,sea_water_temperature]'.",
    )

    parser.add_argument(
        "--dry-run",
        help="Return configuration parameters without running the model.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    to_bool = {
        key: ast.literal_eval(value)
        for key, value in args.kwargs.items()
        if value in ["True", "False"]
    }
    args.kwargs.update(to_bool)

    if "output_file" not in args.kwargs:

        args.kwargs[
            "output_file"
        ] = f"output-results_{datetime.utcnow():%Y-%m-%dT%H%M:%SZ}.nc"

    # this is for running plots at the same time as a simulation
    # Convert the string representation of the dictionary to an actual dictionary
    if "plots" in args.kwargs:
        args.kwargs["plots"] = ast.literal_eval(args.kwargs["plots"])
    else:
        args.kwargs["plots"] = None

    # this is for running plots for an existing simulation file.
    # catch plots first
    if (args.kwargs["plots"] is not None) and not (
        set(args.kwargs) - set(["plots", "output_file"])
    ):

        import particle_tracking_manager.models.opendrift.plot as plot

        plots_dict = plot.make_plots_after_simulation(
            args.kwargs["output_file"], plots=args.kwargs["plots"]
        )
        print(plots_dict)
        return

    m = ptm.OpenDriftModel(**args.kwargs)  # , plots=plots)

    if args.dry_run:

        # run this to make sure everything is updated fully
        m.add_reader()
        print(m.config.model_dump())

    else:

        m.add_reader()

        m.seed()
        m.run()

        print(m.files.output_file)


if __name__ == "__main__":
    main()
