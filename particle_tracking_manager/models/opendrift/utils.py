# following https://fsspec.github.io/kerchunk/test_example.html#
# env xp-inventory on lakes02
# also installed ipython, h5py, cf_xarray, pynco
# copied to /mnt/vault/ciofs/HINDCAST/ciofs_kerchunk_to2012.json

# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List

# Third-party imports
import fsspec
import pandas as pd
import xarray as xr

from kerchunk.combine import MultiZarrToZarr


def narrow_dataset_to_simulation_time(
    ds: xr.Dataset, start_time: datetime, end_time: datetime
) -> xr.Dataset:
    """Narrow the dataset to the simulation time."""
    try:
        units = ds.ocean_time.attrs["units"]
    except KeyError:
        units = ds.ocean_time.encoding["units"]
    datestr = units.split("since ")[1]
    units_date = pd.Timestamp(datestr)

    # narrow model output to simulation time if possible before sending to Reader
    if start_time is not None and end_time is not None:
        dt_model = float(
            ds.ocean_time[1] - ds.ocean_time[0]
        )  # time step of the model output in seconds
        # want to include the next ocean model output before the first drifter simulation time
        # in case it starts before model times
        start_time_num = (start_time - units_date).total_seconds()
        if start_time_num > dt_model:
            start_time_num -= dt_model
        # want to include the next ocean model output after the last drifter simulation time
        end_time_num = (end_time - units_date).total_seconds() + dt_model
        ds = ds.sel(ocean_time=slice(start_time_num, end_time_num))

        if len(ds.ocean_time) == 0:
            raise ValueError(
                "No model output left for simulation time. Check start_time and end_time."
            )
        if len(ds.ocean_time) == 1:
            raise ValueError(
                "Only 1 model output left for simulation time. Check start_time and end_time."
            )
    else:
        raise ValueError(
            "start_time and end_time must be set to narrow model output to simulation time"
        )
    return ds


def apply_known_ocean_model_specific_changes(
    ds: xr.Dataset, ocean_model: str, use_static_masks: bool
) -> xr.Dataset:
    """Apply ocean model specific changes to the dataset.

    This includes renaming variables, adding variables, etc.
    """
    # For NWGOA, need to calculate wetdry mask from a variable
    if ocean_model == "NWGOA" and not use_static_masks:
        ds["wetdry_mask_rho"] = (~ds.zeta.isnull()).astype(int)

    # For CIOFSOP need to rename u/v to have "East" and "North" in the variable names
    # so they aren't rotated in the ROMS reader (the standard names have to be x/y not east/north)
    elif ocean_model == "CIOFSOP":
        ds = ds.rename_vars({"urot": "u_eastward", "vrot": "v_northward"})
    return ds


def apply_user_input_ocean_model_specific_changes(
    ds: xr.Dataset, use_static_masks: bool
) -> xr.Dataset:
    """Apply user input ocean model specific changes to the dataset.

    This includes renaming variables, adding variables, etc.

    For now, assume user has dropped variables ahead of time.
    """

    # check for case that self.config.use_static_masks False (which is the default)
    # but user input doesn't have wetdry masks
    # then raise exception and tell user to set use_static_masks True
    if "wetdry_mask_rho" not in ds.data_vars and not use_static_masks:
        raise ValueError(
            "User input does not have wetdry_mask_rho variable. Set use_static_masks True to use static masks instead."
        )

    # ds = ds.drop_vars(self.config.drop_vars, errors="ignore")

    return ds


def find_json_files_in_date_range(
    fs2,
    make_glob_from_year: Callable[[str], str],
    start: datetime,
    end: datetime,
    filename_date_format: str,
) -> List[str]:
    """Find JSON files in a date range and return their paths."""

    # only glob start and end year files, order isn't important
    if abs(start.year - end.year) > 1:
        raise ValueError(
            f"Start ({start.year}) and end ({end.year}) "
            "dates must be at most 1 year apart."
        )
    start_year = start.strftime("%Y")
    end_year = end.strftime("%Y")
    json_list = fs2.glob(make_glob_from_year(start_year))
    if end_year != start_year:
        json_list += fs2.glob(make_glob_from_year(end_year))

    # if going backward in time, swap start and end
    date_range_from, date_range_to = (start, end) if start <= end else (end, start)

    return [
        pth
        for pth in json_list
        if (
            date_range_from
            <= datetime.strptime(Path(pth).stem, filename_date_format)
            <= date_range_to
        )
    ]


def make_ciofs_kerchunk(start: datetime, end: datetime, name: str) -> dict:
    """_summary_

    Parameters
    ----------
    start, end : datetime
        Start and end time of the simulation.

    Returns
    -------
    kerchunk output
        _description_
    """

    if name == "CIOFS":
        output_dir_single_files = "/mnt/vault/ciofs/HINDCAST/.kerchunk_json"
    elif name == "CIOFS3":
        output_dir_single_files = "/mnt/vault/ciofs/CIOFS3/HINDCAST/.kerchunk_json"
    elif name == "CIOFSFRESH":
        output_dir_single_files = "/mnt/vault/ciofs/HINDCAST_FRESHWATER/.kerchunk_json"
    elif name == "CIOFSOP":
        output_dir_single_files = "/mnt/depot/data/packrat/prod/noaa/coops/ofs/aws_ciofs/processed/.kerchunk_json"
    else:
        raise ValueError(f"Name {name} not recognized")

    fs2 = fsspec.filesystem("")  # local file system to save final jsons to

    if name == "CIOFSOP":
        # base for matching
        def base_str(a_time: str) -> str:
            return f"{output_dir_single_files}/ciofs_{a_time}*.json"

        date_format = "ciofs_%Y-%m-%d"

    else:  # name is "CIOFS" or "CIOFSFRESH" or "CIOFS3"
        # base for matching
        def base_str(a_time: str) -> str:
            return f"{output_dir_single_files}/{a_time}_*.json"

        date_format = "%Y_0%j"

    json_list = find_json_files_in_date_range(fs2, base_str, start, end, date_format)

    # Multi-file JSONs
    # This code uses the output generated above to create a single ensemble dataset,
    # with one set of references pointing to all of the chunks in the individual files.
    # `coo_map = {"ocean_time": "cf:ocean_time"}` is necessary so that both the time
    # values and units are read and interpreted instead of just the values.

    def fix_fill_values(out: dict) -> dict:
        """Fix problem when fill_value and scara both equal 0.0.

        If the fill value and the scalar value are both 0, nan is filled instead. This fixes that.
        """

        for k in list(out):
            if isinstance(out[k], str) and '"fill_value":0.0' in out[k]:
                out[k] = out[k].replace('"fill_value":0.0', '"fill_value":"NaN"')
        return out

    def postprocess(out: dict) -> dict:
        """postprocess function to fix fill values"""
        out = fix_fill_values(out)
        return out

    mzz = MultiZarrToZarr(
        json_list,
        concat_dims=["ocean_time"],
        identical_dims=[
            "lat_rho",
            "lon_rho",
            "lon_psi",
            "lat_psi",
            "lat_u",
            "lon_u",
            "lat_v",
            "lon_v",
            "Akk_bak",
            "Akp_bak",
            "Akt_bak",
            "Akv_bak",
            "Cs_r",
            "Cs_w",
            "FSobc_in",
            "FSobc_out",
            "Falpha",
            "Fbeta",
            "Fgamma",
            "Lm2CLM",
            "Lm3CLM",
            "LnudgeM2CLM",
            "LnudgeM3CLM",
            "LnudgeTCLM",
            "LsshCLM",
            "LtracerCLM",
            "LtracerSrc",
            "LuvSrc",
            "LwSrc",
            "M2nudg",
            "M2obc_in",
            "M2obc_out",
            "M3nudg",
            "M3obc_in",
            "M3obc_out",
            "Tcline",
            "Tnudg",
            "Tobc_in",
            "Tobc_out",
            "Vstretching",
            "Vtransform",
            "Znudg",
            "Zob",
            "Zos",
            "angle",
            "dstart",
            "dt",
            "dtfast",
            "el",
            "f",
            "gamma2",
            "grid",
            "h",
            "hc",
            "mask_psi",
            "mask_rho",
            "mask_u",
            "mask_v",
            "nHIS",
            "nRST",
            "nSTA",
            "ndefHIS",
            "ndtfast",
            "ntimes",
            "pm",
            "pn",
            "rdrg",
            "rdrg2",
            "rho0",
            "spherical",
            "theta_b",
            "theta_s",
            "xl",
        ],
        coo_map={
            "ocean_time": "cf:ocean_time",
        },
        postprocess=postprocess,
    )

    # to keep in memory
    out = mzz.translate()

    return out


def make_nwgoa_kerchunk(start: datetime, end: datetime, name: str = "NWGOA") -> dict:
    """_summary_

    Parameters
    ----------
    start, end : datetime
        Start and end time of the simulation.

    Returns
    -------
    kerchunk output
        _description_
    """

    # this version of the daily json files has the grid file merged
    output_dir_single_files = (
        "/mnt/depot/data/packrat/prod/aoos/nwgoa/processed/.kerchunk_json"
    )

    fs2 = fsspec.filesystem("")  # local file system to save final jsons to

    # base for matching
    def base_str(a_time: str) -> str:
        # this is the base string for the json files
        return f"{output_dir_single_files}/nwgoa_{a_time}-*.json"

    date_format = "nwgoa_%Y-%m-%d"

    json_list = find_json_files_in_date_range(fs2, base_str, start, end, date_format)

    # account for double compression
    # Look at individual variables in the files to see what needs to be changed with
    # h5dump -d ocean_time -p /mnt/depot/data/packrat/prod/aoos/nwgoa/processed/1999/nwgoa_1999-02-01.nc
    def preprocess(refs: dict) -> dict:
        """preprocess function to fix fill values"""
        for k in list(refs):
            if k.endswith("/.zarray"):
                refs[k] = refs[k].replace(
                    '"filters":[{"elementsize":8,"id":"shuffle"}]',
                    '"filters":[{"elementsize":8,"id":"shuffle"},{"id": "zlib", "level":8}]',
                )
                refs[k] = refs[k].replace(
                    '"filters":[{"elementsize":4,"id":"shuffle"}]',
                    '"filters":[{"elementsize":4,"id":"shuffle"},{"id": "zlib", "level":8}]',
                )
        return refs

    import zarr

    def add_time_attr(out: dict) -> dict:
        """add time attributes to the ocean_time variable"""
        out_ = zarr.open(out)
        out_.ocean_time.attrs["axis"] = "T"
        return out

    def postprocess(out: dict) -> dict:
        """postprocess function to fix fill values"""
        out = add_time_attr(out)
        return out

    mzz = MultiZarrToZarr(
        json_list,
        concat_dims=["ocean_time"],
        identical_dims=[
            "lat_rho",
            "lon_rho",
            "lon_psi",
            "lat_psi",
            "lat_u",
            "lon_u",
            "lat_v",
            "lon_v",
            "Akk_bak",
            "Akp_bak",
            "Akt_bak",
            "Akv_bak",
            "Cs_r",
            "Cs_w",
            "FSobc_in",
            "FSobc_out",
            "Falpha",
            "Fbeta",
            "Fgamma",
            "Lm2CLM",
            "Lm3CLM",
            "LnudgeM2CLM",
            "LnudgeM3CLM",
            "LnudgeTCLM",
            "LsshCLM",
            "LtracerCLM",
            "LtracerSrc",
            "LuvSrc",
            "LwSrc",
            "M2nudg",
            "M2obc_in",
            "M2obc_out",
            "M3nudg",
            "M3obc_in",
            "M3obc_out",
            "Tcline",
            "Tnudg",
            "Tobc_in",
            "Tobc_out",
            "Vstretching",
            "Vtransform",
            "Znudg",
            "Zob",
            "Zos",
            "angle",
            "dstart",
            "dt",
            "dtfast",
            "el",
            "f",
            "gamma2",
            "grid",
            "h",
            "hc",
            "mask_psi",
            "mask_rho",
            "mask_u",
            "mask_v",
            "nHIS",
            "nRST",
            "nSTA",
            "ndefHIS",
            "ndtfast",
            "ntimes",
            "pm",
            "pn",
            "rdrg",
            "rdrg2",
            "rho0",
            "spherical",
            "theta_b",
            "theta_s",
            "xl",
            "Charnok_alpha",
            "CrgBan_cw",
            "JLTS",
            "JPRJ",
            "LuvSponge",
            "P1",
            "P2",
            "P3",
            "P4",
            "PLAT",
            "PLONG",
            "ROTA",
            "XOFF",
            "YOFF",
            "Zos_hsig_alpha",
            "depthmax",
            "depthmin",
            "dfdy",
            "dmde",
            "dndx",
            "f0",
            "gls_Kmin",
            "gls_Pmin",
            "gls_c1",
            "gls_c2",
            "gls_c3m",
            "gls_c3p",
            "gls_cmu0",
            "gls_m",
            "gls_n",
            "gls_p",
            "gls_sigk",
            "gls_sigp",
            "h_mask",
            "hraw",
            "nAVG",
            "ndefAVG",
            "nl_visc2",
            "ntsAVG",
            "sz_alpha",
            "wtype_grid",
            "x_psi",
            "x_rho",
            "x_u",
            "x_v",
            "y_psi",
            "y_rho",
            "y_u",
            "y_v",
        ],
        coo_map={
            "ocean_time": "cf:ocean_time",
            #    "eta_rho": list(np.arange(1044))
        },
        preprocess=preprocess,
        postprocess=postprocess,
    )

    # to keep in memory
    out = mzz.translate()

    return out
