# following https://fsspec.github.io/kerchunk/test_example.html#
# env xp-inventory on lakes02
# also installed ipython, h5py, cf_xarray, pynco
# copied to /mnt/vault/ciofs/HINDCAST/ciofs_kerchunk_to2012.json

# import kerchunk.hdf
import fsspec
from pathlib import Path
from kerchunk.combine import MultiZarrToZarr
# from kerchunk.df import refs_to_dataframe
# import ujson
# import numpy as np

def make_ciofs_kerchunk(start, end):
    """_summary_

    Parameters
    ----------
    start, end : str
        Should be something like "2004_0001" for YYYY_0DDD where DDD is dayofyear
        to match the files in the directory, which are by year and day of year.

    Returns
    -------
    kerchunk output
        _description_
    """

    output_dir_single_files = "/home/kristen/projects/kerchunk_setup/ciofs"

    fs2 = fsspec.filesystem('')  # local file system to save final jsons to

    # select the single file Jsons to combine
    json_list = sorted(fs2.glob(f"{output_dir_single_files}/*.json"))  # combine single json files
    json_list = [j for j in json_list if Path(j).stem>=start and Path(j).stem<=end]

    # Multi-file JSONs
    # This code uses the output generated above to create a single ensemble dataset, 
    # with one set of references pointing to all of the chunks in the individual files.
    # `coo_map = {"ocean_time": "cf:ocean_time"}` is necessary so that both the time
    # values and units are read and interpreted instead of just the values.

    def fix_fill_values(out):
        """Fix problem when fill_value and scara both equal 0.0.
        
        If the fill value and the scalar value are both 0, nan is filled instead. This fixes that.
        """

        for k in list(out):
            if isinstance(out[k], str) and '"fill_value":0.0' in out[k]:
                out[k] = out[k].replace('"fill_value":0.0', '"fill_value":"NaN"')
        return out

    def postprocess(out):
        out = fix_fill_values(out)
        return out


    mzz = MultiZarrToZarr(
        json_list,
        concat_dims=["ocean_time"],
        identical_dims= ['lat_rho', 'lon_rho', "lon_psi", "lat_psi",
                        "lat_u", "lon_u", "lat_v", "lon_v", 
                        "Akk_bak","Akp_bak","Akt_bak","Akv_bak","Cs_r","Cs_w",
                        "FSobc_in","FSobc_out","Falpha","Fbeta","Fgamma","Lm2CLM",
                        "Lm3CLM", "LnudgeM2CLM", "LnudgeM3CLM", "LnudgeTCLM",
                        "LsshCLM", "LtracerCLM", "LtracerSrc", "LuvSrc",
                        "LwSrc", "M2nudg", "M2obc_in", "M2obc_out", "M3nudg",
                        "M3obc_in", "M3obc_out", "Tcline", "Tnudg","Tobc_in", "Tobc_out",
                        "Vstretching", "Vtransform", "Znudg", "Zob", "Zos", "angle",
                        "dstart", "dt", "dtfast", "el", "f", "gamma2", "grid", "h",
                        "hc", "mask_psi", "mask_rho", "mask_u", "mask_v", "nHIS", "nRST",
                        "nSTA", "ndefHIS",  "ndtfast", "ntimes", "pm", "pn", "rdrg", 
                        "rdrg2", "rho0", "spherical", "theta_b", "theta_s", "xl",
                        ],
        coo_map = {"ocean_time": "cf:ocean_time",
                },
        postprocess=postprocess,
    )


    # to keep in memory
    out = mzz.translate()

    return out


def make_nwgoa_kerchunk(start, end):
    """_summary_

    Parameters
    ----------
    start, end : str
        Should be something like "1999-01-02" for YYYY-MM-DD 

    Returns
    -------
    kerchunk output
        _description_
    """

    # this version of the daily json files has the grid file merged
    output_dir_single_files = "/home/kristen/projects/kerchunk_setup/nwgoa_with_grids"

    fs2 = fsspec.filesystem('')  # local file system to save final jsons to

    # select the single file Jsons to combine
    json_list = sorted(fs2.glob(f"{output_dir_single_files}/nwgoa*.json"))  # combine single json files
    json_list = [j for j in json_list if Path(j).stem.split("nwgoa_")[1]>=start and Path(j).stem.split("nwgoa_")[1]<=end]

    # Merge one of the model output files and the grid file as shown here, then
    # saved the merged file, then used that in the MultiZarr below to get this to work
    # https://fsspec.github.io/kerchunk/tutorial.html#merging-variables-across-jsons
    # json_list += fs2.glob(f"{output_dir_single_files}/Cook_Inlet_grid_1.json") # combine single json files



    # include grid file 
    # grid_url_classic = Path("/mnt/depot/data/packrat/prod/aoos/nwgoa/Cook_Inlet_grid_1.nc")
    # outf = out_base / Path(grid_url_classic.stem).with_suffix(".json")
    # if not outf.exists():
    #     refs_static = NetCDF3ToZarr(str(grid_url_classic))
    #     with fs2.open(outf, 'wb') as f:
    #         f.write(ujson.dumps(refs_static.translate()).encode());


    # # I merged one of the model output files and the grid file as shown here, then
    # # saved the merged file, then used that in the MultiZarr below to get this to work
    # # https://fsspec.github.io/kerchunk/tutorial.html#merging-variables-across-jsons
    # json_list = fs2.glob(f"{output_dir_single_files}/nwgoa_1999-01-02.json") # combine single json files
    # json_list += fs2.glob(f"{output_dir_single_files}/Cook_Inlet_grid_1.json") # combine single json files

    # d = merge_vars(json_list)  # returns dict, as does translate()


# # write merged version
# outf = out_base / pathlib.Path("nwgoa_1999-01-02.json")
# if not outf.exists():
#     # refs_static = NetCDF3ToZarr(str(grid_url_classic))
#     with fs2.open(outf, 'wb') as f:
#         f.write(ujson.dumps(d).encode());



    # Multi-file JSONs
    # This code uses the output generated above to create a single ensemble dataset, 
    # with one set of references pointing to all of the chunks in the individual files.
    # `coo_map = {"ocean_time": "cf:ocean_time"}` is necessary so that both the time
    # values and units are read and interpreted instead of just the values.


    # account for double compression 
    # Look at individual variables in the files to see what needs to be changed with
    # h5dump -d ocean_time -p /mnt/depot/data/packrat/prod/aoos/nwgoa/processed/1999/nwgoa_1999-02-01.nc
    def preprocess(refs):
        for k in list(refs):
            if k.endswith('/.zarray'):
                refs[k] = refs[k].replace('"filters":[{"elementsize":8,"id":"shuffle"}]',
                                        '"filters":[{"elementsize":8,"id":"shuffle"},{"id": "zlib", "level":8}]')
                refs[k] = refs[k].replace('"filters":[{"elementsize":4,"id":"shuffle"}]',
                                        '"filters":[{"elementsize":4,"id":"shuffle"},{"id": "zlib", "level":8}]')
        return refs

    import zarr
    def add_time_attr(out):
        out_ = zarr.open(out)
        out_.ocean_time.attrs["axis"] = "T"
        return out

    def postprocess(out):
        out = add_time_attr(out)
        return out


    mzz = MultiZarrToZarr(
        json_list,
        concat_dims=["ocean_time"],
        identical_dims= ['lat_rho', 'lon_rho', "lon_psi", "lat_psi",
                        "lat_u", "lon_u", "lat_v", "lon_v", 
                        "Akk_bak","Akp_bak","Akt_bak","Akv_bak","Cs_r","Cs_w",
                        "FSobc_in","FSobc_out","Falpha","Fbeta","Fgamma","Lm2CLM",
                        "Lm3CLM", "LnudgeM2CLM", "LnudgeM3CLM", "LnudgeTCLM",
                        "LsshCLM", "LtracerCLM", "LtracerSrc", "LuvSrc",
                        "LwSrc", "M2nudg", "M2obc_in", "M2obc_out", "M3nudg",
                        "M3obc_in", "M3obc_out", "Tcline", "Tnudg","Tobc_in", "Tobc_out",
                        "Vstretching", "Vtransform", "Znudg", "Zob", "Zos", "angle",
                        "dstart", "dt", "dtfast", "el", "f", "gamma2", "grid", "h",
                        "hc", "mask_psi", "mask_rho", "mask_u", "mask_v", "nHIS", "nRST",
                        "nSTA", "ndefHIS",  "ndtfast", "ntimes", "pm", "pn", "rdrg", 
                        "rdrg2", "rho0", "spherical", "theta_b", "theta_s", "xl",
                        "Charnok_alpha", "CrgBan_cw", "JLTS", "JPRJ", "LuvSponge", "P1", "P2", "P3", "P4", 
    "PLAT", "PLONG", "ROTA", "XOFF", "YOFF", "Zos_hsig_alpha", "depthmax", "depthmin",
    "dfdy", "dmde", "dndx", "f0", "gls_Kmin", "gls_Pmin", "gls_c1", "gls_c2", 
    "gls_c3m",  "gls_c3p", "gls_cmu0", "gls_m", "gls_n", "gls_p", "gls_sigk", "gls_sigp",
    "h_mask", "hraw", "nAVG", "ndefAVG", "nl_visc2", "ntsAVG", "sz_alpha", "wtype_grid",
    "x_psi", "x_rho", "x_u", "x_v", "y_psi", "y_rho", "y_u", "y_v",
                        ],
        coo_map = {"ocean_time": "cf:ocean_time",
                #    "eta_rho": list(np.arange(1044))
                },
        preprocess=preprocess,
        postprocess=postprocess,
    )

    # to keep in memory
    out = mzz.translate()
    # import pdb; pdb.set_trace()
    # merge_vars([out, fs2.glob(f"{output_dir_single_files}/Cook_Inlet_grid_1.json")])
    # d = merge_vars(json_list)  # returns dict, as does translate()

    return out
