from pathlib import Path

import pytest

from particle_tracking_manager.config_misc import SetupOutputFiles


@pytest.mark.parametrize("suffix", ["", ".nc", ".parq", ".parquet", ".txt"])
def test_log_name(suffix):
    m = SetupOutputFiles(output_file=f"outputs/newtest{suffix}")
    assert m.logfile_name == str(Path("outputs/newtest.log"))


@pytest.mark.parametrize("format, suffix", [("netcdf", ".nc"), ("parquet", ".parquet")])
def test_output_file(format, suffix):
    """Make sure output file suffix matches the chosen format"""

    m = SetupOutputFiles(output_format=format)
    assert m.output_file.suffix == suffix
