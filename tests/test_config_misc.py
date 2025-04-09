from particle_tracking_manager.config_misc import SetupOutputFiles


def test_log_name():
    m = SetupOutputFiles(output_file="newtest")
    assert m.logfile_name == "newtest.log"

    m = SetupOutputFiles(output_file="newtest.nc")
    assert m.logfile_name == "newtest.log"

    m = SetupOutputFiles(output_file="newtest.parq")
    assert m.logfile_name == "newtest.log"

    m = SetupOutputFiles(output_file="newtest.parquet")
    assert m.logfile_name == "newtest.log"


def test_output_file():
    """make sure output file is parquet if output_format is parquet"""

    m = SetupOutputFiles(output_format="parquet")
    assert m.output_file.suffix == ".parquet"

    m = SetupOutputFiles(output_format="netcdf")
    assert m.output_file.suffix == ".nc"
