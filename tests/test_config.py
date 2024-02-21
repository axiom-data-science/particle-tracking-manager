"""Test configuration details."""

import particle_tracking_manager as ptm


def test_ptm_config():
    """Test that PTM config is brought in

    ...and takes precedence over model config.
    """

    m = ptm.OpenDriftModel()

    # check for a single key and make sure override default is present
    assert m.show_config()["coastline_action"]["default"] == "previous"


def test_show_config():
    """Test configuration-showing functionality."""

    m = ptm.OpenDriftModel()

    # check sorting by a single level
    assert sorted(m.show_config(level=1).keys()) == [
        "seed:seafloor",
        "seed:z",
        "seed_seafloor",
        "z",
    ]

    # check PTM level sorting
    assert (
        "lon" in m.show_config(ptm_level=1).keys()
        and "log" not in m.show_config(ptm_level=1).keys()
    )


def test_surface_only():
    """Make sure appropriate parameters are set if surface_only is True."""

    m = ptm.OpenDriftModel(surface_only=True, z=-10)


def test_default_overrides():
    """Make sure input to OpenDriftModel and to PTM are represented as values in config."""

    m = ptm.OpenDriftModel(emulsification=False, drift_model="OpenOil", steps=5)

    # the value should be False due to input value
    # check without running update_config which since want to know initial state
    assert not m.show_config_model("processes:emulsification")["value"]
    assert (
        m._config["steps"]["value"]
        == m.config_ptm["steps"]["value"]
        == m.show_config_model("steps")["value"]
        == 5
    )
