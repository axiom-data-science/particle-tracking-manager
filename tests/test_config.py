"""Test configuration details."""

import particle_tracking_manager as ptm


def test_ptm_config():
    """Test that PTM config is brought in 
    
    ...and takes precendence over model config.
    """
    
    m = ptm.OpenDriftModel()
    
    # check for a single key and make sure override default is present
    assert m.show_config()["coastline_action"]["default"] == "previous"
    

def test_show_config():
    """Test configuration-showing functionality."""

    m = ptm.OpenDriftModel()

    # check sorting by a single level
    assert sorted(m.show_config(level=1).keys()) == ['seed:seafloor', 'seed:z', 'seed_seafloor', 'z']
    
    # check PTM level sorting
    assert "lon" in m.show_config(ptm_level=1).keys() and "log" not in m.show_config(ptm_level=1).keys()


def test_default_overrides():
    """Make sure input to OpenDriftModel and to PTM are represented as values in config."""
    
    m = ptm.OpenDriftModel(emulsification=False, driftmodel="OpenOil", steps=5)
    
    # the value should be False due to input value
    assert not m.show_config(key="processes:emulsification")["value"]
    
    # value should be what was entered
    assert m.show_config(key="steps")["value"] == 5


def test_horizontal_diffusivity_logic():
    """Check logic for using default horizontal diff values for known models."""
    
    m = ptm.OpenDriftModel()
    m.ocean_model = "NWGOA"
    assert m.horizontal_diffusivity == 150.0
    m.ocean_model = "CIOFS"
    assert m.horizontal_diffusivity == 10.0
    # or can overwrite it in this order
    m.horizontal_diffusivity = 11
    assert m.horizontal_diffusivity == 11.0
    
    m = ptm.OpenDriftModel(ocean_model="NWGOA")
    assert m.horizontal_diffusivity == 150.0
