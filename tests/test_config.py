"""Test configuration details."""

import particle_tracking_manager as ptm


def test_ptm_config():
    """Test that PTM config is brought in 
    
    ...and takes precendence over model config.
    """
    
    m = ptm.OpenDrift()
    
    # check for a single key and make sure override default is present
    assert m.show_config()["coastline_action"]["default"] == "previous"
    

def test_show_config():
    """Test configuration-showing functionality."""

    m = ptm.OpenDrift()

    # check sorting by a single level
    assert sorted(m.show_config(level=1).keys()) == ['seed:seafloor', 'seed:z', 'seed_seafloor', 'z']
    
    # check PTM level sorting
    assert "lon" in m.show_config(ptm_level=1).keys() and "log" not in m.show_config(ptm_level=1).keys()
