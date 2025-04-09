from particle_tracking_manager.config_logging import LoggerConfig


def test_log_level():
    """Test values of parameters being input."""

    m = LoggerConfig(
        log_level="DEBUG",
    )
    assert m.log_level == "DEBUG"
