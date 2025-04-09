"""Test seeding functionality."""

from datetime import datetime

import numpy as np
import pytest

import particle_tracking_manager as ptm


@pytest.mark.slow
def test_seeding_from_geojson():
    """Seed from GeoJSON."""

    geo = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-90, 28.7],
                    [-90, 28.8],
                    [-90.1, 28.8],
                    [-90.1, 28.7],
                    [-90, 28.7],
                ]
            ],
        },
    }
    m = ptm.OpenDriftModel(
        seed_flag="geojson",
        geojson=geo,
        use_auto_landmask=True,
        number=2,
        steps=1,
        lon=None,
        lat=None,
        ocean_model="TXLA",
        ocean_model_local=False,
        start_time=datetime(2009, 11, 19, 12, 0),
    )
    m.add_reader()
    m.seed()

    expected_lon = [-90.06226, -90.06226]
    expected_lat = [28.733112, 28.733112]

    assert np.allclose(m.initial_drifters.lon, expected_lon)
    assert np.allclose(m.initial_drifters.lat, expected_lat)
