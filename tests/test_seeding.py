"""Test seeding functionality."""

from datetime import datetime

import numpy as np
import pytest

import particle_tracking_manager as ptm


# def test_seeding_from_wkt():
#     """Seed from WKT."""

#     wkt = "POLYGON((-151.0 58.0, -150.0 58.0, -150.0 59.0, -151.0 59.0, -151.0 58.0))"

#     m = ptm.OpenDriftModel(seed_flag="wkt", wkt=wkt, use_auto_landmask=True)
#     m.seed()
#     import pdb; pdb.set_trace()


def test_seeding_from_geojson():
    """Seed from GeoJSON."""

    geo = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-151.0, 58.0],
                    [-150.0, 58.0],
                    [-150.0, 59.0],
                    [-151.0, 59.0],
                    [-151.0, 58.0],
                ]
            ],
        },
    }
    m = ptm.OpenDriftModel(
        seed_flag="geojson",
        start_time="2000-01-01",
        geojson=geo,
        use_auto_landmask=True,
        number=2,
    )
    m.seed()

    expected_lon = [-150.51787, -150.51787]
    expected_lat = [58.25654, 58.25654]

    assert np.allclose(m.initial_drifters.lon, expected_lon)
    assert np.allclose(m.initial_drifters.lat, expected_lat)
