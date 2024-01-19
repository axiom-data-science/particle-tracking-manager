# More examples for running PTM.

## Run with GeoJSON

If you prefer to create a GeoJSON object for seeding drifters, you can use that as an input to PTM, as shown here in an idealized example. These drifters don't travel anywhere since there is no reader defined and the fallback values for currents are all 0.

```
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
m.run()
```
