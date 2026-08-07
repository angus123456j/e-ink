"""Map geometry shared between the plugin and its asset generator.

Both the committed basemap image and the runtime fin positions are derived from
the constants here. Keeping them in one module means the two can never drift
apart -- if they did, fins would land in the wrong place on the coastline.

If you change any constant in this file you must regenerate world_map.png:

    python src/plugins/sharkmap/tools/generate_assets.py
"""

# Canvas split: map on top, caption strip along the bottom.
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 480
STRIP_HEIGHT = 50

MAP_WIDTH = CANVAS_WIDTH
MAP_HEIGHT = CANVAS_HEIGHT - STRIP_HEIGHT  # 430

# Equirectangular bounds. The full globe is the honest default, but the polar
# regions hold almost no shark sightings, so trimming them buys roughly 20%
# more usable map area for the same pixels. See TRIMMED_BOUNDS below.
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0


def project(lon, lat, width=MAP_WIDTH, height=MAP_HEIGHT,
            lon_min=LON_MIN, lon_max=LON_MAX,
            lat_min=LAT_MIN, lat_max=LAT_MAX):
    """Convert longitude/latitude in degrees to pixel coordinates.

    Equirectangular: both axes are linear, so this is just a rescale. Returns
    floats; the caller rounds. Note that y is flipped, because latitude
    increases northward while pixel rows increase downward.
    """
    x = (lon - lon_min) / (lon_max - lon_min) * width
    y = (lat_max - lat) / (lat_max - lat_min) * height
    return x, y


def in_bounds(lon, lat, lat_min=LAT_MIN, lat_max=LAT_MAX):
    """True if a coordinate falls inside the mapped area."""
    return (
        lon is not None
        and lat is not None
        and -180.0 <= lon <= 180.0
        and lat_min <= lat <= lat_max
    )
