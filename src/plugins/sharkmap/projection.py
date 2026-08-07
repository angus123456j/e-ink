"""Map geometry shared between the plugin and its asset generator.

Both the committed basemap image and the runtime sighting positions are derived
from the constants here. Keeping them in one module means the two can never
drift apart -- if they did, fins would land off the coastline.

If you change any constant in this file you must regenerate world_map.png:

    python src/plugins/sharkmap/tools/generate_assets.py

Canvas layout, top to bottom:

    +--------------------------------------------------+  0
    |  header: title, compass, source credit           |
    +--------------------------------------------------+  HEADER_HEIGHT
    |                                                  |
    |  map band (the basemap image)                    |
    |                                                  |
    +--------------------------------------------------+  HEADER_HEIGHT + MAP_HEIGHT
    |  caption strip: most recent sighting             |
    +--------------------------------------------------+  CANVAS_HEIGHT
"""

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 480

HEADER_HEIGHT = 48
STRIP_HEIGHT = 42

MAP_WIDTH = CANVAS_WIDTH
MAP_HEIGHT = CANVAS_HEIGHT - HEADER_HEIGHT - STRIP_HEIGHT  # 390
MAP_TOP = HEADER_HEIGHT

# Equirectangular bounds.
#
# The far north is still cropped -- nothing is reported from the high Arctic and
# it is all ice -- but the south now reaches far enough to include the Antarctic
# coastline, so the map reads as a complete world rather than one that stops
# short. 162 degrees of latitude over 390 pixels is 2.41 px/degree against
# 2.22 px/degree horizontally, so the projection is very close to square.
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -82.0, 80.0


def project(lon, lat, width=MAP_WIDTH, height=MAP_HEIGHT):
    """Convert longitude/latitude in degrees to pixel coordinates.

    Returns coordinates local to the map band, so y=0 is the top of the map
    rather than the top of the canvas; callers working on the full canvas add
    MAP_TOP. Equirectangular means both axes are linear, so this is a rescale.
    y is flipped because latitude increases northward while pixel rows increase
    downward.
    """
    x = (lon - LON_MIN) / (LON_MAX - LON_MIN) * width
    y = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * height
    return x, y


def in_bounds(lon, lat):
    """True if a coordinate falls inside the mapped area.

    Coordinates outside the cropped latitude band are dropped rather than
    clamped, so nothing gets drawn pinned to the top or bottom edge in a place
    it was not observed.
    """
    return (
        lon is not None
        and lat is not None
        and -180.0 <= lon <= 180.0
        and LAT_MIN <= lat <= LAT_MAX
    )
