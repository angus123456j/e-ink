"""Build-time generator for the SharkMap image assets.

Run this only when you want to regenerate the artwork; the plugin itself just
loads the committed PNGs. Nothing here runs on the Raspberry Pi.

    python src/plugins/sharkmap/tools/generate_assets.py

It produces three files next to the plugin:

  world_map.png  the line-chart basemap: white ocean, black coastlines and
                 black place labels
  fin.png        the fin marker, derived from the supplied newfin.png artwork by
                 filling its outline into a solid silhouette
  icon.png       the plugin icon for the InkyPi web UI

newfin.png is hand-drawn source artwork and is never modified by this script.

Two decisions worth knowing about.

The basemap is drawn from vectors rather than downsampled from a photograph. A
photo produces thousands of intermediate colours which the Inky driver then
dithers into mud; filled polygons and 1px strokes at the exact output
resolution give flat colour and hard edges that survive the six-colour
conversion untouched.

The ocean is pure white, not a pale blue. Spectra 6 has saturated blue but no
pale blue, and the obvious workaround -- a sparse lattice of blue pixels read as
a tint -- does not survive contact with the hardware. The panel is about 159mm
wide for 800 pixels, so one pixel is 0.2mm; the tightest useful lattice puts
dots 0.8mm apart, while the eye resolves roughly 0.15mm at 50cm. The dots stay
visible as dots instead of blending. White ocean with black coastlines is the
honest version of this design at this pixel size.

Land geometry is Natural Earth (naturalearthdata.com), which is in the public
domain. It is downloaded on demand and is not committed to this repository.
"""

import argparse
import json
import os
import sys
import urllib.request
from collections import deque

from PIL import Image, ImageDraw, ImageFont

# The plugin dir holds projection.py, the single source of truth for geometry.
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PLUGIN_DIR)

import projection  # noqa: E402  (needs the sys.path line above)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
FIN_ART_PATH = os.path.join(PLUGIN_DIR, "newfin.png")
FONT_DIR = os.path.join(PLUGIN_DIR, "fonts")

# --- Palette --------------------------------------------------------------
# The design is deliberately black and white only. The panel can show six
# colours, but using two means every pixel is already exactly on-palette, so
# the driver never dithers anything and every edge stays hard. On a 0.2mm pixel
# this is what keeps fine linework and small type legible.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

OCEAN_COLOR = WHITE
LAND_COLOR = WHITE       # land is defined by its outline, as on a line chart
COAST_COLOR = BLACK

# Natural Earth land polygons. 50m has enough detail for real coastlines at
# 800px wide; 110m is visibly blocky.
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"

# Islands smaller than this (in square degrees) are dropped. At 800px wide, 50m
# data turns archipelagos into static that reads as noise rather than geography.
MIN_ISLAND_AREA = 0.75

REGULAR = "LiberationSerif-Regular.ttf"
BOLD = "LiberationSerif-Bold.ttf"
ITALIC = "LiberationSerif-Italic.ttf"

_font_cache = {}


def font(name, size):
    key = (name, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    return _font_cache[key]


# Type is collected into a mask and thresholded rather than drawn straight onto
# the image. Pillow anti-aliases TrueType text, leaving grey pixels along every
# stroke, and grey is not a colour this panel has -- the Inky driver dithers each
# one into a scattered red, green or blue dot, so small labels come out speckled
# and weak. Thresholding makes every text pixel pure black or pure white, which
# the driver maps straight across.
#
# Slightly below the midpoint: at 8px a serif stem only partly covers its pixels,
# and a strict 128 thins those strokes until they break.
TEXT_THRESHOLD = 100


def letterspace(draw, xy, text, fnt, spacing=1.5, centre=True):
    """Draw text with extra tracking, the way atlas labels are set.

    Pillow has no letter-spacing, so characters are placed individually.
    `draw` must target a mask; see TEXT_THRESHOLD.
    """
    widths = [draw.textlength(ch, font=fnt) for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x = xy[0] - total / 2 if centre else xy[0]
    for char, width in zip(text, widths):
        draw.text((x, xy[1]), char, font=fnt, fill=255, anchor="lm")
        x += width + spacing
    return total


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def fetch_land_geojson():
    """Download the land polygons, caching them so reruns are instant."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, os.path.basename(LAND_URL))

    if not os.path.exists(cache_path):
        print(f"Downloading {LAND_URL}")
        request = urllib.request.Request(
            LAND_URL, headers={"User-Agent": "InkyPi-SharkMap-assetgen/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        with open(cache_path, "wb") as handle:
            handle.write(data)
        print(f"  cached ({len(data) // 1024} KB)")
    else:
        print(f"Using cached {os.path.basename(cache_path)}")

    with open(cache_path) as handle:
        return json.load(handle)


def iter_polygons(geojson):
    """Yield each polygon as a ring list, flattening MultiPolygons.

    A GeoJSON ring list is [exterior, hole, hole, ...].
    """
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        coords = geometry.get("coordinates") or []
        if kind == "Polygon":
            yield coords
        elif kind == "MultiPolygon":
            for polygon in coords:
                yield polygon


def ring_area(ring):
    """Absolute area of a ring in square degrees, via the shoelace formula.

    Only used to decide whether an island is big enough to draw, so treating
    degrees as planar is fine.
    """
    total = 0.0
    for index in range(len(ring)):
        x1, y1 = ring[index][0], ring[index][1]
        x2, y2 = ring[(index + 1) % len(ring)][0], ring[(index + 1) % len(ring)][1]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


# --------------------------------------------------------------------------
# Basemap
# --------------------------------------------------------------------------

# No graticule and no degree labels. A latitude/longitude grid is conventional
# on an atlas plate, but at 800x480 it competes with the coastlines and the fins
# for the same few pixels, and it carries no information this display needs --
# the fins are read against the continents, not against coordinates.


# Continents and countries, in black to match the coastlines.
LAND_LABELS = [
    ("CANADA", -101, 61, 8),
    ("UNITED STATES", -99, 41, 8),
    ("MEXICO", -103, 24, 7),
    ("BRAZIL", -53, -10, 8),
    ("EUROPE", 19, 51, 8),
    ("AFRICA", 21, 3, 9),
    ("ASIA", 95, 47, 10),
    ("AUSTRALIA", 134, -25, 8),
    ("ANTARCTICA", 20, -76, 8),
]

# Ocean names in italic. With everything in black, the italic is what
# distinguishes water labels from the upright land labels.
OCEAN_LABELS = [
    ("NORTH PACIFIC OCEAN", -147, 26, 8),
    ("SOUTH PACIFIC OCEAN", -125, -30, 8),
    ("NORTH ATLANTIC OCEAN", -41, 19, 8),
    ("SOUTH ATLANTIC OCEAN", -21, -36, 8),
    ("INDIAN OCEAN", 80, -30, 8),
    ("SOUTHERN OCEAN", -60, -62, 8),
]


def generate_world_map(out_path=None, quiet=False):
    """Draw the basemap: white ocean, black coastlines, black place labels."""
    width, height = projection.MAP_WIDTH, projection.MAP_HEIGHT
    geojson = fetch_land_geojson()

    image = Image.new("RGB", (width, height), OCEAN_COLOR)
    draw = ImageDraw.Draw(image)

    polygons = list(iter_polygons(geojson))

    def to_pixels(ring):
        return [projection.project(p[0], p[1], width, height) for p in ring]

    # Drop specks up front so both the fill and the outline stay calm.
    drawable = [rings for rings in polygons
                if rings and ring_area(rings[0]) >= MIN_ISLAND_AREA]
    dropped = len(polygons) - len(drawable)

    for rings in drawable:
        outer = to_pixels(rings[0])
        if len(outer) >= 3:
            draw.polygon(outer, fill=LAND_COLOR)

    # Interior rings (inland seas such as the Caspian) read as water again.
    for rings in drawable:
        for hole in rings[1:]:
            pixels = to_pixels(hole)
            if len(pixels) >= 3:
                draw.polygon(pixels, fill=OCEAN_COLOR)

    # Coastlines last of the linework, so nothing paints over them.
    for rings in drawable:
        for ring in rings:
            pixels = to_pixels(ring)
            if len(pixels) >= 2:
                draw.line(pixels + [pixels[0]], fill=COAST_COLOR, width=1)

    # Labels are collected into a mask, thresholded, then stamped in solid black,
    # so no anti-aliased grey survives into the committed image.
    label_mask = Image.new("L", (width, height), 0)
    label_draw = ImageDraw.Draw(label_mask)

    for text, lon, lat, size in LAND_LABELS:
        x, y = projection.project(lon, lat, width, height)
        letterspace(label_draw, (x, y), text, font(REGULAR, size), spacing=1.3)

    for text, lon, lat, size in OCEAN_LABELS:
        x, y = projection.project(lon, lat, width, height)
        letterspace(label_draw, (x, y), text, font(ITALIC, size), spacing=1.7)

    hard = label_mask.point(lambda value: 255 if value >= TEXT_THRESHOLD else 0)
    image.paste(Image.new("RGB", (width, height), BLACK), (0, 0), hard)

    if out_path is None:
        out_path = os.path.join(PLUGIN_DIR, "world_map.png")
    image.save(out_path, "PNG", optimize=True)

    if not quiet:
        colours = image.getcolors(maxcolors=64)
        print(f"Wrote {out_path}")
        print(f"  {width}x{height}, {len(colours) if colours else 'many'} "
              f"distinct colours (want 2: white and black)")
        print(f"  {len(drawable)} land polygons drawn, {dropped} specks dropped "
              f"(< {MIN_ISLAND_AREA} sq deg)")
    return image


# --------------------------------------------------------------------------
# Fin marker, derived from the supplied artwork
# --------------------------------------------------------------------------

def fill_enclosed(mask):
    """Fill the interior of an outlined shape, returning a solid silhouette.

    The supplied fin is line art. Drawn hollow at 13px its strokes fall below a
    pixel and break up, and worse, a 1px outline carries exactly the same visual
    weight as the coastlines, so the fins stop reading as markers and blend into
    the map. Filling the interior keeps the artwork's shape and its waterline
    while making it the only solid mass on the page, which is what lets it stay
    legible when small.

    Works by flooding the background inward from the border; anything the flood
    cannot reach is enclosed, and gets filled.
    """
    binary = mask.point(lambda a: 255 if a >= 128 else 0)
    width, height = binary.size

    # Pad by a pixel so the flood always has an outside to start from, even
    # where a stroke touches the edge of the artwork.
    padded = Image.new("L", (width + 2, height + 2), 0)
    padded.paste(binary, (1, 1))
    pw, ph = padded.size
    pixels = padded.load()

    seen = bytearray(pw * ph)
    queue = deque([(0, 0)])
    seen[0] = 1
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < pw and 0 <= ny < ph and not seen[ny * pw + nx] \
                    and not pixels[nx, ny]:
                seen[ny * pw + nx] = 1
                queue.append((nx, ny))

    filled = 0
    for y in range(ph):
        for x in range(pw):
            if not pixels[x, y] and not seen[y * pw + x]:
                pixels[x, y] = 255
                filled += 1

    return padded.crop((1, 1, width + 1, height + 1)), filled


def load_fin_art():
    """Return the alpha mask of the supplied fin artwork, trimmed."""
    art = Image.open(FIN_ART_PATH)
    art.load()
    mask = art.split()[-1] if art.mode == "RGBA" else art.convert("L")
    box = mask.getbbox()
    return mask.crop(box) if box else mask


def generate_fin(out_path=None, quiet=False):
    """Write the solid fin marker the plugin draws on the map."""
    solid, filled = fill_enclosed(load_fin_art())

    fin = Image.new("RGBA", solid.size, (0, 0, 0, 0))
    fin.putalpha(solid)

    if out_path is None:
        out_path = os.path.join(PLUGIN_DIR, "fin.png")
    fin.save(out_path, "PNG", optimize=True)

    if not quiet:
        print(f"Wrote {out_path}  ({solid.size[0]}x{solid.size[1]} alpha mask, "
              f"{filled} interior pixels filled)")
    return fin


# --------------------------------------------------------------------------
# Web UI icon
# --------------------------------------------------------------------------

ICON_SIZE = 512


def generate_icon(out_path=None, quiet=False):
    """Draw the plugin icon shown in the InkyPi web UI.

    Derived from the same fin artwork the map uses, so the icon and the markers
    are recognisably the same thing.
    """
    icon = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))

    mask, _ = fill_enclosed(load_fin_art())

    target = int(ICON_SIZE * 0.78)
    w, h = mask.size
    scale = target / max(w, h)
    fin = mask.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    left = (ICON_SIZE - fin.size[0]) // 2
    top = (ICON_SIZE - fin.size[1]) // 2
    # Black, matching the fins on the map itself.
    icon.paste(Image.new("RGBA", fin.size, BLACK + (255,)), (left, top), fin)

    if out_path is None:
        out_path = os.path.join(PLUGIN_DIR, "icon.png")
    icon.save(out_path, "PNG", optimize=True)

    if not quiet:
        print(f"Wrote {out_path}  ({ICON_SIZE}x{ICON_SIZE} RGBA)")
    return icon


# --------------------------------------------------------------------------

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    print("=== SharkMap asset generation ===")
    generate_world_map()
    generate_fin()
    generate_icon()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
