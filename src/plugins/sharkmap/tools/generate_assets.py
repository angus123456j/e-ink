"""Build-time generator for the SharkMap image assets.

Run this only when you want to regenerate the artwork; the plugin itself just
loads the committed PNGs. Nothing here runs on the Raspberry Pi.

    python src/plugins/sharkmap/tools/generate_assets.py
    python src/plugins/sharkmap/tools/generate_assets.py --tint 3 --compare

It produces three files next to the plugin:

  world_map.png  the nautical-atlas basemap: tinted ocean, white land, blue
                 coastline, graticule and place labels, all baked in
  fin.png        a shark-fin glyph whose *alpha channel* is the shape, so the
                 plugin can recolour it at draw time
  icon.png       the plugin icon for the InkyPi web UI

Two ideas underpin the artwork.

First, the basemap is drawn from vectors rather than downsampled from a
photograph. A photo produces thousands of intermediate colours which the Inky
driver then dithers into mud; filled polygons at the exact output resolution
give flat colour that survives the six-colour conversion intact.

Second, the pale blue ocean is a DESIGNED halftone. A Spectra 6 panel has
saturated blue but no pale blue, so the ocean is a sparse lattice of pure blue
pixels on pure white. Every pixel is already on-palette, so the driver has
nothing to dither and passes it through untouched, while the eye averages the
lattice into a tint. Letting the driver dither a genuinely pale colour instead
would produce the speckled mess visible on photographic plugins.

Everything static lives in the image so the Pi only has to draw fins, the
header and the caption each refresh.

Land geometry is Natural Earth (naturalearthdata.com), which is in the public
domain. It is downloaded on demand and is not committed to this repository.
"""

import argparse
import json
import os
import sys
import urllib.request

from PIL import Image, ImageDraw, ImageFont

# The plugin dir holds projection.py, the single source of truth for geometry.
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PLUGIN_DIR)

import projection  # noqa: E402  (needs the sys.path line above)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
FONT_DIR = os.path.join(PLUGIN_DIR, "fonts")

# --- Spectra 6 palette ----------------------------------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 190, 0)

# Ocean tint lattice pitch: one blue pixel every TINT_STEP**2, so a larger
# number is a fainter tint. 4 reads as roughly (239,239,255); 3 as (227,227,255).
# 2 is too strong -- the lattice becomes a visible pattern rather than a tint.
DEFAULT_TINT_STEP = 4

# Natural Earth land polygons. 50m has enough detail for real coastlines at
# 800px wide; 110m is visibly blocky.
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"

# Islands smaller than this (in square degrees) are dropped from the coastline.
# At 800px wide, 50m data turns archipelagos into blue static that reads as
# noise rather than geography; the reference atlas look depends on calm coasts.
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


def letterspace(draw, xy, text, fnt, fill, spacing=1.5, centre=True):
    """Draw text with extra tracking, the way atlas labels are set.

    Pillow has no letter-spacing, so characters are placed individually.
    Returns the total advance so callers can lay things out after it.
    """
    widths = [draw.textlength(ch, font=fnt) for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x = xy[0] - total / 2 if centre else xy[0]
    for ch, width in zip(text, widths):
        draw.text((x, xy[1]), ch, font=fnt, fill=fill, anchor="lm")
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

    Used only to decide whether an island is big enough to draw, so the
    distortion of treating degrees as planar does not matter.
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

def tint_ocean(image, step, width, height):
    """Lay the sparse blue lattice that stands in for pale blue.

    Rows are offset progressively so the dots form a diagonal lattice; a plain
    square grid shows up as visible rows and columns.
    """
    if not step:
        return
    pixels = image.load()
    for y in range(0, height, step):
        offset = ((y // step) * (step // 2 + 1)) % step
        for x in range(offset, width, step):
            pixels[x, y] = BLUE


def draw_graticule(draw, width, height, pitch=7):
    """Dotted 30-degree grid, sparse enough not to fight the ocean tint."""
    for lon in range(-150, 180, 30):
        x, _ = projection.project(lon, 0.0, width, height)
        for y in range(2, height, pitch):
            draw.point((x, y), fill=BLUE)

    for lat in (-60, -30, 0, 30, 60):
        _, y = projection.project(0.0, lat, width, height)
        for x in range(2, width, pitch):
            draw.point((x, y), fill=BLUE)


def draw_degree_labels(draw, width, height):
    """Latitude down the left edge, longitude along the bottom, both inset."""
    small = font(REGULAR, 8)

    for lat in (-60, -30, 0, 30, 60):
        _, y = projection.project(0.0, lat, width, height)
        hemisphere = "S" if lat < 0 else ("N" if lat > 0 else "")
        draw.text((4, y), f"{abs(lat)}°{hemisphere}", font=small,
                  fill=BLUE, anchor="lm")

    for lon in range(-120, 180, 60):
        x, _ = projection.project(lon, 0.0, width, height)
        hemisphere = "W" if lon < 0 else ("E" if lon > 0 else "")
        draw.text((x, height - 6), f"{abs(lon)}°{hemisphere}", font=small,
                  fill=BLUE, anchor="mm")


# Continents and countries, in black so they read against white land.
LAND_LABELS = [
    ("CANADA", -101, 61, 8),
    ("UNITED STATES", -99, 41, 8),
    ("MEXICO", -103, 24, 7),
    ("BRAZIL", -53, -10, 8),
    ("EUROPE", 19, 51, 8),
    ("AFRICA", 21, 3, 9),
    ("ASIA", 95, 47, 10),
    ("AUSTRALIA", 134, -25, 8),
]

# Ocean names in black rather than blue: blue italic over the blue tint was
# too faint to read at 8px.
OCEAN_LABELS = [
    ("NORTH PACIFIC OCEAN", -147, 28, 8),
    ("SOUTH PACIFIC OCEAN", -125, -30, 8),
    ("NORTH ATLANTIC OCEAN", -40, 20, 8),
    ("SOUTH ATLANTIC OCEAN", -20, -36, 8),
    ("INDIAN OCEAN", 80, -30, 8),
]


def generate_world_map(tint_step=DEFAULT_TINT_STEP, out_path=None, quiet=False):
    """Draw the atlas basemap: tint, land, coastline, graticule and labels."""
    width, height = projection.MAP_WIDTH, projection.MAP_HEIGHT
    geojson = fetch_land_geojson()

    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)

    tint_ocean(image, tint_step, width, height)

    polygons = list(iter_polygons(geojson))

    def to_pixels(ring):
        return [projection.project(p[0], p[1], width, height) for p in ring]

    # Drop specks up front so both the fill and the outline stay calm.
    drawable = [rings for rings in polygons
                if rings and ring_area(rings[0]) >= MIN_ISLAND_AREA]
    dropped = len(polygons) - len(drawable)

    # Land is pure white so it lifts cleanly out of the tinted ocean.
    for rings in drawable:
        outer = to_pixels(rings[0])
        if len(outer) >= 3:
            draw.polygon(outer, fill=WHITE)

    # Interior rings (inland seas such as the Caspian) read as water again.
    for rings in drawable:
        for hole in rings[1:]:
            pixels = to_pixels(hole)
            if len(pixels) >= 3:
                draw.polygon(pixels, fill=BLUE if tint_step == 0 else WHITE)
                if tint_step:
                    # Re-tint the hole so it matches the surrounding ocean.
                    bbox = [int(min(p[0] for p in pixels)), int(min(p[1] for p in pixels)),
                            int(max(p[0] for p in pixels)), int(max(p[1] for p in pixels))]
                    patch = Image.new("RGB", (max(1, bbox[2] - bbox[0]),
                                              max(1, bbox[3] - bbox[1])), WHITE)
                    tint_ocean(patch, tint_step, patch.width, patch.height)
                    mask = Image.new("L", patch.size, 0)
                    ImageDraw.Draw(mask).polygon(
                        [(p[0] - bbox[0], p[1] - bbox[1]) for p in pixels], fill=255)
                    image.paste(patch, (bbox[0], bbox[1]), mask)

    draw_graticule(draw, width, height)

    # Coastline last of the linework, so it sits above the graticule.
    for rings in drawable:
        for ring in rings:
            pixels = to_pixels(ring)
            if len(pixels) >= 2:
                draw.line(pixels + [pixels[0]], fill=BLUE, width=1)

    draw_degree_labels(draw, width, height)

    for text, lon, lat, size in LAND_LABELS:
        x, y = projection.project(lon, lat, width, height)
        letterspace(draw, (x, y), text, font(REGULAR, size), BLACK, spacing=1.3)

    for text, lon, lat, size in OCEAN_LABELS:
        x, y = projection.project(lon, lat, width, height)
        letterspace(draw, (x, y), text, font(ITALIC, size), BLACK, spacing=1.7)

    if out_path is None:
        out_path = os.path.join(PLUGIN_DIR, "world_map.png")
    image.save(out_path, "PNG", optimize=True)

    if not quiet:
        density = 0.0 if not tint_step else 1.0 / (tint_step ** 2)
        perceived = tuple(round(255 * (1 - density) + c * density) for c in BLUE)
        print(f"Wrote {out_path}")
        print(f"  {width}x{height}, tint step {tint_step or 'none'} "
              f"-> perceived ocean {perceived}")
        print(f"  {len(drawable)} land polygons drawn, {dropped} specks dropped "
              f"(< {MIN_ISLAND_AREA} sq deg)")
    return image


# --------------------------------------------------------------------------
# Fin glyph
# --------------------------------------------------------------------------

def quad_bezier(p0, p1, p2, steps=80):
    """Sample a quadratic bezier curve; used to shape the fin's edges."""
    points = []
    for index in range(steps + 1):
        t = index / steps
        inv = 1.0 - t
        points.append((
            inv * inv * p0[0] + 2 * inv * t * p1[0] + t * t * p2[0],
            inv * inv * p0[1] + 2 * inv * t * p1[1] + t * t * p2[1],
        ))
    return points


# A shark's first dorsal fin is *falcate*: the leading edge rakes back convexly
# to the apex, the trailing edge is strongly concave, and it ends in a "free
# rear tip" projecting backwards, low and close to the body. That rear tip and
# the notch under it are what make the silhouette read as a shark fin instead
# of a triangle, and they still read at 12px once downscaled. Unit square,
# y=0 at the apex.
FIN_APEX = (0.40, 0.00)
FIN_APEX_CTRL = (0.13, 0.26)   # bulges forward -> convex leading edge
FIN_REAR_TIP = (1.00, 0.70)
FIN_TRAIL_CTRL = (0.62, 0.22)  # pulls inward   -> concave trailing edge
FIN_NOTCH = (0.60, 1.00)
FIN_BASE_FRONT = (0.00, 1.00)

# The fin sits on a waterline. Silhouette subtlety is lost below ~32px, so the
# horizontal stroke is what signals "in the water" at small sizes.
FIN_BODY_FRACTION = 0.76
FIN_WATER_GAP = 0.07
FIN_WATER_WEIGHT = 0.12
FIN_WATER_SPREAD = 1.06

FIN_MASTER_SIZE = 336


def build_fin_mask(size=FIN_MASTER_SIZE, waterline=True):
    """Return an 'L' mode mask holding the fin silhouette."""
    outline = []
    outline += quad_bezier(FIN_BASE_FRONT, FIN_APEX_CTRL, FIN_APEX)
    outline += quad_bezier(FIN_APEX, FIN_TRAIL_CTRL, FIN_REAR_TIP)
    outline.append(FIN_NOTCH)
    outline.append(FIN_BASE_FRONT)

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    body_height = size * FIN_BODY_FRACTION
    draw.polygon([(x * (size - 1), y * body_height) for x, y in outline], fill=255)

    if waterline:
        weight = max(1, int(size * FIN_WATER_WEIGHT))
        top = body_height + size * FIN_WATER_GAP
        centre = size / 2.0
        half = (size / 2.0) * FIN_WATER_SPREAD
        draw.rectangle([centre - half, top, centre + half, top + weight], fill=255)

    bbox = mask.getbbox()
    return mask.crop(bbox) if bbox else mask


def generate_fin(out_path=None, quiet=False):
    """Write the fin glyph as an alpha-only PNG."""
    mask = build_fin_mask()
    fin = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    fin.putalpha(mask)

    if out_path is None:
        out_path = os.path.join(PLUGIN_DIR, "fin.png")
    fin.save(out_path, "PNG", optimize=True)

    if not quiet:
        print(f"Wrote {out_path}  ({mask.size[0]}x{mask.size[1]} alpha mask)")
    return fin


# --------------------------------------------------------------------------
# Web UI icon
# --------------------------------------------------------------------------

ICON_SIZE = 512


def generate_icon(out_path=None, quiet=False):
    """Draw the plugin icon shown in the InkyPi web UI."""
    icon = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))

    mask = build_fin_mask()
    target = int(ICON_SIZE * 0.74)
    w, h = mask.size
    scale = target / max(w, h)
    fin = mask.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    fin = fin.point(lambda a: 255 if a >= 128 else 0)

    left = (ICON_SIZE - fin.size[0]) // 2
    top = (ICON_SIZE - fin.size[1]) // 2
    icon.paste(Image.new("RGBA", fin.size, BLACK + (255,)), (left, top), fin)

    if out_path is None:
        out_path = os.path.join(PLUGIN_DIR, "icon.png")
    icon.save(out_path, "PNG", optimize=True)

    if not quiet:
        print(f"Wrote {out_path}  ({ICON_SIZE}x{ICON_SIZE} RGBA)")
    return icon


# --------------------------------------------------------------------------

def write_comparisons():
    """Render several tint densities so the look can be compared at real size."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for step in (0, 5, 4, 3, 2):
        path = os.path.join(CACHE_DIR, f"map_tint{step}.png")
        generate_world_map(step, out_path=path, quiet=True)
        print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tint", type=int, default=DEFAULT_TINT_STEP,
                        help="ocean tint lattice pitch; larger is fainter, "
                             "0 for a plain white ocean "
                             f"(default: {DEFAULT_TINT_STEP})")
    parser.add_argument("--compare", action="store_true",
                        help="also write a range of tint densities to .cache/")
    args = parser.parse_args()

    print("=== SharkMap asset generation ===")
    generate_world_map(args.tint)
    generate_fin()
    generate_icon()

    if args.compare:
        print("\nTint comparison renders:")
        write_comparisons()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
