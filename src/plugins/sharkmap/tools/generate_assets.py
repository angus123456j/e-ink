"""Build-time generator for the SharkMap image assets.

Run this only when you want to regenerate the artwork; the plugin itself just
loads the committed PNGs. Nothing here runs on the Raspberry Pi.

    python src/plugins/sharkmap/tools/generate_assets.py
    python src/plugins/sharkmap/tools/generate_assets.py --compare

It produces two files next to the plugin:

  world_map.png  the equirectangular basemap, flat-filled for the Spectra 6
                 palette (ocean, land, black coastline)
  fin.png        a shark-fin glyph whose *alpha channel* is the shape; the
                 plugin recolours it at draw time, so one file serves both the
                 black fins and the red hotspot fins

Why generate the basemap from vectors instead of shipping a photograph:
a photographic map has to be downsampled, which produces thousands of
intermediate colours. The Inky library then dithers those to six colours and
the result is mud. Drawing filled polygons at the exact output resolution gives
genuinely flat colour that survives the conversion intact.

Land geometry is Natural Earth (naturalearthdata.com), which is in the public
domain. It is downloaded on demand and is not committed to this repository.
"""

import argparse
import json
import os
import sys
import urllib.request

from PIL import Image, ImageDraw

# The plugin dir holds projection.py, which defines the canvas geometry that
# both this generator and the running plugin must agree on.
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PLUGIN_DIR)

import projection  # noqa: E402  (needs the sys.path line above)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

# --- Spectra 6 palette ----------------------------------------------------
# Saturated values on purpose. The panel can only show these six, and muted
# tones dither into sludge.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 190, 0)

# Named basemap looks. `graticule` draws a dotted 30-degree grid, which reads as
# cartographic rather than decorative and costs almost nothing in ink.
MAP_STYLES = {
    "classic": {
        "ocean": WHITE, "land": YELLOW, "coast": BLACK,
        "graticule": None,
        "description": "white ocean, yellow land - maximum contrast for black fins",
    },
    "atlas": {
        "ocean": WHITE, "land": YELLOW, "coast": BLACK,
        "graticule": BLACK,
        "description": "classic plus a dotted 30-degree graticule",
    },
    "ocean": {
        "ocean": BLUE, "land": YELLOW, "coast": BLACK,
        "graticule": None,
        "description": "blue ocean, yellow land - most map-like, less fin contrast",
    },
    "chart": {
        "ocean": WHITE, "land": GREEN, "coast": BLACK,
        "graticule": BLACK,
        "description": "white ocean, green land, graticule",
    },
}

DEFAULT_STYLE = "atlas"

# Natural Earth land polygons. 50m is detailed enough that coastlines read
# clearly at 800px wide; 110m is noticeably blocky.
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"


# --------------------------------------------------------------------------
# Basemap
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
        print(f"  cached at {cache_path} ({len(data) // 1024} KB)")
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


def draw_graticule(draw, color, width, height, step=30):
    """Dotted latitude/longitude grid every `step` degrees.

    Dotted rather than solid so it sits behind the coastlines instead of
    competing with them -- we have no grey available, only pure black.
    """
    dot_on, dot_off = 2, 6

    for lon in range(-180 + step, 180, step):
        x, _ = projection.project(lon, 0.0, width, height)
        y = 0
        while y < height:
            draw.line([(x, y), (x, min(y + dot_on, height))], fill=color, width=1)
            y += dot_on + dot_off

    for lat in range(-90 + step, 90, step):
        _, y = projection.project(0.0, lat, width, height)
        x = 0
        while x < width:
            draw.line([(x, y), (min(x + dot_on, width), y)], fill=color, width=1)
            x += dot_on + dot_off


def generate_world_map(style_name=DEFAULT_STYLE, out_path=None, quiet=False):
    """Rasterise land polygons into the flat-colour basemap."""
    style = MAP_STYLES[style_name]
    width, height = projection.MAP_WIDTH, projection.MAP_HEIGHT
    geojson = fetch_land_geojson()

    image = Image.new("RGB", (width, height), style["ocean"])
    draw = ImageDraw.Draw(image)

    # Graticule goes down first so coastlines and land overpaint it.
    if style["graticule"]:
        draw_graticule(draw, style["graticule"], width, height)

    polygons = list(iter_polygons(geojson))

    def to_pixels(ring):
        return [projection.project(p[0], p[1], width, height) for p in ring]

    # Pass 1: fill land. ImageDraw.polygon is deliberately not anti-aliased,
    # which is what we want -- every pixel lands exactly on a palette colour.
    for rings in polygons:
        if rings:
            outer = to_pixels(rings[0])
            if len(outer) >= 3:
                draw.polygon(outer, fill=style["land"])

    # Pass 2: punch out interior rings (inland seas such as the Caspian) so
    # they read as water rather than land.
    for rings in polygons:
        for hole in rings[1:]:
            pixels = to_pixels(hole)
            if len(pixels) >= 3:
                draw.polygon(pixels, fill=style["ocean"])

    # Pass 3: trace coastlines. The 1px black outline is what makes the
    # continents genuinely recognisable at this size.
    for rings in polygons:
        for ring in rings:
            pixels = to_pixels(ring)
            if len(pixels) >= 2:
                draw.line(pixels + [pixels[0]], fill=style["coast"], width=1)

    if out_path is None:
        out_path = os.path.join(PLUGIN_DIR, "world_map.png")
    image.save(out_path, "PNG", optimize=True)

    if not quiet:
        colors = image.getcolors(maxcolors=64)
        print(f"Wrote {out_path}  [{style_name}: {style['description']}]")
        print(f"  {width}x{height}, {len(polygons)} polygons, "
              f"{len(colors) if colors else 'many'} distinct colours")
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
# rear tip" that projects backwards, low and close to the body. That rear tip
# and the notch under it are what make the silhouette read as a shark fin
# instead of a triangle -- and, crucially, they still read at 20px once the
# glyph is downscaled. Values are a unit square with y=0 at the apex.
FIN_APEX = (0.40, 0.00)
FIN_APEX_CTRL = (0.13, 0.26)   # bulges forward -> convex leading edge
FIN_REAR_TIP = (1.00, 0.70)
FIN_TRAIL_CTRL = (0.62, 0.22)  # pulls inward   -> concave trailing edge
FIN_NOTCH = (0.60, 1.00)
FIN_BASE_FRONT = (0.00, 1.00)

# The fin sits on a waterline. Silhouette subtlety is lost below ~32px, so the
# horizontal stroke is what signals "in the water" at small sizes.
FIN_BODY_FRACTION = 0.76   # share of glyph height taken by the fin itself
FIN_WATER_GAP = 0.07       # clear space between fin base and waterline
FIN_WATER_WEIGHT = 0.12    # waterline thickness
FIN_WATER_SPREAD = 1.06    # how far the waterline extends past the fin

# Drawn this many times larger than needed, then downscaled and re-thresholded,
# so the silhouette stays crisp at any display size.
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

    # Crop to the silhouette so the glyph has no dead margin; the plugin scales
    # by the glyph's own height.
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
    """Draw the plugin icon shown in the InkyPi web UI.

    A fin reads as "shark" far more quickly than a dot map does, so the icon
    uses the fin silhouette even though the current render draws dots.
    Transparent background, matching the other plugin icons.
    """
    icon = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))

    mask = build_fin_mask()
    # Leave a margin so the glyph is not flush to the edge.
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
# Comparison sheets (for choosing a look, not needed to build the assets)
# --------------------------------------------------------------------------

def write_comparisons():
    """Render every map style and a fin size ladder into the cache dir."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    for name in MAP_STYLES:
        path = os.path.join(CACHE_DIR, f"map_{name}.png")
        generate_world_map(name, out_path=path, quiet=True)
        print(f"  {path}")

    mask = build_fin_mask()
    sizes = (16, 20, 26, 32, 40)
    pad = 12
    sheet = Image.new("RGB", (sum(sizes) + pad * (len(sizes) + 1), max(sizes) + pad * 2), WHITE)
    x = pad
    for target in sizes:
        w, h = mask.size
        scale = target / max(w, h)
        small = mask.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        small = small.point(lambda a: 255 if a >= 128 else 0)
        patch = Image.new("RGB", small.size, BLACK)
        sheet.paste(patch, (x, pad), small)
        x += target + pad
    ladder = os.path.join(CACHE_DIR, "fin_sizes.png")
    sheet.resize((sheet.width * 3, sheet.height * 3), Image.NEAREST).save(ladder)
    print(f"  {ladder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", default=DEFAULT_STYLE, choices=sorted(MAP_STYLES),
                        help=f"basemap look (default: {DEFAULT_STYLE})")
    parser.add_argument("--compare", action="store_true",
                        help="also write every style and a fin size ladder to .cache/")
    args = parser.parse_args()

    print("=== SharkMap asset generation ===")
    generate_world_map(args.style)
    generate_fin()
    generate_icon()

    if args.compare:
        print("\nComparison renders:")
        write_comparisons()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
