"""SharkMap: a nautical-atlas world map of recent shark sightings.

Drawn with Pillow rather than HTML, because Chromium costs 40+ seconds of boot
and render time on a Pi Zero 2 W. Everything static -- coastlines and place
names -- is baked into world_map.png at build time, so a refresh only draws the
header, the fins and the caption.

The whole design is black on white. The panel can show six colours, but two
means every pixel is already exactly on-palette, so the driver never dithers
and every edge stays hard. At 0.2mm per pixel that is what keeps 1px linework
and 8px type readable.

Data comes from the iNaturalist API v1, which needs no key. See fetch_sightings
for the query and the traps in it.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone

from PIL import Image, ImageDraw, ImageFont

from plugins.base_plugin.base_plugin import BasePlugin
from plugins.sharkmap.projection import (
    CANVAS_HEIGHT,
    HEADER_HEIGHT,
    STRIP_HEIGHT,
    in_bounds,
    project,
)
from utils.http_client import get_http_session

logger = logging.getLogger(__name__)

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
WORLD_MAP_PATH = os.path.join(PLUGIN_DIR, "world_map.png")
# Derived from newfin.png at build time by filling its outline; see
# tools/generate_assets.py for why the marker is solid rather than line art.
FIN_PATH = os.path.join(PLUGIN_DIR, "fin.png")
FONT_DIR = os.path.join(PLUGIN_DIR, "fonts")

REGULAR = "LiberationSerif-Regular.ttf"
BOLD = "LiberationSerif-Bold.ttf"
ITALIC = "LiberationSerif-Italic.ttf"

# --- iNaturalist ----------------------------------------------------------
INAT_OBSERVATIONS_URL = "https://api.inaturalist.org/v1/observations"
INAT_AUTOCOMPLETE_URL = "https://api.inaturalist.org/v1/taxa/autocomplete"
INAT_PLACES_URL = "https://api.inaturalist.org/v1/places"

# Administrative levels iNaturalist tags places with. Used to turn the raw
# place_ids on an observation into a readable location.
ADMIN_LEVEL_COUNTRY = 0
ADMIN_LEVEL_STATE = 10
ADMIN_LEVEL_COUNTY = 20

# Selachii, the infraclass containing all sharks.
#
# Two traps here, both verified against the live API:
#  1. "Selachimorpha" is not a name iNaturalist knows. Querying it alongside
#     quality_grade=research returns zero results.
#  2. taxon_name matches ONLY observations identified at exactly that taxon,
#     while taxon_id walks the whole subtree. For this taxon that is the
#     difference between ~1,350 and ~96,000 observations, so we must use the id.
SHARK_TAXON_ID = 551307

# iNaturalist asks for roughly 1 request/second and 10k/day, and serves at most
# 200 results per page. A 30-minute refresh is ~48 calls/day, well inside that.
INAT_PER_PAGE = 200
REQUEST_TIMEOUT = 20  # seconds

# Sent alongside InkyPi's shared User-Agent so iNaturalist can identify this
# specific client, which their API terms ask for.
USER_AGENT = "InkyPi-SharkMap/1.0 (+https://github.com/fatihak/InkyPi)"

# --- Settings bounds ------------------------------------------------------
DEFAULT_LOOKBACK_DAYS = 7
MIN_LOOKBACK_DAYS, MAX_LOOKBACK_DAYS = 1, 90
DEFAULT_MAX_SPOTS = 40
MIN_MAX_SPOTS, MAX_MAX_SPOTS = 5, 200

# --- Palette --------------------------------------------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# The whole design is black on white. Using two colours means every pixel is
# already exactly on-palette, so the driver never dithers and every edge stays
# hard -- which is what keeps 1px linework and 8px type legible on a 0.2mm pixel.
FIN_COLOR = BLACK

# --- Fin layout -----------------------------------------------------------
# Sightings are binned into small cells for counting, then thinned by distance so
# fins cannot overlap. Binning alone is not enough: a fin is placed at its cell's
# centroid rather than the cell centre, so two adjacent cells can put their fins a
# couple of pixels apart and the glyphs merge into a blob. Thinning by actual
# distance keeps the natural scatter of real positions while guaranteeing every
# fin reads as a separate mark.
GRID_CELL_PX = 14

# Extra clearance between fins, on top of the glyph's own width. Separation has
# to be derived from the drawn width rather than fixed: the artwork is wider than
# it is tall, so a constant that looked right for one fin height silently allowed
# overlaps at another.
FIN_CLEARANCE_PX = 7

# Every fin is the same size. Scaling them by how many sightings they stood for
# made the map look busier than the data warranted, and the caption already
# reports the total.
FIN_HEIGHT_PX = 11

# Threshold used to harden the glyph after downscaling. 128 is right for a solid
# silhouette; line art needs a lower value, or strokes landing on part-covered
# pixels drop out and the outline breaks up.
FIN_ALPHA_THRESHOLD = 128

PLACE_MAX_CHARS = 42      # place_guess is free text and can run long


class CrispText:
    """Collects text into a 1-bit mask so it can be pasted with hard edges.

    Pillow anti-aliases TrueType text, which leaves grey pixels along every
    stroke. Grey is not a colour this panel has, so the Inky driver dithers each
    of those pixels into a scattered red, green or blue dot, and small type ends
    up looking speckled and weak rather than black.

    Drawing type into a mask and thresholding it means every text pixel is either
    pure black or pure white, so the driver maps it straight across and the
    letters stay hard. The cost is slightly chunkier small text, which is a good
    trade at 8px on a 0.2mm pixel.
    """

    # Chosen a little below the midpoint: at 8px a serif stem covers only part of
    # its pixels, and a strict 128 thins those strokes to the point of breaking.
    THRESHOLD = 100

    def __init__(self, size):
        self.layer = Image.new("L", size, 0)
        self.draw = ImageDraw.Draw(self.layer)

    def text(self, xy, string, font, anchor="lm"):
        self.draw.text(xy, string, font=font, fill=255, anchor=anchor)

    def length(self, string, font):
        return self.draw.textlength(string, font=font)

    def flush(self, canvas, colour):
        """Threshold the collected type and paste it onto the canvas."""
        mask = self.layer.point(lambda value: 255 if value >= self.THRESHOLD else 0)
        canvas.paste(Image.new("RGB", canvas.size, colour), (0, 0), mask)


class SharkMap(BasePlugin):
    """Paints recent shark sightings onto a nautical-atlas world map."""

    # ------------------------------------------------------------------
    # InkyPi entry point
    # ------------------------------------------------------------------
    def generate_image(self, settings, device_config):
        lookback_days = self._read_int(
            settings.get("lookbackDays"), DEFAULT_LOOKBACK_DAYS,
            MIN_LOOKBACK_DAYS, MAX_LOOKBACK_DAYS, "lookbackDays")
        max_spots = self._read_int(
            settings.get("maxSpots"), DEFAULT_MAX_SPOTS,
            MIN_MAX_SPOTS, MAX_MAX_SPOTS, "maxSpots")
        show_strip = str(settings.get("showStrip", "true")).lower() != "false"
        species_filter = (settings.get("speciesFilter") or "").strip()

        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]

        sightings, total_reported, stale_since = self._load_sightings(
            lookback_days, species_filter)

        if not sightings:
            # Nothing to draw and nothing cached: surface it in the web UI
            # rather than painting a blank map.
            raise RuntimeError(
                "No shark sightings available. iNaturalist returned no "
                "observations and there is no cached data to fall back on."
            )

        return self._render(
            dimensions=dimensions,
            sightings=sightings,
            total_reported=total_reported,
            max_spots=max_spots,
            show_strip=show_strip,
            lookback_days=lookback_days,
            stale_since=stale_since,
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    @staticmethod
    def _read_int(raw, default, low, high, label):
        """Parse a form value into an int, clamping to a sane range.

        Form fields arrive as strings and may be blank. A junk value is not
        worth failing a refresh over, so we log it and use the default.
        """
        if raw is None or str(raw).strip() == "":
            return default
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            logger.warning("SharkMap: %s=%r is not a number, using %d", label, raw, default)
            return default

        clamped = max(low, min(high, value))
        if clamped != value:
            logger.warning("SharkMap: %s=%d out of range, clamped to %d", label, value, clamped)
        return clamped

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _load_sightings(self, lookback_days, species_filter):
        """Return (sightings, total_reported, stale_since).

        Tries the network first. If anything goes wrong -- timeout, DNS
        failure, non-200, malformed JSON -- falls back to the last good
        response on disk, so a flaky connection degrades to slightly old data
        instead of an error screen. stale_since is None when the data is fresh,
        otherwise the time the cache was written.

        total_reported is how many observations matched the query overall,
        which can exceed the single page we fetched.
        """
        cache_path = self._cache_path(lookback_days, species_filter)

        try:
            sightings, total_reported = self.fetch_sightings(
                lookback_days, species_filter)
            if sightings:
                # Only the sighting the caption names needs a readable place, so
                # the extra lookup is one request per refresh, not two hundred.
                self._resolve_place_label(self._most_recent(sightings))
                # place_ids exist only to feed that lookup; drop them before
                # caching so the file stays small.
                for sighting in sightings:
                    sighting.pop("place_ids", None)
                self._write_cache(cache_path, sightings, total_reported)
                return sightings, total_reported, None
            # A successful but empty response is not worth caching; fall
            # through to cached data if we have any.
            logger.warning("SharkMap: iNaturalist returned no observations")
        except RuntimeError:
            # Raised for genuine configuration problems, such as a species
            # filter that matches nothing. The web UI should show these.
            raise
        except Exception as error:  # noqa: BLE001 - must never break the refresh loop
            logger.warning("SharkMap: fetch failed (%s), trying cache", error)

        cached, cached_total, written_at = self._read_cache(cache_path)
        if cached:
            logger.info("SharkMap: serving %d cached sightings from %s",
                        len(cached), written_at)
            return cached, cached_total, written_at

        return [], 0, None

    def fetch_sightings(self, lookback_days, species_filter):
        """Query iNaturalist; return (sightings, total_reported).

        We keep only the fields the render needs. The raw payload for 200
        observations is over a megabyte and the service runs under a 200MB cap,
        so there is no reason to hold on to it.
        """
        session = get_http_session()
        headers = {"User-Agent": USER_AGENT}

        taxon_id = SHARK_TAXON_ID
        if species_filter:
            taxon_id = self._resolve_species(session, headers, species_filter)

        since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        params = {
            "taxon_id": taxon_id,
            "geo": "true",                   # must have coordinates to plot
            "quality_grade": "research",      # community-verified identifications
            "order_by": "observed_on",
            "order": "desc",
            "per_page": INAT_PER_PAGE,
            "d1": since.date().isoformat(),
        }

        logger.info("SharkMap: requesting iNaturalist observations "
                    "(taxon_id=%s, last %d days)", taxon_id, lookback_days)
        response = session.get(INAT_OBSERVATIONS_URL, params=params,
                               headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            raise RuntimeError(f"iNaturalist returned HTTP {response.status_code}")

        payload = response.json()
        results = payload.get("results") or []
        # total_results counts every match, which may exceed the single page we
        # asked for; order=desc means the page we got is the most recent slice.
        total_reported = payload.get("total_results") or len(results)
        logger.info("SharkMap: %d observations returned (%s total match)",
                    len(results), total_reported)

        distilled = (self._distil(record) for record in results)
        return [s for s in distilled if s is not None], int(total_reported)

    def _resolve_species(self, session, headers, species_filter):
        """Turn a typed species name into an iNaturalist taxon id.

        Accepts common or scientific names. Ancestry is checked so a filter of
        "seal" cannot quietly redirect the map away from sharks.
        """
        response = session.get(
            INAT_AUTOCOMPLETE_URL,
            params={"q": species_filter, "is_active": "true", "per_page": 10},
            headers=headers, timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Could not look up species '{species_filter}': "
                f"iNaturalist returned HTTP {response.status_code}"
            )

        for taxon in response.json().get("results") or []:
            ancestry = taxon.get("ancestor_ids") or []
            if SHARK_TAXON_ID in ancestry or taxon.get("id") == SHARK_TAXON_ID:
                logger.info("SharkMap: species filter %r resolved to %s (id=%s)",
                            species_filter, taxon.get("name"), taxon.get("id"))
                return taxon["id"]

        raise RuntimeError(
            f"'{species_filter}' did not match any shark on iNaturalist. "
            "Try a name like 'Tiger Shark' or 'Carcharodon carcharias', "
            "or clear the field to show all sharks."
        )

    def _resolve_place_label(self, sighting):
        """Turn a sighting's place_ids into a readable location, in place.

        place_guess is free text typed by the observer and is often unusable: a
        bare ISO country code such as "BS", a subdivision code such as
        "DK-ND, DK", or a country name in the observer's own language such as
        "Spagna" for Spain. Roughly one record in ten looks like this.

        iNaturalist also tags each observation with place_ids, which resolve to
        properly named places carrying an administrative level. Those give
        "New Providence, Bahamas" instead of "BS", in English, regardless of who
        entered the record.

        Best effort only: on any failure the caption falls back to place_guess,
        because a slightly cryptic location is much better than a failed refresh.
        """
        if not sighting:
            return

        place_ids = sighting.get("place_ids") or []
        if not place_ids:
            return

        try:
            session = get_http_session()
            # One batched request. Cap the ids so a record tagged with dozens of
            # project boundaries cannot build an unreasonable URL.
            ids = ",".join(str(int(pid)) for pid in place_ids[:40])
            response = session.get(
                f"{INAT_PLACES_URL}/{ids}",
                headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT,
            )
            if response.status_code != 200:
                logger.info("SharkMap: place lookup returned HTTP %s",
                            response.status_code)
                return

            by_level = {}
            for place in response.json().get("results") or []:
                level = place.get("admin_level")
                name = (place.get("name") or "").strip()
                # Places without an admin level are ecoregions, project
                # boundaries and similar; they are not locations a reader wants.
                if level is None or not name:
                    continue
                by_level.setdefault(int(level), name)

            # Most specific administrative unit, then the country, which is what
            # actually orients someone looking at a world map.
            specific = by_level.get(ADMIN_LEVEL_COUNTY) or by_level.get(ADMIN_LEVEL_STATE)
            country = by_level.get(ADMIN_LEVEL_COUNTRY)
            parts = [part for part in (specific, country) if part]

            if parts:
                sighting["place_label"] = ", ".join(parts)
                logger.info("SharkMap: resolved place %r to %r",
                            sighting.get("place"), sighting["place_label"])
        except Exception as error:  # noqa: BLE001 - never break a refresh for a label
            logger.info("SharkMap: place lookup failed (%s)", error)

    @staticmethod
    def _distil(record):
        """Reduce one API record to the fields we draw, or None if unusable."""
        coords = (record.get("geojson") or {}).get("coordinates")
        if not coords or len(coords) != 2:
            return None

        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            return None
        if not in_bounds(lon, lat):
            return None

        taxon = record.get("taxon") or {}
        return {
            "lon": lon,
            "lat": lat,
            # preferred_common_name is frequently null; the scientific name is
            # the documented fallback.
            "common": taxon.get("preferred_common_name") or None,
            "scientific": taxon.get("name") or None,
            # Free text typed by the observer. Often absent, sometimes long, and
            # sometimes just a country code. Used only as a fallback when the
            # structured place lookup below cannot produce anything.
            "place": record.get("place_guess") or None,
            # Structured place references, resolved to a readable name for the
            # single sighting the caption describes, then discarded.
            "place_ids": record.get("place_ids") or [],
            # time_observed_at is often null, in which case only the date is
            # known and we must not invent a time.
            "time": record.get("time_observed_at") or None,
            "date": record.get("observed_on") or None,
            # iNaturalist randomises coordinates for threatened species. Worth
            # saying so rather than implying the position is exact.
            "obscured": bool(record.get("obscured") or record.get("geoprivacy")),
        }

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------
    def _cache_dir(self):
        """Writable directory for cached responses.

        Prefers a dot-dir beside the plugin so the cache survives reboots, and
        falls back to the system temp dir if the install is read-only.
        """
        preferred = os.path.join(PLUGIN_DIR, ".cache")
        try:
            os.makedirs(preferred, exist_ok=True)
            probe = os.path.join(preferred, ".writable")
            with open(probe, "w") as handle:
                handle.write("")
            os.remove(probe)
            return preferred
        except OSError:
            fallback = os.path.join(tempfile.gettempdir(), "inkypi-sharkmap")
            os.makedirs(fallback, exist_ok=True)
            return fallback

    def _cache_path(self, lookback_days, species_filter):
        """One cache file per distinct query, so settings do not collide."""
        species = "".join(
            ch if ch.isalnum() else "_" for ch in species_filter.lower()
        ) or "all"
        return os.path.join(
            self._cache_dir(), f"sightings_{lookback_days}d_{species}.json"
        )

    @staticmethod
    def _write_cache(path, sightings, total_reported):
        """Persist the distilled sightings. Cache failures are never fatal."""
        try:
            payload = {
                "written_at": datetime.now(timezone.utc).isoformat(),
                "total_reported": total_reported,
                "sightings": sightings,
            }
            # Write then rename, so an interrupted write cannot leave a
            # truncated cache behind.
            temp_path = f"{path}.tmp"
            with open(temp_path, "w") as handle:
                json.dump(payload, handle)
            os.replace(temp_path, path)
        except OSError as error:
            logger.warning("SharkMap: could not write cache: %s", error)

    @staticmethod
    def _read_cache(path):
        """Return (sightings, total_reported, written_at), or empties."""
        try:
            with open(path) as handle:
                payload = json.load(handle)
            sightings = payload.get("sightings") or []
            total = payload.get("total_reported") or len(sightings)
            return sightings, int(total), parse_iso(payload.get("written_at"))
        except (OSError, ValueError) as error:
            logger.info("SharkMap: no usable cache at %s (%s)", path, error)
            return [], 0, None

    # ------------------------------------------------------------------
    # Fonts and text
    # ------------------------------------------------------------------
    def _font(self, filename, size):
        """Load a bundled font.

        Liberation Serif ships with the plugin because it is metrically
        identical to Times New Roman but freely redistributable, and Raspberry
        Pi OS has no Times New Roman to fall back on.
        """
        path = os.path.join(FONT_DIR, filename)
        try:
            return ImageFont.truetype(path, max(6, int(size)))
        except OSError:
            logger.warning("SharkMap: missing font %s, using PIL default", path)
            return ImageFont.load_default()

    @staticmethod
    def _letterspace(ink, xy, text, font, spacing=1.5, centre=True):
        """Draw text with extra tracking, the way atlas labels are set.

        Pillow has no letter-spacing, so characters are placed individually.
        Returns the total advance.
        """
        widths = [ink.length(ch, font) for ch in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = xy[0] - total / 2 if centre else xy[0]
        for char, width in zip(text, widths):
            ink.text((x, xy[1]), char, font)
            x += width + spacing
        return total

    @staticmethod
    def _fit_text(ink, text, font, max_width):
        """Trim text with an ellipsis until it fits `max_width` pixels."""
        if ink.length(text, font) <= max_width:
            return text
        trimmed = text
        while trimmed and ink.length(trimmed + "…", font) > max_width:
            trimmed = trimmed[:-1]
        return trimmed.rstrip() + "…"

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self, dimensions, sightings, total_reported, max_spots,
                show_strip, lookback_days, stale_since):
        width, height = int(dimensions[0]), int(dimensions[1])

        # Everything is authored for 800x480. On any other panel the bands and
        # type scale proportionally rather than being clipped.
        scale = height / CANVAS_HEIGHT

        header_height = max(24, round(HEADER_HEIGHT * scale))
        strip_height = max(30, round(STRIP_HEIGHT * scale)) if show_strip else 0
        map_height = height - header_height - strip_height

        canvas = Image.new("RGB", (width, height), WHITE)
        canvas.paste(self._basemap(width, map_height), (0, header_height))

        draw = ImageDraw.Draw(canvas)
        # All type goes into this mask and is thresholded on the way out, so no
        # anti-aliased grey ever reaches the panel. See CrispText.
        ink = CrispText((width, height))
        self._draw_header(draw, ink, width, header_height, scale)

        # The glyph is prepared once: every fin is identical, and its drawn width
        # is what decides how far apart fins have to be to stay distinct.
        glyph = self._fin_glyph(scale)
        separation = glyph.width + FIN_CLEARANCE_PX * scale

        latest = self._most_recent(sightings)
        cells = self._bin_sightings(sightings, width, map_height, max_spots,
                                    separation, latest)
        self._draw_fins(canvas, glyph, cells, header_height, map_height)

        if show_strip:
            self._draw_strip(draw, ink, latest, total_reported, width, height,
                             strip_height, lookback_days, stale_since, scale)

        ink.flush(canvas, BLACK)
        return canvas

    def _basemap(self, width, map_height):
        """Load the basemap, resizing only if the panel is not 800x480.

        NEAREST is deliberate: it cannot invent intermediate colours, so the
        image stays exactly on-palette at any size. A smoother filter would
        blend the ocean tint lattice into intermediate blues, and the driver
        would then dither those into visible noise.
        """
        try:
            basemap = Image.open(WORLD_MAP_PATH)
            basemap.load()
            basemap = basemap.convert("RGB")
        except OSError as error:
            raise RuntimeError(
                f"SharkMap could not load its basemap ({WORLD_MAP_PATH}): {error}"
            ) from error

        if basemap.size != (width, map_height):
            basemap = basemap.resize((width, map_height), Image.NEAREST)
        return basemap

    def _draw_header(self, draw, ink, width, header_height, scale):
        """Title block: name, subtitle, compass rose and the source credit."""
        draw.rectangle([0, 0, width, header_height], fill=WHITE)

        # Sized to stack a title and a subtitle inside a 34px band without
        # either one touching the rule below or the top edge.
        title_font = self._font(BOLD, 16 * scale)
        sub_font = self._font(REGULAR, 8 * scale)

        self._letterspace(ink, (width / 2, header_height * 0.38),
                          "SHARK SIGHTINGS", title_font, spacing=2.6 * scale)
        self._letterspace(ink, (width / 2, header_height * 0.79),
                          "LIVE MARITIME OBSERVATIONS", sub_font,
                          spacing=2.2 * scale)

        draw.line([(0, header_height - 1), (width, header_height - 1)],
                  fill=BLACK, width=1)

        # The rose plus its "N" has to fit inside a 34px band, so it is small and
        # sits low: the letter goes above the northern spike, and at a larger
        # radius it was being clipped by the top edge.
        self._draw_compass(draw, ink, width * 0.055, header_height * 0.62,
                           header_height * 0.22, scale)

        # The source credit iNaturalist's terms ask for, given the prominence a
        # decorative brand block would otherwise take.
        credit_x = width - 150 * scale
        self._letterspace(ink, (credit_x, header_height * 0.36),
                          "DATA: iNATURALIST", self._font(BOLD, 8 * scale),
                          spacing=1.0 * scale, centre=False)
        ink.text((credit_x, header_height * 0.66), "research-grade observations",
                 self._font(ITALIC, 8.5 * scale))

    def _draw_compass(self, draw, ink, cx, cy, radius, scale):
        """A flat compass rose: four cardinal spikes plus four minor ones."""
        radius = max(6, radius)
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            tip = (cx + dx * radius, cy + dy * radius)
            base = radius * 0.26
            points = ([tip, (cx, cy - base), (cx, cy + base)] if dx
                      else [tip, (cx - base, cy), (cx + base, cy)])
            draw.polygon(points, fill=BLACK)
        for dx, dy in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
            tip = (cx + dx * radius * 0.5, cy + dy * radius * 0.5)
            draw.polygon([tip, (cx, cy - radius * 0.13), (cx, cy + radius * 0.13)],
                         fill=BLACK)
        # Sits clear above the northern spike rather than tucked against it.
        ink.text((cx, cy - radius - 6.5 * scale), "N",
                 self._font(BOLD, 8 * scale), anchor="mm")

    @staticmethod
    def _bin_sightings(sightings, width, map_height, max_spots, separation,
                       latest=None):
        """Group sightings into grid cells and return the cells to draw.

        One fin per record would be noise -- 200 overlapping glyphs. Binning
        turns the same data into something that reads as a map. Each surviving
        cell sits at the mean position of its own sightings rather than the cell
        centre, so fins land where the animals were actually reported.

        Cells are then thinned so no two fins land within `separation` pixels of
        each other, busiest first. Without this the glyphs overlap in popular
        diving areas and merge into a single blob.

        The cell holding `latest` is considered first, so it always survives the
        thinning. Nothing marks it on the map, but the caption describes that
        sighting, so it would be odd for its location to have no fin at all.
        """
        def cell_key(lon, lat):
            x, y = project(lon, lat, width, map_height)
            return (int(x // GRID_CELL_PX), int(y // GRID_CELL_PX)), x, y

        buckets = {}
        for sighting in sightings:
            key, x, y = cell_key(sighting["lon"], sighting["lat"])
            bucket = buckets.setdefault(key, {"count": 0, "x": 0.0, "y": 0.0})
            bucket["count"] += 1
            bucket["x"] += x
            bucket["y"] += y

        cells = {
            key: {"count": b["count"],
                  "x": b["x"] / b["count"],
                  "y": b["y"] / b["count"]}
            for key, b in buckets.items()
        }

        latest_key = None
        if latest:
            latest_key, _, _ = cell_key(latest["lon"], latest["lat"])

        # Busiest first, so where fins compete for space the stronger signal
        # wins -- except the newest sighting, which is considered before all of
        # them because the caption is about it.
        candidates = sorted(cells.items(),
                            key=lambda item: item[1]["count"], reverse=True)
        if latest_key is not None and latest_key in cells:
            candidates.sort(key=lambda item: item[0] != latest_key)

        minimum_squared = separation ** 2
        kept = []
        for _key, cell in candidates:
            crowded = any(
                (cell["x"] - other["x"]) ** 2 + (cell["y"] - other["y"]) ** 2
                < minimum_squared
                for other in kept
            )
            if crowded:
                continue
            kept.append(cell)
            if len(kept) >= max_spots:
                break

        return kept

    def _fin_glyph(self, scale):
        """Load and prepare the fin marker once, at the size it will be drawn.

        Every fin is identical, so this is done a single time per render. Returns
        a 1-bit mask; the caller pastes ink through it.
        """
        try:
            art = Image.open(FIN_PATH)
            art.load()
            mask = art.split()[-1] if art.mode == "RGBA" else art.convert("L")
        except OSError as error:
            raise RuntimeError(
                f"SharkMap could not load its fin glyph ({FIN_PATH}): {error}"
            ) from error

        # Trim transparent margin so the requested height is the fin itself
        # rather than the height of the artwork's canvas.
        bbox = mask.getbbox()
        if bbox:
            mask = mask.crop(bbox)

        height_px = max(6, int(round(FIN_HEIGHT_PX * scale)))
        width_px = max(6, int(round(height_px * mask.size[0] / mask.size[1])))

        # Resize then re-threshold: a soft alpha ramp would dither into grey
        # speckle on the panel, so the glyph stays hard-edged.
        return mask.resize((width_px, height_px), Image.LANCZOS).point(
            lambda alpha: 255 if alpha >= FIN_ALPHA_THRESHOLD else 0
        )

    @staticmethod
    def _draw_fins(canvas, glyph, cells, header_height, map_height):
        """Paste the prepared fin at every kept cell position."""
        if not cells:
            return

        ink = Image.new("RGB", glyph.size, FIN_COLOR)
        fin_w, fin_h = glyph.size

        for cell in cells:
            # The artwork includes its own waterline, so the fin is anchored at
            # the bottom of the glyph and nothing extra is drawn.
            left = int(round(cell["x"] - fin_w / 2))
            top = int(round(cell["y"] - fin_h)) + header_height
            left = max(0, min(canvas.width - fin_w, left))
            top = max(header_height, min(header_height + map_height - fin_h, top))

            canvas.paste(ink, (left, top), glyph)

    # ------------------------------------------------------------------
    # Caption strip
    # ------------------------------------------------------------------
    def _draw_strip(self, draw, ink, sighting, total_reported, width, height,
                    strip_height, lookback_days, stale_since, scale):
        """Species on the left, circumstances on the right, as on a chart."""
        top = height - strip_height
        draw.rectangle([0, top, width, height], fill=WHITE)
        draw.line([(0, top), (width, top)], fill=BLACK, width=2)

        margin = 34 * scale
        divider = width * 0.375

        # --- left: what it is
        name_font = self._font(BOLD, 11 * scale)
        sci_font = self._font(ITALIC, 10 * scale)
        available = divider - margin - 12 * scale

        common = (self._species_name(sighting) if sighting else "No recent sightings")
        self._letterspace(ink, (margin, top + strip_height * 0.32),
                          self._fit_text(ink, common.upper(), name_font, available),
                          name_font, spacing=1.1 * scale, centre=False)

        # Suppress the italic line when it would just repeat the headline,
        # which happens whenever iNaturalist has no common name for the taxon.
        scientific = (sighting or {}).get("scientific")
        if scientific and scientific.casefold() != common.casefold():
            ink.text((margin, top + strip_height * 0.72),
                     self._fit_text(ink, scientific, sci_font, available), sci_font)

        draw.line([(divider, top + 6 * scale), (divider, height - 6 * scale)],
                  fill=BLACK, width=1)

        # --- right: where, when, and how much to trust it
        text_x = divider + 34 * scale
        available = width - text_x - 12 * scale

        # Sized to fit three lines inside a 42px strip without crowding.
        line_font = self._font(REGULAR, 11 * scale)
        detail_font = self._font(REGULAR, 8 * scale)
        note_font = self._font(ITALIC, 8 * scale)

        ink.text((text_x, top + strip_height * 0.26),
                 self._fit_text(ink, self._where_when(sighting), line_font, available),
                 line_font)

        ink.text((text_x, top + strip_height * 0.56),
                 self._fit_text(ink, self._provenance(sighting), detail_font, available),
                 detail_font)

        ink.text((text_x, top + strip_height * 0.82),
                 self._fit_text(ink, self._note_text(total_reported, lookback_days,
                                                     stale_since),
                                note_font, available),
                 note_font)

    @staticmethod
    def _most_recent(sightings):
        """The single latest sighting, by best available timestamp.

        order_by=observed_on sorts by date only, so within the most recent day
        the API order says nothing about which sighting is newest. Sorting on
        the full timestamp here is what actually answers "most recent".
        """
        if not sightings:
            return None

        def sort_key(sighting):
            return (
                parse_iso(sighting.get("time"))
                or parse_iso(sighting.get("date"))
                or datetime.min.replace(tzinfo=timezone.utc)
            )

        return max(sightings, key=sort_key)

    @staticmethod
    def _species_name(sighting):
        """Common name if iNaturalist has one, else the scientific name."""
        return (sighting.get("common") or sighting.get("scientific")
                or "Unidentified shark")

    def _where_when(self, sighting):
        """"Cape Cod, Massachusetts  ·  18 minutes ago", degrading gracefully."""
        if not sighting:
            return ""

        parts = []
        # The resolved label when we have one, otherwise the observer's own text.
        place = sighting.get("place_label") or sighting.get("place")
        if place:
            parts.append(self._truncate(" ".join(place.split()), PLACE_MAX_CHARS))

        when = self._format_when(sighting)
        if when:
            parts.append(when)

        return "  ·  ".join(parts) if parts else "Location not recorded"

    @staticmethod
    def _provenance(sighting):
        """How much the position can be trusted.

        Every record we plot is research grade, because the query filters on it.
        Coordinates are deliberately randomised by iNaturalist for threatened
        species, so saying so is more honest than implying a precise fix.
        """
        parts = ["Research grade"]
        if sighting and sighting.get("obscured"):
            parts.append("location obscured by iNaturalist")
        else:
            parts.append("community-verified identification")
        return "  ·  ".join(parts)

    @staticmethod
    def _truncate(text, limit):
        """Shorten free text to `limit` characters, on a word break if possible."""
        if len(text) <= limit:
            return text
        clipped = text[:limit].rstrip(" ,;:-")
        tail_start = int(limit * 0.6)
        if " " in clipped[tail_start:]:
            clipped = clipped[:clipped.rfind(" ")]
        return clipped.rstrip(" ,;:-") + "…"

    @staticmethod
    def _format_when(sighting):
        """Relative time when a real timestamp exists, otherwise a plain date.

        time_observed_at is null for a small share of records. Inventing a time
        for those would be a lie, so they get a date instead.
        """
        stamp = parse_iso(sighting.get("time"))
        if stamp:
            seconds = (datetime.now(timezone.utc) - stamp).total_seconds()
            minutes = int(seconds // 60)
            if seconds < 0 or minutes < 2:
                return "just now"
            if minutes < 60:
                return f"{minutes} minutes ago"
            hours = minutes // 60
            if hours < 24:
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            days = hours // 24
            if days < 30:
                return f"{days} day{'s' if days != 1 else ''} ago"
            return format_date(stamp)

        date = parse_iso(sighting.get("date"))
        return format_date(date) if date else ""

    @staticmethod
    def _note_text(total_reported, lookback_days, stale_since):
        """The framing line.

        This is citizen-science data: it maps where people report sharks, not
        where sharks are. Saying "reported" keeps the display from overclaiming.
        """
        window = "24 hours" if lookback_days == 1 else f"{lookback_days} days"
        plural = "sighting" if total_reported == 1 else "sightings"
        note = (f"Most recent of {total_reported} {plural} reported "
                f"in the last {window}")

        if stale_since:
            hours = int((datetime.now(timezone.utc) - stale_since).total_seconds() // 3600)
            note += (f"  ·  cached {hours}h ago, no connection" if hours >= 1
                     else "  ·  cached, no connection")

        return note


def parse_iso(value):
    """Parse an ISO 8601 string to an aware datetime, or None.

    Handles the trailing Z that datetime.fromisoformat rejects on older
    Pythons, and dates with no time. Anything unparseable returns None so
    callers can fall back rather than raise.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_date(stamp):
    """Format a date as e.g. "6 Aug 2026", without platform-specific codes."""
    return f"{stamp.day} {stamp:%b %Y}"
