"""SharkMap: a world map of recent shark sightings reported to iNaturalist.

Drawn with Pillow rather than HTML, because Chromium costs 40+ seconds of boot
and render time on a Pi Zero 2 W. The whole render is a basemap paste plus a
few dozen small shapes, so it finishes in well under a second.

Data comes from the iNaturalist API v1, which needs no key. See fetch_sightings
for the query and the traps in it.

Sightings are drawn as plain dots for now. fin.png is already generated and
sitting next to this file for a later pass at the artwork; nothing here uses it
yet, deliberately -- the point of this version is that the data path works.
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
    STRIP_HEIGHT,
    in_bounds,
    project,
)
from utils.http_client import get_http_session

logger = logging.getLogger(__name__)

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
WORLD_MAP_PATH = os.path.join(PLUGIN_DIR, "world_map.png")
FONT_DIR = os.path.join(PLUGIN_DIR, "fonts")

# --- iNaturalist ----------------------------------------------------------
INAT_OBSERVATIONS_URL = "https://api.inaturalist.org/v1/observations"
INAT_AUTOCOMPLETE_URL = "https://api.inaturalist.org/v1/taxa/autocomplete"

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

# --- Spectra 6 palette ----------------------------------------------------
# The panel shows only these six. Saturated values survive dithering; muted
# ones turn to mud.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# --- Spot layout ----------------------------------------------------------
# Observations are binned into square cells so 200 records do not become 200
# overlapping dots. Cells are a little wider than the largest dot.
GRID_CELL_PX = 22
SPOT_MIN_RADIUS = 4      # a single sighting
SPOT_MAX_RADIUS = 9      # the busiest cell
HOTSPOT_COUNT = 3        # busiest cells drawn in red
PLACE_MAX_CHARS = 40     # place_guess is free text and can run long


class SharkMap(BasePlugin):
    """Paints recent shark sightings onto an equirectangular world map."""

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
        which can exceed the page we fetched.
        """
        cache_path = self._cache_path(lookback_days, species_filter)

        try:
            sightings, total_reported = self.fetch_sightings(
                lookback_days, species_filter)
            if sightings:
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
            # Free text typed by the observer. Often absent, sometimes long.
            "place": record.get("place_guess") or None,
            # time_observed_at is often null, in which case only the date is
            # known and we must not invent a time.
            "time": record.get("time_observed_at") or None,
            "date": record.get("observed_on") or None,
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
    # Rendering
    # ------------------------------------------------------------------
    def _render(self, dimensions, sightings, total_reported, max_spots,
                show_strip, lookback_days, stale_since):
        width, height = int(dimensions[0]), int(dimensions[1])

        # Scale the strip with the canvas so non-native resolutions stay
        # proportional. Zero when the caption is switched off.
        strip_height = 0
        if show_strip:
            strip_height = max(28, round(height * STRIP_HEIGHT / CANVAS_HEIGHT))
        map_height = height - strip_height

        canvas = Image.new("RGB", (width, height), WHITE)
        canvas.paste(self._basemap(width, map_height), (0, 0))

        cells = self._bin_sightings(sightings, width, map_height, max_spots)
        self._draw_spots(canvas, cells)

        if show_strip:
            self._draw_strip(canvas, sightings, total_reported, width, height,
                             strip_height, lookback_days, stale_since)

        return canvas

    def _basemap(self, width, map_height):
        """Load the basemap, resizing only if the panel is not 800x480.

        NEAREST is deliberate: it cannot invent intermediate colours, so the
        image stays exactly three flat tones at any size. A smoother filter
        would blend ocean into land, and the Inky dither would turn those
        blended edges into noise.
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

    @staticmethod
    def _bin_sightings(sightings, width, map_height, max_spots):
        """Group sightings into grid cells, keeping the busiest `max_spots`.

        One dot per record would be noise -- 200 overlapping marks. Binning to
        cells turns the same data into something readable. Each surviving cell
        is placed at the mean position of its own sightings rather than the
        cell centre, so dots sit where the animals were actually reported.
        """
        buckets = {}
        for sighting in sightings:
            x, y = project(sighting["lon"], sighting["lat"], width, map_height)
            key = (int(x // GRID_CELL_PX), int(y // GRID_CELL_PX))
            bucket = buckets.setdefault(key, {"count": 0, "x": 0.0, "y": 0.0})
            bucket["count"] += 1
            bucket["x"] += x
            bucket["y"] += y

        cells = [
            {"count": b["count"], "x": b["x"] / b["count"], "y": b["y"] / b["count"]}
            for b in buckets.values()
        ]

        # Busiest cells win when there are more occupied cells than dots
        # allowed, so the map keeps the strongest signal.
        cells.sort(key=lambda cell: cell["count"], reverse=True)
        return cells[:max_spots]

    def _draw_spots(self, canvas, cells):
        """Draw one dot per cell, sized by count, busiest few in red.

        Each dot gets a white outline so it stays legible over the land fill as
        well as over the white ocean.
        """
        if not cells:
            return

        draw = ImageDraw.Draw(canvas)
        busiest = max(cell["count"] for cell in cells)

        hotspots = {
            id(cell) for cell in cells[:HOTSPOT_COUNT] if cell["count"] > 1
        }

        # Draw ascending so the biggest dots end up on top.
        for cell in sorted(cells, key=lambda c: c["count"]):
            radius = self._spot_radius(cell["count"], busiest)
            colour = RED if id(cell) in hotspots else BLACK
            box = [
                cell["x"] - radius, cell["y"] - radius,
                cell["x"] + radius, cell["y"] + radius,
            ]
            draw.ellipse(box, fill=colour, outline=WHITE, width=2)

    @staticmethod
    def _spot_radius(count, busiest):
        """Dot radius in pixels, scaled by cell count.

        Square-root scaling: a cell with 25 sightings should read as busier
        than one with 1, but not 25 times wider.
        """
        if busiest <= 1:
            return SPOT_MIN_RADIUS
        share = (count ** 0.5 - 1) / (busiest ** 0.5 - 1)
        return int(round(SPOT_MIN_RADIUS + share * (SPOT_MAX_RADIUS - SPOT_MIN_RADIUS)))

    # ------------------------------------------------------------------
    # Caption strip
    # ------------------------------------------------------------------
    def _draw_strip(self, canvas, sightings, total_reported, width, height,
                    strip_height, lookback_days, stale_since):
        """Bottom strip: the single most recent sighting, plus honest framing."""
        top = height - strip_height
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, top, width, height], fill=WHITE)
        draw.line([(0, top), (width, top)], fill=BLACK, width=2)

        headline_font = self._font("LiberationSerif-Bold.ttf",
                                   max(13, int(strip_height * 0.40)))
        note_font = self._font("LiberationSerif-Italic.ttf",
                               max(10, int(strip_height * 0.26)))

        margin = max(8, int(width * 0.0125))
        available = width - margin * 2

        headline = self._headline_text(self._most_recent(sightings))
        note = self._note_text(total_reported, lookback_days, stale_since)

        # Two lines inside the strip: the sighting, then the framing.
        draw.text((margin, top + strip_height * 0.30),
                  self._fit_text(draw, headline, headline_font, available),
                  font=headline_font, fill=BLACK, anchor="lm")
        draw.text((margin, top + strip_height * 0.74),
                  self._fit_text(draw, note, note_font, available),
                  font=note_font, fill=BLACK, anchor="lm")

    def _font(self, filename, size):
        """Load a bundled font.

        Liberation Serif ships with the plugin because it is metrically
        identical to Times New Roman but freely redistributable, and Raspberry
        Pi OS has no Times New Roman to fall back on.
        """
        path = os.path.join(FONT_DIR, filename)
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            logger.warning("SharkMap: missing font %s, using PIL default", path)
            return ImageFont.load_default()

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

    def _headline_text(self, sighting):
        """"Tiger Shark - Oahu, HI, USA - 3 hours ago", degrading gracefully."""
        if not sighting:
            return "No recent sightings"

        parts = [self._species_name(sighting)]

        place = sighting.get("place")
        if place:
            parts.append(self._truncate(" ".join(place.split()), PLACE_MAX_CHARS))

        when = self._format_when(sighting)
        if when:
            parts.append(when)

        return "  ·  ".join(parts)

    @staticmethod
    def _species_name(sighting):
        """Common name if iNaturalist has one, else the scientific name."""
        return sighting.get("common") or sighting.get("scientific") or "Unidentified shark"

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

        This is citizen-science data: it maps where people dive and snorkel,
        not where sharks are, and iNaturalist deliberately fuzzes coordinates
        for threatened species. Saying "reported to iNaturalist" keeps the
        display from overclaiming, and credits the source as their terms ask.
        """
        window = "24 hours" if lookback_days == 1 else f"{lookback_days} days"
        note = (f"Most recent of {total_reported} sightings reported to "
                f"iNaturalist in the last {window}")

        if stale_since:
            hours = int((datetime.now(timezone.utc) - stale_since).total_seconds() // 3600)
            note += f"  ·  cached {hours}h ago, no connection" if hours >= 1 \
                else "  ·  cached, no connection"

        return note

    @staticmethod
    def _fit_text(draw, text, font, max_width):
        """Trim text with an ellipsis until it fits `max_width` pixels."""
        if draw.textlength(text, font=font) <= max_width:
            return text
        trimmed = text
        while trimmed and draw.textlength(trimmed + "…", font=font) > max_width:
            trimmed = trimmed[:-1]
        return trimmed.rstrip() + "…"


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
