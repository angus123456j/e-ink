"""SharkMap: a nautical-atlas world map of recent shark sightings.

Drawn with Pillow rather than HTML, because Chromium costs 40+ seconds of boot
and render time on a Pi Zero 2 W. Everything static -- ocean tint, land,
coastline, graticule, place names -- is baked into world_map.png at build time,
so a refresh only draws the header, the fins, and the caption.

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
FIN_PATH = os.path.join(PLUGIN_DIR, "fin.png")
FONT_DIR = os.path.join(PLUGIN_DIR, "fonts")

REGULAR = "LiberationSerif-Regular.ttf"
BOLD = "LiberationSerif-Bold.ttf"
ITALIC = "LiberationSerif-Italic.ttf"

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
BLUE = (0, 0, 255)

# --- Fin layout -----------------------------------------------------------
# Sightings are binned into cells roughly one fin wide, so at most one fin
# lands per cell and they do not pile up on each other.
GRID_CELL_PX = 22
FIN_MIN_PX = 11           # a single sighting
FIN_MAX_PX = 18           # the busiest cell
PLACE_MAX_CHARS = 42      # place_guess is free text and can run long
HIGHLIGHT_LABEL_CHARS = 24


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
    def _letterspace(draw, xy, text, font, fill, spacing=1.5, centre=True):
        """Draw text with extra tracking, the way atlas labels are set.

        Pillow has no letter-spacing, so characters are placed individually.
        Returns the total advance.
        """
        widths = [draw.textlength(ch, font=font) for ch in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = xy[0] - total / 2 if centre else xy[0]
        for char, width in zip(text, widths):
            draw.text((x, xy[1]), char, font=font, fill=fill, anchor="lm")
            x += width + spacing
        return total

    @staticmethod
    def _fit_text(draw, text, font, max_width):
        """Trim text with an ellipsis until it fits `max_width` pixels."""
        if draw.textlength(text, font=font) <= max_width:
            return text
        trimmed = text
        while trimmed and draw.textlength(trimmed + "…", font=font) > max_width:
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
        self._draw_header(draw, width, header_height, scale)

        cells = self._bin_sightings(sightings, width, map_height, max_spots)
        self._draw_fins(canvas, cells, header_height, map_height, scale)

        latest = self._most_recent(sightings)
        if latest:
            self._draw_highlight(draw, latest, width, map_height,
                                 header_height, scale)

        if show_strip:
            self._draw_strip(draw, latest, total_reported, width, height,
                             strip_height, lookback_days, stale_since, scale)

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

    def _draw_header(self, draw, width, header_height, scale):
        """Title block: name, subtitle, compass rose and the source credit."""
        draw.rectangle([0, 0, width, header_height], fill=WHITE)

        title_font = self._font(BOLD, 22 * scale)
        sub_font = self._font(REGULAR, 9 * scale)

        self._letterspace(draw, (width / 2, header_height * 0.40),
                          "SHARK SIGHTINGS", title_font, BLACK, spacing=3.0 * scale)
        self._letterspace(draw, (width / 2, header_height * 0.75),
                          "LIVE MARITIME OBSERVATIONS", sub_font, BLUE,
                          spacing=2.4 * scale)

        draw.line([(0, header_height - 1), (width, header_height - 1)],
                  fill=BLACK, width=1)

        self._draw_compass(draw, width * 0.055, header_height * 0.48,
                           header_height * 0.32, scale)

        # The source credit iNaturalist's terms ask for, given the prominence a
        # decorative brand block would otherwise take.
        credit_x = width - 150 * scale
        self._letterspace(draw, (credit_x, header_height * 0.36),
                          "DATA: iNATURALIST", self._font(BOLD, 7 * scale),
                          BLACK, spacing=1.0 * scale, centre=False)
        draw.text((credit_x, header_height * 0.66), "research-grade observations",
                  font=self._font(ITALIC, 7 * scale), fill=BLUE, anchor="lm")

    def _draw_compass(self, draw, cx, cy, radius, scale):
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
                         fill=BLUE)
        draw.text((cx, cy - radius - 6 * scale), "N",
                  font=self._font(BOLD, 7 * scale), fill=BLACK, anchor="mm")

    @staticmethod
    def _bin_sightings(sightings, width, map_height, max_spots):
        """Group sightings into grid cells, keeping the busiest `max_spots`.

        One fin per record would be noise -- 200 overlapping glyphs. Binning to
        roughly fin-sized cells turns the same data into something that reads as
        a map. Each surviving cell sits at the mean position of its own
        sightings rather than the cell centre, so fins land where the animals
        were actually reported.
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

        # Busiest cells win when there are more occupied cells than fins
        # allowed, so the map keeps the strongest signal.
        cells.sort(key=lambda cell: cell["count"], reverse=True)
        return cells[:max_spots]

    def _draw_fins(self, canvas, cells, header_height, map_height, scale):
        """Paste a fin per cell, sized by how many sightings it represents."""
        if not cells:
            return

        try:
            glyph = Image.open(FIN_PATH)
            glyph.load()
            mask = glyph.split()[-1] if glyph.mode == "RGBA" else glyph.convert("L")
        except OSError as error:
            raise RuntimeError(
                f"SharkMap could not load its fin glyph ({FIN_PATH}): {error}"
            ) from error

        busiest = max(cell["count"] for cell in cells)
        aspect = mask.size[0] / mask.size[1]

        # Draw ascending so the biggest fins finish on top.
        for cell in sorted(cells, key=lambda c: c["count"]):
            height_px = self._fin_height(cell["count"], busiest, scale)
            width_px = max(6, int(round(height_px * aspect)))

            # Resize then re-threshold: a soft alpha ramp would dither into
            # grey speckle on the panel, so the silhouette stays hard-edged.
            shape = mask.resize((width_px, height_px), Image.LANCZOS).point(
                lambda alpha: 255 if alpha >= 128 else 0
            )

            # The glyph already contains its own waterline, so the fin is
            # anchored at the bottom of the glyph and nothing extra is drawn.
            left = int(round(cell["x"] - width_px / 2))
            top = int(round(cell["y"] - height_px)) + header_height
            left = max(0, min(canvas.width - width_px, left))
            top = max(header_height,
                      min(header_height + map_height - height_px, top))

            canvas.paste(Image.new("RGB", shape.size, BLACK), (left, top), shape)

    @staticmethod
    def _fin_height(count, busiest, scale):
        """Fin height in pixels, scaled by cell count.

        Square-root scaling: a cell with 25 sightings should read as busier
        than one with 1, but not 25 times taller.
        """
        low, high = FIN_MIN_PX * scale, FIN_MAX_PX * scale
        if busiest <= 1:
            return max(6, int(round(low)))
        share = (count ** 0.5 - 1) / (busiest ** 0.5 - 1)
        return max(6, int(round(low + share * (high - low))))

    def _draw_highlight(self, draw, sighting, width, map_height,
                        header_height, scale):
        """Ring the newest sighting and name its place.

        Red because the palette has no cyan, and because red also carries the
        right meaning: this is the one the caption is describing.
        """
        x, y = project(sighting["lon"], sighting["lat"], width, map_height)
        y += header_height
        radius = max(6, 12 * scale)

        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     outline=RED, width=1)

        label = self._highlight_label(sighting.get("place"))
        if not label:
            return

        font = self._font(BOLD, 7 * scale)
        gap = radius + 5 * scale
        # Flip the label to the left near the right edge so it cannot run off.
        if x + gap + draw.textlength(label, font=font) * 1.2 > width:
            self._letterspace(draw, (x - gap, y - 6 * scale), label, font, RED,
                              spacing=0.8, centre=False)
        else:
            self._letterspace(draw, (x + gap, y - 6 * scale), label, font, RED,
                              spacing=0.8, centre=False)

    @staticmethod
    def _highlight_label(place):
        """Short, upper-case place name for the map annotation.

        place_guess is free text, often a long administrative chain. The first
        two components carry the useful part.
        """
        if not place:
            return ""
        parts = [p.strip() for p in place.split(",") if p.strip()]
        label = ", ".join(parts[:2]) if parts else place.strip()
        if len(label) > HIGHLIGHT_LABEL_CHARS:
            label = label[:HIGHLIGHT_LABEL_CHARS].rstrip(" ,") + "…"
        return label.upper()

    # ------------------------------------------------------------------
    # Caption strip
    # ------------------------------------------------------------------
    def _draw_strip(self, draw, sighting, total_reported, width, height,
                    strip_height, lookback_days, stale_since, scale):
        """Species on the left, circumstances on the right, as on a chart."""
        top = height - strip_height
        draw.rectangle([0, top, width, height], fill=WHITE)
        draw.line([(0, top), (width, top)], fill=BLACK, width=2)

        margin = 22 * scale
        divider = width * 0.375

        # --- left: what it is
        name_font = self._font(BOLD, 12 * scale)
        sci_font = self._font(ITALIC, 11 * scale)
        available = divider - margin * 2

        common = (self._species_name(sighting) if sighting else "No recent sightings")
        self._letterspace(draw, (margin, top + strip_height * 0.33),
                          self._fit_text(draw, common.upper(), name_font, available),
                          name_font, BLACK, spacing=1.1 * scale, centre=False)

        scientific = (sighting or {}).get("scientific")
        if scientific:
            draw.text((margin, top + strip_height * 0.66),
                      self._fit_text(draw, scientific, sci_font, available),
                      font=sci_font, fill=BLUE, anchor="lm")

        draw.line([(divider, top + 10 * scale), (divider, height - 10 * scale)],
                  fill=BLUE, width=1)

        # --- right: where, when, and how much to trust it
        text_x = divider + 22 * scale
        available = width - text_x - margin * 0.5

        line_font = self._font(REGULAR, 12 * scale)
        detail_font = self._font(REGULAR, 9 * scale)
        note_font = self._font(ITALIC, 9 * scale)

        draw.text((text_x, top + strip_height * 0.28),
                  self._fit_text(draw, self._where_when(sighting), line_font, available),
                  font=line_font, fill=BLACK, anchor="lm")

        draw.text((text_x, top + strip_height * 0.56),
                  self._fit_text(draw, self._provenance(sighting), detail_font, available),
                  font=detail_font, fill=BLACK, anchor="lm")

        draw.text((text_x, top + strip_height * 0.81),
                  self._fit_text(draw, self._note_text(total_reported, lookback_days,
                                                       stale_since),
                                 note_font, available),
                  font=note_font, fill=BLUE, anchor="lm")

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
        place = sighting.get("place")
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
        note = (f"Most recent of {total_reported} sightings reported "
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
