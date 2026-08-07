"""Preview how a render will actually look on an Inky Impression (Spectra 6).

Everything in mock_display_output/ is the image InkyPi *sends* to the display,
not what the display shows. On real hardware the Inky library then quantises and
dithers it. That step is where flat design survives and photographic or pastel
design falls apart, so it is worth seeing before the panel arrives.

This reproduces the driver's own maths, taken from inky/inky_e673.py (the 2025
Spectra 6 7.3in):

  - the palette is a blend between a SATURATED and a DESATURATED set, controlled
    by the `saturation` argument, which InkyPi passes from its
    image_settings.inky_saturation and defaults to 0.5
  - the image is then quantized to six colours with Floyd-Steinberg dithering
  - the one exception is a P-mode image already holding exactly six colours,
    which the driver maps with no dithering at all

Note the consequence of the blend: at the default saturation the driver is not
aiming at pure colours. "Blue" is (30,29,174) and even "white" is (208,209,210).
A preview against pure RGB primaries is therefore misleading, which is the
mistake this script exists to avoid.

Usage:

    python scripts/preview_spectra6.py mock_display_output/latest.png
    python scripts/preview_spectra6.py latest.png --saturation 0.8
    python scripts/preview_spectra6.py latest.png --sweep
"""

import argparse
import os
import sys

from PIL import Image

# Straight from inky/inky_e673.py. Six usable colours; the driver's tables carry
# a seventh "clean" entry which is not a displayable colour.
DESATURATED_PALETTE = [
    [0, 0, 0],          # black
    [255, 255, 255],    # white
    [255, 255, 0],      # yellow
    [255, 0, 0],        # red
    [0, 0, 255],        # blue
    [0, 255, 0],        # green
]
SATURATED_PALETTE = [
    [0, 0, 0],
    [161, 164, 165],
    [208, 190, 71],
    [156, 72, 75],
    [61, 59, 94],
    [58, 91, 70],
]
COLOUR_NAMES = ["black", "white", "yellow", "red", "blue", "green"]

# What InkyPi passes unless the settings page says otherwise.
DEFAULT_SATURATION = 0.5


def blended_palette(saturation):
    """The palette the driver quantises against, for a given saturation."""
    palette = []
    for index in range(len(DESATURATED_PALETTE)):
        saturated = [c * saturation for c in SATURATED_PALETTE[index]]
        desaturated = [c * (1.0 - saturation) for c in DESATURATED_PALETTE[index]]
        palette.append(tuple(int(s + d) for s, d in zip(saturated, desaturated)))
    return palette


def palette_image(colours):
    flat = []
    for colour in colours:
        flat.extend(colour)
    flat.extend([0, 0, 0] * (256 - len(colours)))
    image = Image.new("P", (1, 1))
    image.putpalette(flat)
    return image


def render_to_panel(image, saturation, appearance="pure"):
    """Quantise and dither exactly as the driver would.

    Two different palettes are in play, and confusing them makes the preview
    misleading:

      - The BLENDED palette is what the driver quantises *against*. It decides
        which source colour becomes which of the six inks. This is the one that
        depends on `saturation`.
      - The pigment the panel then lays down is its own. It is not the blended
        value. Rendering the result using the blended numbers paints "white" as
        a mid grey and makes everything look far weaker than it will be.

    So: choose indices with the blended palette, then show them with the pure
    palette. `appearance="blended"` keeps the old behaviour for comparison.
    """
    colours = blended_palette(saturation)
    quantised = image.convert("RGB").quantize(
        colors=len(colours), palette=palette_image(colours),
        dither=Image.Dither.FLOYDSTEINBERG,
    )

    if appearance == "pure":
        # Repaint the same indices with the panel's actual colours. Real e-ink
        # white is a shade off paper and its black is closer to charcoal, so the
        # truth sits between this and the blended render -- but this is much the
        # closer of the two.
        flat = []
        for colour in DESATURATED_PALETTE:
            flat.extend(colour)
        flat.extend([0, 0, 0] * (256 - len(DESATURATED_PALETTE)))
        quantised.putpalette(flat)

    return quantised.convert("RGB"), colours


def report(image, panel, colours, appearance="pure"):
    """Say how much of the source had to be dithered, and what it became."""
    total = image.width * image.height
    source_colours = image.convert("RGB").getcolors(maxcolors=10 ** 7)

    exact = sum(count for count, colour in (source_colours or [])
                if tuple(colour) in colours)
    print(f"  source colours          : {len(source_colours) if source_colours else 'very many'}")
    print(f"  already exactly on-palette: {exact / total * 100:5.2f}%  "
          f"(the rest is what gets dithered)")

    counts = {colour: count for count, colour in panel.getcolors(maxcolors=64)}
    shown = ([tuple(c) for c in DESATURATED_PALETTE] if appearance == "pure"
             else colours)
    print("  panel output mix:")
    for name, colour in zip(COLOUR_NAMES, shown):
        share = counts.get(colour, 0) / total * 100
        if share:
            print(f"    {name:7s} {str(colour):>17s}  {share:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="PNG to preview, e.g. mock_display_output/latest.png")
    parser.add_argument("--saturation", type=float, default=DEFAULT_SATURATION,
                        help=f"driver saturation, 0.0-1.0 (default {DEFAULT_SATURATION}, "
                             "matching InkyPi)")
    parser.add_argument("--sweep", action="store_true",
                        help="write a comparison across several saturation values")
    parser.add_argument("--appearance", choices=("pure", "blended"), default="pure",
                        help="'pure' shows the panel's own inks (default and more "
                             "realistic); 'blended' paints the driver's "
                             "quantisation targets, which look washed out")
    parser.add_argument("--out", help="output path (default: alongside the input)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        parser.error(f"no such file: {args.image}")

    source = Image.open(args.image).convert("RGB")
    print(f"{args.image}  {source.size}")

    if args.sweep:
        levels = (0.0, 0.3, 0.5, 0.7, 1.0)
        panels = []
        for level in levels:
            print(f"\nsaturation {level}")
            panel, colours = render_to_panel(source, level, args.appearance)
            report(source, panel, colours, args.appearance)
            panels.append((level, panel))

        gap = 8
        sheet = Image.new("RGB", (source.width,
                                  (source.height + gap) * len(panels) - gap),
                          (110, 110, 110))
        for index, (_level, panel) in enumerate(panels):
            sheet.paste(panel, (0, index * (source.height + gap)))
        out = args.out or args.image.replace(".png", "_sweep.png")
        sheet.save(out)
        print(f"\nwrote {out}")
        print("order top to bottom: " + ", ".join(f"saturation {lv}" for lv, _ in panels))
        return 0

    panel, colours = render_to_panel(source, args.saturation, args.appearance)
    print(f"\nsaturation {args.saturation}")
    report(source, panel, colours, args.appearance)

    out = args.out or args.image.replace(".png", "_panel.png")
    panel.save(out)
    print(f"\nwrote {out}")

    # Side by side, so the difference is obvious rather than remembered.
    gap = 8
    pair = Image.new("RGB", (source.width, source.height * 2 + gap), (110, 110, 110))
    pair.paste(source, (0, 0))
    pair.paste(panel, (0, source.height + gap))
    pair_out = out.replace(".png", "_compare.png")
    pair.save(pair_out)
    print(f"wrote {pair_out}  (top: what InkyPi sends, bottom: what the panel shows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
