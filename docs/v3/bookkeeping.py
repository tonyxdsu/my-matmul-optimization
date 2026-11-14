#!/usr/bin/env python3
"""Draws the chart in the v3 section of README.md: the instructions in one row update (one i and
one k of multiplyBlock, 48 multiply-adds at tile 48) in v2 and in v3, split into the vector loop
and the setup around it.

    python3 docs/v3/bookkeeping.py      # writes bookkeeping.png next to this file

The counts come from GCC 13.3's output for n = 2000, tile 48, with the Makefile's flags, read
along the path a row update takes (`objdump -d -C`). GROUPS repeats the README's table, so the
two can be checked against each other.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
# The palette of bench/plot.py's dark charts. Blue is the useful work, orange the overhead,
# as in docs/v2/alignment.png.
COLORS = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "loop": "#3987e5", "setup": "#d95926",
}
WIDTH, HEIGHT, DPI = 8.0, 2.5, 200    # inches; the PNG is 1600 x 500 pixels
LEFT = 0.32                           # left and right margin, inches
BAR_X = 0.78                          # where the bars start, inches
BAR_H = 0.26                          # bar thickness, inches (26 px at GitHub's usual scale)
GAP = 0.02                            # surface gap between the two segments, inches
RADIUS = 0.04                         # rounding of a bar's far end, inches
SCALE = 0.048                         # inches per instruction

# Instructions per row update, by what they do. The vector loop is 6 passes of 6 instructions.
GROUPS = {
    "v2": {"vector loop": 36, "load A[i][k] into 8 lanes": 1, "overlap checks": 14,
           "reloads and spills": 9, "address arithmetic": 7, "length checks": 8, "loop control": 6,
           "padding (nop)": 0},
    "v3": {"vector loop": 36, "load A[i][k] into 8 lanes": 2, "overlap checks": 5,
           "reloads and spills": 2, "address arithmetic": 3, "length checks": 6, "loop control": 3,
           "padding (nop)": 6},
}
# Instructions per multiply-add from the hardware counters (bench/results/v2.json and v3.json).
MEASURED = {"v2": 1.70, "v3": 1.33}


def text_width(fig, artist):
    """Width of a drawn text artist, in inches."""
    return artist.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi


def bar(fig, ax, y, label):
    """Draws one version's bar centred on y: the vector loop, a gap, then the setup."""
    counts = GROUPS[label]
    loop = counts["vector loop"]
    setup = sum(counts.values()) - loop
    ax.text(LEFT, y, label, ha="left", va="center", fontsize=10, fontweight="bold", color=COLORS["ink"])

    # The vector loop sits on the baseline: square at both ends, the setup segment follows.
    loop_w = loop * SCALE
    ax.add_patch(Rectangle((BAR_X, y - BAR_H / 2), loop_w - GAP / 2, BAR_H, facecolor=COLORS["loop"],
                           linewidth=0))
    ax.text(BAR_X + loop_w / 2, y, f"{loop}", ha="center", va="center", fontsize=9, fontweight="bold",
            color=COLORS["surface"])

    # The setup segment: rounded at the far end only, so a square patch covers its near corners.
    start = BAR_X + loop_w + GAP / 2
    setup_w = setup * SCALE - GAP / 2
    ax.add_patch(FancyBboxPatch((start, y - BAR_H / 2), setup_w, BAR_H,
                                boxstyle=f"round,pad=0,rounding_size={RADIUS}", facecolor=COLORS["setup"],
                                linewidth=0))
    ax.add_patch(Rectangle((start, y - BAR_H / 2), setup_w - RADIUS, BAR_H, facecolor=COLORS["setup"],
                           linewidth=0))
    ax.text(start + setup_w / 2, y, f"{setup}", ha="center", va="center", fontsize=9, fontweight="bold",
            color=COLORS["surface"])

    # The total at the tip, then what the hardware counters measured.
    tip = ax.text(start + setup_w + 0.1, y, f"{loop + setup}", ha="left", va="center", fontsize=10,
                  fontweight="bold", color=COLORS["ink"])
    ax.text(start + setup_w + 0.1 + text_width(fig, tip) + 0.06, y,
            f"instructions, {MEASURED[label]:.2f} per multiply-add measured", ha="left", va="center",
            fontsize=8.4, color=COLORS["ink2"])


def legend(ax, y):
    """One row: a swatch beside each segment's name, in bar order."""
    x = LEFT
    for color, name in ((COLORS["loop"], "vector loop: 6 passes of 6 instructions, the same in both"),
                        (COLORS["setup"], "setup around it, run once per row update")):
        ax.add_patch(FancyBboxPatch((x, y - 0.055), 0.11, 0.11, boxstyle="round,pad=0,rounding_size=0.02",
                                    facecolor=color, linewidth=0))
        label = ax.text(x + 0.17, y, name, ha="left", va="center", fontsize=8.4, color=COLORS["ink"])
        x += 0.17 + text_width(ax.figure, label) + 0.36


def main():
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor=COLORS["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    ax.text(LEFT, 2.27, "Hoisting cut the setup around the vector loop from 45 to 27 instructions",
            fontsize=11.5, fontweight="bold", color=COLORS["ink"], va="center")
    ax.text(LEFT, 2.04, "Instructions in one row update (one i and one k: 48 multiply-adds) at n = 2000, "
                        "tile 48, read from GCC's output.", fontsize=7.8, color=COLORS["ink2"], va="center")
    legend(ax, 1.71)
    bar(fig, ax, 1.27, "v2")
    bar(fig, ax, 0.77, "v3")
    ax.text(LEFT, 0.33, "v3's setup includes 6 nops: padding that starts the vector loop on a 64-byte boundary "
                        "(-falign-loops=64). Without them it is 21.", fontsize=7.8, color=COLORS["ink2"],
            va="center")

    fig.savefig(OUT / "bookkeeping.png", facecolor=COLORS["surface"])
    plt.close(fig)
    print(f"wrote {OUT / 'bookkeeping.png'}")


if __name__ == "__main__":
    main()
