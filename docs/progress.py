#!/usr/bin/env python3
"""Draws the chart at the top of README.md: FLOP per cycle of every version on one thread, with
OpenBLAS and the core's limit for scale.

    python3 docs/progress.py      # writes progress.png next to this file

The values are the medians of `FLOP/cycle` at n = 2000, tile 48, in bench/results/v0.json to
v3.json (naive and OpenBLAS read the same in all four). When a version lands, add its row to ROWS
and update TITLE.

v4* is the exception while v4 is in progress: its kernel only handles whole register blocks, so
its value comes from bench/results/v4-partial.json and its step over v3 from v3-same-sizes.json,
both at n = 2016, tile 192, and FOOTNOTE says so. When v4 handles edges, replace that row with
the number at n = 2000 and set FOOTNOTE to None.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
# The palette of bench/plot.py's dark charts. Blue is this project's code; the reference library
# is a neutral gray, so it reads as context.
COLORS = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "edge": "#3a3a37",
    "ours": "#3987e5", "reference": "#898781",
}
PEAK = 32.0                           # FLOP/cycle: two 512-bit FMAs per cycle, 8 doubles each
TITLE = "Five changes took one thread from 0.80 to 29.1* FLOP per cycle"
SUBTITLE = ("Multiplies and adds per cycle on one thread at n = 2000, tile 48 (v4*: see the note). "
            "The step over the version before is in gray.")
# (name, what changed, FLOP/cycle, step over the row above measured in the same run, kind)
ROWS = [
    ("naive", "textbook i-j-k loop", 0.80, None, "ours"),
    ("v0", "cache blocking", 1.49, 1.85, "ours"),
    ("v1", "i-k-j order, vectorized", 2.58, 1.73, "ours"),
    ("v2", "aligned storage", 4.31, 1.67, "ours"),
    ("v3", "hoisted invariants", 6.10, 1.42, "ours"),
    ("v4*", "register blocking", 29.1, 3.89, "ours"),
    ("OpenBLAS", "reference library", 26.0, None, "reference"),
]
# Printed under the chart, or None.
FOOTNOTE = ("* In progress: v4 handles whole register blocks only, so it is measured at n = 2016, tile 192, where\n"
            "v3 runs at 7.48 (the 3.89x) and OpenBLAS at 25.8. Other sizes need the edge code that is still to come.")

WIDTH, HEIGHT, DPI = 8.0, 3.95, 200   # inches; the PNG is 1600 x 790 pixels
LEFT = 0.32                           # left margin, inches
NAME_X = LEFT                         # version names
WHAT_X = 1.34                         # what changed
BAR_X = 3.0                           # where the bars start, inches
SCALE = (7.3 - BAR_X) / PEAK          # inches per FLOP/cycle: the limit lands at x = 7.3
BAR_H = 0.2                           # bar thickness, inches
RADIUS = 0.04                         # rounding of a bar's far end, inches
TOP_Y, PITCH = 3.13, 0.32             # centre of the first bar, and the distance between bars


def bar(ax, y, value, color):
    """One bar from the baseline: square at the baseline, rounded at the far end."""
    width = value * SCALE
    radius = min(RADIUS, width / 2)
    ax.add_patch(FancyBboxPatch((BAR_X, y - BAR_H / 2), width, BAR_H,
                                boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=color, linewidth=0))
    ax.add_patch(Rectangle((BAR_X, y - BAR_H / 2), width - radius, BAR_H, facecolor=color, linewidth=0))
    return BAR_X + width


def main():
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor=COLORS["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    ax.text(LEFT, HEIGHT - 0.25, TITLE, fontsize=11.5, fontweight="bold", color=COLORS["ink"], va="center")
    ax.text(LEFT, HEIGHT - 0.5, SUBTITLE, fontsize=7.8, color=COLORS["ink2"], va="center")

    bottom_y = TOP_Y - PITCH * (len(ROWS) - 1)
    # The core's limit: a dashed line down the right, labelled above the bars.
    peak_x = BAR_X + PEAK * SCALE
    ax.plot([peak_x, peak_x], [bottom_y - 0.2, TOP_Y + 0.2], color=COLORS["muted"], linewidth=1,
            linestyle=(0, (3, 3)))
    ax.text(WIDTH - LEFT, bottom_y - 0.42, "32: the core's limit, two 512-bit FMAs per cycle", ha="right",
            va="center", fontsize=7.8, color=COLORS["ink2"])
    if FOOTNOTE:
        ax.text(LEFT, bottom_y - 0.84, FOOTNOTE, ha="left", va="center", fontsize=7.4, color=COLORS["ink2"],
                linespacing=1.45)
    # The baseline the bars grow from.
    ax.plot([BAR_X, BAR_X], [bottom_y - 0.2, TOP_Y + 0.2], color=COLORS["edge"], linewidth=1)

    for row, (name, what, value, step, kind) in enumerate(ROWS):
        y = TOP_Y - PITCH * row
        ax.text(NAME_X, y, name, ha="left", va="center", fontsize=9.5, fontweight="bold", color=COLORS["ink"])
        ax.text(WHAT_X, y, what, ha="left", va="center", fontsize=8.4, color=COLORS["ink2"])
        tip = bar(ax, y, value, COLORS[kind])
        label = ax.text(tip + 0.09, y, f"{value:.2f}" if value < 10 else f"{value:.1f}", ha="left",
                        va="center", fontsize=9.5, fontweight="bold", color=COLORS["ink"])
        width = label.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
        # A bar that ends near the limit line has no room after it: its labels go inside its end.
        inside = value < PEAK and tip + 0.09 + width + 0.06 > peak_x
        if inside:
            label.set_position((tip - 0.1, y))
            label.set_ha("right")
        if step is not None:
            if inside:
                ax.text(tip - 0.1 - width - 0.12, y, f"{step:.2f}x", ha="right", va="center", fontsize=8.4,
                        color=COLORS["ink"])
            else:
                ax.text(tip + 0.09 + width + 0.1, y, f"{step:.2f}x", ha="left", va="center", fontsize=8.4,
                        color=COLORS["muted"])

    fig.savefig(OUT / "progress.png", facecolor=COLORS["surface"])
    plt.close(fig)
    print(f"wrote {OUT / 'progress.png'}")


if __name__ == "__main__":
    main()
