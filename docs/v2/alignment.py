#!/usr/bin/env python3
"""Draws the diagram in the v2 section of README.md: where a large matrix's data starts in v1 and
in v2, and which 64-byte loads cross a cache line because of it.

    python3 docs/v2/alignment.py      # writes alignment.png next to this file

Addresses are drawn to scale, from the start of a block malloc got from mmap: v1's plain
std::vector data starts at page + 16 and v2's AlignedAllocator data at page + 64.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
# The palette of bench/plot.py's dark charts.
COLORS = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "cell": "#252523", "edge": "#3a3a37", "header": "#4a4a46", "fits": "#3987e5", "split": "#d95926",
}
WIDTH, HEIGHT, DPI = 8.0, 4.55, 200   # inches; the PNG is 1600 x 910 pixels
BYTES = 192                           # how much of the block to draw
LINE = 64                             # cache line, bytes
LEFT = 0.32                           # left and right margin, inches
SCALE = (WIDTH - 2 * LEFT) / BYTES    # inches per byte
STRIP_H, LOAD_H = 0.34, 0.27          # heights of the memory strip and of a load, inches


def x(byte):
    return LEFT + byte * SCALE


def memory_strip(ax, y, data_start, header_start):
    """Draws the strip with its top edge at y: unused bytes, malloc's header, then 8-byte doubles."""
    if header_start > 0:
        ax.text((x(0) + x(header_start)) / 2, y - STRIP_H / 2, f"skipped: {header_start} bytes",
                ha="center", va="center", fontsize=8, color=COLORS["muted"])
    ax.add_patch(Rectangle((x(header_start), y - STRIP_H), x(data_start) - x(header_start), STRIP_H,
                           facecolor=COLORS["header"], edgecolor=COLORS["edge"], linewidth=0.8))
    ax.text((x(header_start) + x(data_start)) / 2, y - STRIP_H / 2, "header", ha="center", va="center",
            fontsize=8, color=COLORS["ink"])
    for byte in range(data_start, BYTES, 8):
        ax.add_patch(Rectangle((x(byte), y - STRIP_H), 8 * SCALE, STRIP_H,
                               facecolor=COLORS["cell"], edgecolor=COLORS["edge"], linewidth=0.8))
    ax.add_patch(Rectangle((x(0), y - STRIP_H), BYTES * SCALE, STRIP_H, facecolor="none",
                           edgecolor=COLORS["muted"], linewidth=0.8))
    for byte in range(LINE, BYTES, LINE):
        ax.plot([x(byte)] * 2, [y + 0.05, y - STRIP_H - 0.07], color=COLORS["ink2"], linewidth=1.8,
                solid_capstyle="butt")


def address_labels(ax, y, data_start):
    """Labels the line boundaries and the data start above a strip whose top edge is at y."""
    ticks = [(0, "left"), (BYTES, "right")] + [(b, "center") for b in range(LINE, BYTES, LINE) if b != data_start]
    for byte, align in ticks:
        ax.text(x(byte), y + 0.09, str(byte), ha=align, va="bottom", fontsize=8, color=COLORS["muted"])
    ax.text(x(data_start) - 0.07, y + 0.09, str(data_start), ha="right", va="bottom", fontsize=8,
            color=COLORS["ink2"])
    ax.annotate("", xy=(x(data_start), y + 0.01), xytext=(x(data_start), y + 0.36),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["ink2"], linewidth=1, shrinkA=0, shrinkB=0))
    ax.text(x(data_start) + 0.08, y + 0.24, "data.data()", ha="left", va="center", fontsize=8.5,
            family="monospace", color=COLORS["ink"])


def load(ax, y, start, number, color):
    """Draws one 64-byte load under a strip; a line boundary inside it is marked in its color."""
    pad = 0.015
    ax.add_patch(FancyBboxPatch((x(start) + pad, y - LOAD_H), LINE * SCALE - 2 * pad, LOAD_H,
                                boxstyle="round,pad=0,rounding_size=0.04", facecolor=color + "33",
                                edgecolor=color, linewidth=1.2))
    boundary = (start // LINE + 1) * LINE
    crosses = start % LINE != 0
    if crosses:
        ax.plot([x(boundary)] * 2, [y + 0.04, y - LOAD_H - 0.04], color=color, linewidth=2.4,
                solid_capstyle="butt")
    # Center the words in the part before the boundary, so the boundary mark never crosses them.
    end = boundary if crosses else start + LINE
    ax.text((x(start) + x(end)) / 2, y - LOAD_H / 2, f"load {number}: bytes {start}-{start + LINE - 1}",
            ha="center", va="center", fontsize=8, color=COLORS["ink"])


def main():
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor=COLORS["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    ax.text(LEFT, 4.32, "A 64-byte load crosses a cache line unless the buffer starts on one", fontsize=11.5,
            fontweight="bold", color=COLORS["ink"], va="center")
    ax.text(LEFT, 4.09, "First 192 bytes of a matrix from mmap, to scale. Each cell is one 8-byte double; "
                        "thick lines are cache-line boundaries.", fontsize=7.8, color=COLORS["ink2"], va="center")

    # v1
    ax.text(LEFT, 3.76, "v1: plain std::vector. malloc returns page + 16, which is 16 bytes into a line",
            fontsize=9, fontweight="bold", color=COLORS["ink"], va="center")
    memory_strip(ax, 3.25, data_start=16, header_start=0)
    address_labels(ax, 3.25, data_start=16)
    load(ax, 2.78, 16, 1, COLORS["split"])
    load(ax, 2.78, 80, 2, COLORS["split"])
    ax.text(LEFT, 2.33, "Both loads cross a line, so each needs two cache accesses: 1 load can start per cycle.",
            fontsize=8.6, color=COLORS["ink"], va="center")

    # v2
    ax.text(LEFT, 1.92, "v2: AlignedAllocator. aligned operator new returns page + 64, which starts a line",
            fontsize=9, fontweight="bold", color=COLORS["ink"], va="center")
    memory_strip(ax, 1.41, data_start=64, header_start=48)
    address_labels(ax, 1.41, data_start=64)
    load(ax, 0.94, 64, 1, COLORS["fits"])
    load(ax, 0.94, 128, 2, COLORS["fits"])
    ax.text(LEFT, 0.49, "Each load stays in one line, so each needs one cache access: 2 loads can start per cycle.",
            fontsize=8.6, color=COLORS["ink"], va="center")
    ax.text(LEFT, 0.26, "malloc over-allocates, skips ahead to the next line and writes its header just before it.",
            fontsize=7.8, color=COLORS["ink2"], va="center")

    fig.savefig(OUT / "alignment.png", facecolor=COLORS["surface"])
    plt.close(fig)
    print(f"wrote {OUT / 'alignment.png'}")


if __name__ == "__main__":
    main()
