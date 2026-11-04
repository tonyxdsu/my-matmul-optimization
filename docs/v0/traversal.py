#!/usr/bin/env python3
"""Draws the two GIFs in the v0 section of README.md: the order in which multiplyNaive and
multiplyBlocked work through the same 12 x 12 multiply, the second with tile edge 4.

    python3 docs/v0/traversal.py      # writes naive.gif and blocking.gif next to this file

naive_steps() and blocked_steps() follow the loops in src/matmul.cpp. Each frame highlights what is
being read in A and B and what is being written in C. C's shading shows how much of each element or
tile is done. The order inside one tile is drawn by docs/v1/traversal.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

N = 12                               # matrix edge, in elements
EDGE = 4                             # tile edge, in elements
TILES = N // EDGE                    # tiles per side of each matrix
OUT = Path(__file__).resolve().parent
# The palette of bench/plot.py's dark charts.
COLORS = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "cell": "#252523", "edge": "#3a3a37", "read": "#3987e5", "trail": "#283c52", "write": "#d95926",
}
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
WIDTH, HEIGHT, DPI = 8.0, 3.7, 100   # inches; each GIF is 800 x 370 pixels, like the v1 GIFs
CELL = 0.16                          # element edge, inches
GAP = 0.8                            # gap between grids, inches
# Both GIFs do the same N^3 multiply-adds, so they run at the same pace and loop in the same time.
BLOCKED_STEP_MS, BLOCKED_HOLD_MS = 600, 2500
LOOP_MS = (TILES ** 3 - 1) * BLOCKED_STEP_MS + BLOCKED_HOLD_MS
NAIVE_STEP_MS = 110
NAIVE_HOLD_MS = LOOP_MS - (N ** 2 - 1) * NAIVE_STEP_MS


def naive_steps():
    """Yields (i, j) in the order multiplyNaive runs them: element i, j of C."""
    for i in range(N):
        for j in range(N):
            yield i, j


def blocked_steps():
    """Yields (i, j, b) in the order src/matmul.cpp runs them: tile i, j of C and k block b."""
    for i in range(TILES):           # multiplyBlocked: rows of tiles of C
        for j in range(TILES):       # multiplyBlocked: columns of tiles of C
            for b in range(TILES):   # multiplyBlock: k blocks, the tiles of A's row i and B's column j
                yield i, j, b


def blend(color, toward, t):
    a, b = np.array(to_rgb(color)), np.array(to_rgb(toward))
    return tuple(a + (b - a) * t)


def sub(*indices):
    return "".join(str(i) for i in indices).translate(SUBSCRIPT)


class Grid:
    """One N x N matrix, painted in squares of unit x unit elements. Squares bigger than one element
    are tiles: they get a label and heavier lines between them."""

    def __init__(self, ax, name, x0, top, unit):
        size, side = unit * CELL, N * CELL
        self.x0, self.top, self.size = x0, top, size
        self.squares = {}
        for r in range(N // unit):
            for c in range(N // unit):
                rect = Rectangle((x0 + c * size, top - (r + 1) * size), size, size, linewidth=0, zorder=2)
                ax.add_patch(rect)
                text = None
                if unit > 1:
                    text = ax.text(x0 + (c + 0.5) * size, top - (r + 0.5) * size, name + sub(r, c), ha="center",
                                   va="center", fontsize=10, zorder=5)
                self.squares[r, c] = (rect, text)
        for n in range(N + 1):
            heavy = n % unit == 0 if unit > 1 else n in (0, N)
            color = COLORS["muted"] if heavy else COLORS["edge"]
            width, zorder = (1.2, 4) if heavy else (0.6, 3)
            at = n * CELL
            ax.plot([x0 + at] * 2, [top, top - side], color=color, linewidth=width, zorder=zorder)
            ax.plot([x0, x0 + side], [top - at] * 2, color=color, linewidth=width, zorder=zorder)
        self.outline = Rectangle((x0, top), size, size, facecolor="none", edgecolor=COLORS["ink"],
                                 linewidth=2.2 if unit > 1 else 1.6, zorder=6, visible=False)
        ax.add_patch(self.outline)
        ax.text(x0 + side / 2, top + 0.13, name, ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=COLORS["ink"])

    def paint(self, read=(), trail=(), filled=None, target=None):
        """read and trail are squares of A or B. filled maps squares of C to the fraction done; the
        target square gets an outline."""
        for square, (rect, text) in self.squares.items():
            if square in read:
                face, ink = COLORS["read"], COLORS["ink"]
            elif square in trail:
                face, ink = COLORS["trail"], COLORS["ink2"]
            elif filled and filled.get(square, 0) > 0:
                face, ink = blend(COLORS["cell"], COLORS["write"], 0.15 + 0.7 * filled[square]), COLORS["ink"]
            else:
                face, ink = COLORS["cell"], COLORS["muted"]
            rect.set_facecolor(face)
            if text is not None:
                text.set_color(ink)
                # The label sits where two element lines cross, so it needs a background of its own.
                text.set_bbox(dict(facecolor=face, edgecolor="none", pad=1.5))
        self.outline.set_visible(target is not None)
        if target is not None:
            r, c = target
            self.outline.set_xy((self.x0 + c * self.size, self.top - (r + 1) * self.size))


class Figure:
    def __init__(self, title, subtitle, unit):
        self.fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
        self.fig.patch.set_facecolor(COLORS["surface"])
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_axis_off()
        ax.text(0.25, HEIGHT - 0.2, title, ha="left", va="top", fontsize=12, fontweight="bold", color=COLORS["ink"])
        ax.text(0.25, HEIGHT - 0.48, subtitle, ha="left", va="top", fontsize=9, color=COLORS["ink2"], linespacing=1.4)
        side = N * CELL
        left, top = (WIDTH - 3 * side - 2 * GAP) / 2, HEIGHT - 1.25
        xs = [left + n * (side + GAP) for n in range(3)]
        self.A, self.B, self.C = (Grid(ax, name, x, top, unit) for name, x in zip("ABC", xs))
        for x, symbol in ((xs[0] + side + GAP / 2, "×"), (xs[1] + side + GAP / 2, "→")):
            ax.text(x, top - side / 2, symbol, ha="center", va="center", fontsize=16, color=COLORS["muted"])
        self.caption = ax.text(0.25, 0.25, "", ha="left", va="center", fontsize=10, family="monospace",
                               color=COLORS["ink"])
        self.counter = ax.text(WIDTH - 0.25, 0.25, "", ha="right", va="center", fontsize=8.5, color=COLORS["muted"])
        self.frames = []

    def snap(self, caption, counter):
        self.caption.set_text(caption)
        self.counter.set_text(counter)
        self.fig.canvas.draw()
        self.frames.append(Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()))

    def save(self, path, step_ms, hold_ms):
        # One palette for every frame, so colors cannot shift from frame to frame.
        picks = self.frames[:: max(1, len(self.frames) // 7)] + [self.frames[-1]]
        sheet = Image.new("RGB", (self.frames[0].width, self.frames[0].height * len(picks)))
        for i, frame in enumerate(picks):
            sheet.paste(frame, (0, i * frame.height))
        palette = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in self.frames]
        durations = [step_ms] * (len(frames) - 1) + [hold_ms]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
        plt.close(self.fig)
        print(f"wrote {path} ({len(frames)} frames, {path.stat().st_size // 1024} KiB)")


def naive():
    """i-j-k over the whole matrices: each element of C is a dot product of a row and a column."""
    fig = Figure("Naive: one element of C at a time",
                 "Each step computes one element of C from a whole row of A and a whole column of B.\n"
                 "B's column touches one cache line per element, so once it outgrows the cache, nearly every read misses.",
                 unit=1)
    done = {}
    all_steps = list(naive_steps())
    for step, (i, j) in enumerate(all_steps, start=1):
        done[i, j] = 1
        fig.A.paint(read={(i, c) for c in range(N)})
        fig.B.paint(read={(r, j) for r in range(N)})
        fig.C.paint(filled=done, target=(i, j))
        fig.snap(f"i={i} j={j}   C({i},{j}) = sum over k of A({i},k) × B(k,{j})",
                 f"step {step} of {len(all_steps)}, {N} multiply-adds each")
    fig.save(OUT / "naive.gif", NAIVE_STEP_MS, NAIVE_HOLD_MS)


def blocked():
    """One tile of C at a time: each step multiplies a tile of A by a tile of B."""
    fig = Figure("v0: cache blocking, one tile at a time",
                 "Each step adds one tile of A times one tile of B into one tile of C: three small tiles that fit in cache.\n"
                 "A tile of C is done after one step per k block, then the next tile starts, left to right, top to bottom.",
                 unit=EDGE)
    done = {}
    all_steps = list(blocked_steps())
    for step, (i, j, b) in enumerate(all_steps, start=1):
        done[i, j] = (b + 1) / TILES
        fig.A.paint(read={(i, b)}, trail={(i, c) for c in range(TILES)})
        fig.B.paint(read={(b, j)}, trail={(r, j) for r in range(TILES)})
        fig.C.paint(filled=done, target=(i, j))
        fig.snap(f"tile i={i} j={j}, k block b={b}   C{sub(i, j)} += A{sub(i, b)} × B{sub(b, j)}",
                 f"step {step} of {len(all_steps)}, {EDGE ** 3} multiply-adds each")
    fig.save(OUT / "blocking.gif", BLOCKED_STEP_MS, BLOCKED_HOLD_MS)


if __name__ == "__main__":
    naive()
    blocked()
