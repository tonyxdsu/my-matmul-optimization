"""Shared look for the GIFs in the "Matmul part 2" guide (a Claude Docs doc): the palette, frame capture and palette-locked GIF writer of
docs/v1/traversal.py, so the guide's animations match the README's."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parent
COLORS = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "cell": "#252523", "edge": "#3a3a37", "read": "#3987e5", "trail": "#283c52", "write": "#d95926",
    "alert": "#e8b931",
}
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
DPI = 100


def blend(color, toward, t):
    a, b = np.array(to_rgb(color)), np.array(to_rgb(toward))
    return tuple(a + (b - a) * t)


def sub(*indices):
    return "".join(str(i) for i in indices).translate(SUBSCRIPT)


class Canvas:
    """A dark figure in inches, with a title, a subtitle, and a caption and counter along the bottom."""

    def __init__(self, width, height, title, subtitle):
        self.W, self.H = width, height
        self.fig = plt.figure(figsize=(width, height), dpi=DPI)
        self.fig.patch.set_facecolor(COLORS["surface"])
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_axis_off()
        self.ax = ax
        ax.text(0.25, height - 0.2, title, ha="left", va="top", fontsize=12, fontweight="bold", color=COLORS["ink"])
        ax.text(0.25, height - 0.48, subtitle, ha="left", va="top", fontsize=9, color=COLORS["ink2"],
                linespacing=1.4)
        self.caption = ax.text(0.25, 0.25, "", ha="left", va="center", fontsize=10, family="monospace",
                               color=COLORS["ink"])
        self.counter = ax.text(width - 0.25, 0.25, "", ha="right", va="center", fontsize=8.5,
                               color=COLORS["muted"])
        self.frames, self.durations = [], []

    def rect(self, x, y, w, h, face=None, edge=None, lw=1, z=2):
        r = Rectangle((x, y), w, h, linewidth=lw, facecolor=face or COLORS["cell"],
                      edgecolor=edge or COLORS["edge"], zorder=z)
        self.ax.add_patch(r)
        return r

    def text(self, x, y, s, size=8.5, color="ink2", **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        return self.ax.text(x, y, s, fontsize=size, color=COLORS.get(color, color), zorder=5, **kw)

    def snap(self, caption, counter, ms):
        self.caption.set_text(caption)
        self.counter.set_text(counter)
        self.fig.canvas.draw()
        self.frames.append(Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()))
        self.durations.append(ms)

    def save(self, name, still=None):
        """Write out/<name>.gif with one palette for every frame; still = index of a frame to also save as PNG."""
        OUT.mkdir(exist_ok=True)
        picks = self.frames[:: max(1, len(self.frames) // 7)] + [self.frames[-1]]
        sheet = Image.new("RGB", (self.frames[0].width, self.frames[0].height * len(picks)))
        for i, frame in enumerate(picks):
            sheet.paste(frame, (0, i * frame.height))
        palette = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in self.frames]
        path = OUT / f"{name}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=self.durations, loop=0)
        if still is not None:
            self.frames[still].save(OUT / f"{name}_still.png")
            self.frames[-1].save(OUT / f"{name}_last.png")
        plt.close(self.fig)
        print(f"wrote {path} ({len(frames)} frames, {path.stat().st_size // 1024} KiB)")
