"""Run: python3 docs/guide/g2_packing.py (writes the GIF next to this file).

Fix 4, idea: packing a B panel into NR-wide strips, each stored k by k as one contiguous stream.

Illustration: n = 16 doubles per row of B, a panel of 4 rows of k x 12 columns, NR = 4 (3 strips).
"""

from gifkit import COLORS, Canvas, blend

N, KB, NR, STRIPS = 16, 4, 4, 3
W, H = 8.0, 4.9

cv = Canvas(W, H, "Packing: copy each B strip once, in the order the kernel reads it",
            "B is row-major, so a strip NR columns wide is scattered: each row's piece starts n doubles after the last.\n"
            "Packing copies each strip k by k into one buffer, so the kernel then reads one contiguous stream per strip.\n"
            "Illustration: n = 16, a panel of 4 rows of k × 12 columns, NR = 4. A is packed the same way, in strips MR rows tall.")

other = blend(COLORS["cell"], COLORS["surface"], 0.6)
done_dst = blend(COLORS["cell"], COLORS["write"], 0.45)

# B as a grid
gx, gw, gh, gtop = 1.0, 0.3, 0.24, 3.42
grid = {}
for k in range(KB):
    for j in range(N):
        grid[k, j] = cv.rect(gx + j * gw, gtop - (k + 1) * gh, gw, gh)
    cv.text(gx - 0.08, gtop - (k + 0.5) * gh, f"k={k}", size=7.5, color="muted", ha="right")
for s in range(STRIPS):
    cv.text(gx + (s * NR + NR / 2) * gw, gtop + 0.12, f"strip {s}", size=8, color="ink2")
cv.text(gx + (STRIPS * NR + (N - STRIPS * NR) / 2) * gw, gtop + 0.12, "other panels", size=8, color="muted")
for s in range(1, STRIPS + 1):
    cv.ax.plot([gx + s * NR * gw] * 2, [gtop - KB * gh, gtop], color=COLORS["muted"], lw=1.2, zorder=4)
cv.text(gx + N * gw + 0.2, gtop - KB * gh / 2, "B, 4 rows of k\n× 16 columns", size=8, color="ink2", ha="left",
        linespacing=1.35)

# B in memory, row by row
mx, mw, mh, mgap, my = 1.0, 0.1, 0.2, 0.06, 1.88
mem = {}
for k in range(KB):
    for j in range(N):
        mem[k, j] = cv.rect(mx + (k * N + j) * mw + k * mgap, my, mw, mh, lw=0.8)
    cv.text(mx + (k * N + N / 2) * mw + k * mgap, my - 0.12, f"row k={k}", size=7, color="muted")
cv.text(mx, my + 0.32, "B in memory, row by row: each strip's 4 doubles of a row sit 16 doubles after the last row's",
        size=8, color="ink2", ha="left")

# Packed B
px, pw, ph, pgap, py = 1.0, 0.1, 0.2, 0.12, 0.92
packed = {}
for s in range(STRIPS):
    for k in range(KB):
        for c in range(NR):
            i = s * KB * NR + k * NR + c
            packed[s, k, c] = cv.rect(px + i * pw + s * pgap, py, pw, ph, lw=0.8)
    cv.text(px + (s * KB * NR + KB * NR / 2) * pw + s * pgap, py - 0.12, f"strip {s}", size=7, color="muted")
cv.text(px, py + 0.32, "packed B: strip 0, strip 1, strip 2, each stored k by k", size=8, color="ink2", ha="left")


def paint(cur=None, copied=(), reading=None):
    """cur = (s, k) being copied; copied = set of (s, k) done; reading = (s, k) the kernel reads."""
    for (k, j), rect in list(grid.items()) + list(mem.items()):
        rect.set_edgecolor(COLORS["edge"])
        rect.set_linewidth(1 if rect in grid.values() else 0.8)
        rect.set_zorder(2)
        s = j // NR
        if j >= STRIPS * NR:
            rect.set_facecolor(other)
        elif cur == (s, k) or reading == (s, k):
            rect.set_facecolor(COLORS["read"])
        elif (s, k) in copied:
            rect.set_facecolor(COLORS["trail"])
        else:
            rect.set_facecolor(COLORS["cell"])
    for (s, k, c), rect in packed.items():
        rect.set_edgecolor(COLORS["edge"])
        rect.set_linewidth(0.8)
        rect.set_zorder(2)
        if reading == (s, k):
            rect.set_facecolor(COLORS["read"])
            rect.set_edgecolor(COLORS["ink"])
            rect.set_linewidth(1.5)
            rect.set_zorder(3)
        elif cur == (s, k):
            rect.set_facecolor(COLORS["write"])
        elif (s, k) in copied:
            rect.set_facecolor(done_dst)
        else:
            rect.set_facecolor(COLORS["cell"])


paint()
cv.snap("a panel of B: 3 strips, each 4 rows of k × NR = 4", "", 1800)

copied = set()
for s in range(STRIPS):
    for k in range(KB):
        paint(cur=(s, k), copied=copied)
        src, dst = k * N + s * NR, (s * KB + k) * NR
        cv.snap(f"strip {s}, k={k}: B[{src:>2}..{src + 3:>2}] → packed[{dst:>2}..{dst + 3:>2}]",
                f"copied {(len(copied) + 1) * NR} of {STRIPS * KB * NR} doubles", 650)
        copied.add((s, k))

for k in range(KB):
    paint(copied=copied, reading=(0, k))
    cv.snap(f"kernel, strip 0, k={k}: reads packed[{k * NR}..{k * NR + 3}]",
            "the next k is the next 4 doubles", 900)

paint(copied=copied)
cv.snap("each strip is one contiguous stream", "the copy is paid once, then reused for every MR rows of A", 4000)

cv.save("packing")
