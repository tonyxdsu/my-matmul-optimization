"""Run: python3 docs/guide/g3_l1_sets.py (writes the GIF next to this file).

Fix 4, evidence: why n = 2048 piles a B strip into one L1 set while n = 2000 spreads it.

Zen 5 L1D: 48 KB, 12 ways, 64-byte lines -> 64 sets; set = address bits 6..11 = (address / 64) mod 64.
Simplified: the strip is one 64-byte line per row of k, 24 rows, LRU replacement.
"""

from gifkit import COLORS, Canvas

SETS, WAYS, ROWS, BASE = 64, 12, 24, 12
W, H = 8.0, 5.0

cv = Canvas(W, H, "Rows a power of two apart pile into one L1 set",
            "Zen 5's L1 has 64 sets of 12 lines; address bits 6 to 11 pick the set. The kernel walks down a B strip, a line per row.\n"
            "n = 2048: rows are 16,384 bytes apart, a multiple of 4,096, so every row lands in the same set and only 12 rows fit.\n"
            "n = 2000: rows are 16,000 bytes apart, so each row lands 58 sets further on (mod 64): all 24 rows fit.")

cw, ch, x0 = 0.105, 0.085, 0.6


class Panel:
    def __init__(self, label, top, stride_sets):
        self.stride = stride_sets
        self.cells = {}
        for s in range(SETS):
            for w in range(WAYS):
                self.cells[s, w] = cv.rect(x0 + s * cw, top - (WAYS - w) * ch, cw, ch, lw=0.5)
        cv.text(0.25, top + 0.17, label, size=10, color="ink", ha="left", fontweight="bold")
        self.status = cv.text(W - 0.25, top + 0.17, "", size=8.5, color="ink2", ha="right")
        for s in range(0, SETS, 8):
            cv.text(x0 + (s + 0.5) * cw, top - WAYS * ch - 0.1, f"set {s}" if s == 0 else str(s), size=7,
                    color="muted")
        cv.text(x0 - 0.1, top - WAYS * ch / 2, "12 ways", size=7, color="muted", rotation=90)
        self.ways = {s: [] for s in range(SETS)}   # per set: rows held, oldest first
        self.slot = {}                              # row -> (set, way)
        self.evicted = 0

    def set_of(self, k):
        return (BASE + self.stride * k) % SETS

    def touch(self, k):
        """Bring row k's line in; return the row it evicted, or None."""
        s, out = self.set_of(k), None
        if len(self.ways[s]) == WAYS:
            out = self.ways[s].pop(0)
            way = self.slot.pop(out)[1]
            self.evicted += 1
        else:
            used = {self.slot[r][1] for r in self.ways[s]}
            way = min(w for w in range(WAYS) if w not in used)
        self.ways[s].append(k)
        self.slot[k] = (s, way)
        return out

    def paint(self, current=None):
        for cell, rect in self.cells.items():
            rect.set_facecolor(COLORS["cell"])
            rect.set_edgecolor("#2f2f2c")
            rect.set_linewidth(0.5)
            rect.set_zorder(2)
        for row, cell in self.slot.items():
            rect = self.cells[cell]
            if row == current:
                rect.set_facecolor(COLORS["read"])
                rect.set_edgecolor(COLORS["ink"])
                rect.set_linewidth(1.5)
                rect.set_zorder(3)
            else:
                rect.set_facecolor("#2f6db5")

    def held(self):
        return len(self.slot)


p2048 = Panel("n = 2048", 3.62, 0)       # 16,384 B per row = 256 lines = 0 mod 64
p2000 = Panel("n = 2000", 1.92, 58)      # 16,000 B per row = 250 lines = 58 mod 64

for k in range(ROWS):
    out = p2048.touch(k)
    p2000.touch(k)
    p2048.paint(k)
    p2000.paint(k)
    p2048.status.set_text(f"rows in L1: {p2048.held()} of {k + 1}   evicted: {p2048.evicted}")
    p2000.status.set_text(f"rows in L1: {p2000.held()} of {k + 1}   evicted: {p2000.evicted}")
    note = f", evicts row {out}" if out is not None else ""
    cv.snap(f"row {k:>2}  n=2048: set {p2048.set_of(k)}{note}   n=2000: set {p2000.set_of(k)}",
            f"row {k + 1} of {ROWS}", 1100 if k in (11, 12) else 380)

p2048.paint()
p2000.paint()
p2048.status.set_text("next pass over the strip: 24 misses")
p2000.status.set_text("next pass over the strip: 0 misses")
cv.snap("next pass, for the next MR rows of A", "n = 2048: each reload evicts a row still needed", 4500)

cv.save("l1_sets")

