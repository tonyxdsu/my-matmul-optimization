"""Run: python3 docs/guide/g1_register_block.py (writes the GIF next to this file).

Fix 3: one MR x NR block of C kept in registers for a whole k block, counted against v2 doing the same work.

Illustration shape MR = 4, NR/8 = 3 (the same grid as the guide's static drawing), k block 8.
"""

from gifkit import COLORS, Canvas, blend, sub

MR, NV, KB = 4, 3, 8
W, H = 8.0, 4.8

cv = Canvas(W, H, "Register blocking: the C block stays in registers for a whole k block",
            "Each step of k broadcasts MR values of A and loads NR/8 vectors of B, then does MR × NR/8 vector FMAs.\n"
            "C is loaded once before the k loop and stored once after it. Illustration: MR = 4, NR/8 = 3, k block of 8.")

# B strip: KB rows of NV vectors, above the C block
bx, bw, bgap = 3.25, 0.62, 0.06
brow_h, brow_gap, btop = 0.15, 0.03, 3.75
B = {}
for k in range(KB):
    y = btop - (k + 1) * brow_h - k * brow_gap
    for v in range(NV):
        B[k, v] = cv.rect(bx + v * (bw + bgap), y, bw, brow_h)
    cv.text(bx - 0.08, y + brow_h / 2, f"k={k}", size=7.5, color="muted", ha="right")
cv.text(2.62, btop - 0.75, "B strip\nk block × NR\none row of k = NR/8 vectors", size=8, color="ink2", ha="right",
        linespacing=1.35)

# C block: MR rows of NV accumulators
crow_h, crow_gap, ctop = 0.28, 0.06, 2.08
C, Ctext = {}, {}
for r in range(MR):
    y = ctop - (r + 1) * crow_h - r * crow_gap
    for v in range(NV):
        C[r, v] = cv.rect(bx + v * (bw + bgap), y, bw, crow_h)
        Ctext[r, v] = cv.text(bx + v * (bw + bgap) + bw / 2, y + crow_h / 2, "c" + sub(r, v), size=8.5, color="muted")
cv.text(bx + (NV * bw + (NV - 1) * bgap) / 2, ctop - MR * (crow_h + crow_gap) - 0.1,
        "C block: 12 accumulators in zmm registers", size=8, color="ink2", va="top")

# A strip: MR rows of KB scalars, rows lined up with C's rows
ax0, aw, agap = 0.5, 0.28, 0.04
A = {}
for r in range(MR):
    y = ctop - (r + 1) * crow_h - r * crow_gap
    for k in range(KB):
        A[r, k] = cv.rect(ax0 + k * (aw + agap), y, aw, crow_h)
    cv.text(ax0 - 0.08, y + crow_h / 2, f"r{r}", size=7.5, color="muted", ha="right")
for k in range(KB):
    cv.text(ax0 + k * (aw + agap) + aw / 2, ctop + 0.1, f"k{k}", size=7.5, color="muted")
cv.text(ax0, ctop + 0.32, "A strip: MR rows × k block", size=8, color="ink2", ha="left")

# Counters: the micro-kernel against v2 doing the same block of work
tx, tk, tv = 5.55, 7.05, 7.72
rows_y = [3.42, 3.22, 3.02, 2.82, 2.62]
names = ["vector FMAs", "B vector loads", "A broadcasts", "C vector loads", "C vector stores"]
cv.text(tx, 3.66, "so far", size=8.5, color="muted", ha="left", family="monospace")
cv.text(tk, 3.66, "kernel", size=8.5, color="ink", ha="right", family="monospace", fontweight="bold")
cv.text(tv, 3.66, "v2", size=8.5, color="ink", ha="right", family="monospace", fontweight="bold")
for y, name in zip(rows_y, names):
    cv.text(tx, y, name, size=8.5, color="ink2", ha="left", family="monospace")
kern = [cv.text(tk, y, "", size=8.5, color="ink", ha="right", family="monospace") for y in rows_y]
v2 = [cv.text(tv, y, "", size=8.5, color="ink", ha="right", family="monospace") for y in rows_y]
cv.text(tx, 2.4, "v2 = same FMAs, row update by row update", size=7.5, color="muted", ha="left")

# The register file, for this illustration's shape
cv.text(tx, 2.12, "32 zmm registers in this illustration", size=8, color="ink2", ha="left")
uses = ["acc"] * 12 + ["b"] * 3 + ["a"] + ["free"] * 16
face = {"acc": blend(COLORS["cell"], COLORS["write"], 0.6), "b": COLORS["read"], "a": "#8fb8ee",
        "free": COLORS["cell"]}
for i, use in enumerate(uses):
    row, col = divmod(i, 16)
    cv.rect(tx + col * 0.135, 1.83 - row * 0.16, 0.12, 0.13, face=face[use])
legend = [("acc", "12 C accumulators", tx, 1.5), ("b", "3 B vectors", tx + 1.25, 1.5),
          ("a", "1 A broadcast", tx, 1.3), ("free", "16 free", tx + 1.25, 1.3)]
for use, label, x, y in legend:
    cv.rect(x, y - 0.05, 0.1, 0.1, face=face[use])
    cv.text(x + 0.15, y, label, size=7.5, color="ink2", ha="left")


def paint(k=None, loaded=False, done_steps=0, outline=False):
    for (r, kk), rect in A.items():
        if k is not None and kk == k:
            rect.set_facecolor(COLORS["read"])
        elif kk < done_steps:
            rect.set_facecolor(COLORS["trail"])
        else:
            rect.set_facecolor(COLORS["cell"])
    for (kk, v), rect in B.items():
        if k is not None and kk == k:
            rect.set_facecolor(COLORS["read"])
        elif kk < done_steps:
            rect.set_facecolor(COLORS["trail"])
        else:
            rect.set_facecolor(COLORS["cell"])
    for cell, rect in C.items():
        if loaded:
            rect.set_facecolor(blend(COLORS["cell"], COLORS["write"], 0.15 + 0.7 * done_steps / KB))
            Ctext[cell].set_color(COLORS["ink"])
        else:
            rect.set_facecolor(COLORS["cell"])
            Ctext[cell].set_color(COLORS["muted"])
        rect.set_edgecolor(COLORS["ink"] if outline else COLORS["edge"])
        rect.set_linewidth(2 if outline else 1)
        rect.set_zorder(3 if outline else 2)


def count(steps, c_loads, c_stores):
    f = MR * NV * steps
    for t, val in zip(kern, [f, NV * steps, MR * steps, c_loads, c_stores]):
        t.set_text(str(val))
    for t, val in zip(v2, [f, f, MR * steps, f, f]):
        t.set_text(str(val))


paint()
count(0, 0, 0)
cv.snap("the kernel starts on one 4 × 3 block of C", "", 1200)

paint(loaded=True, outline=True)
count(0, 12, 0)
cv.snap("load the C block: 12 vector loads, once", "before the k loop", 1300)

for k in range(KB):
    paint(k=k, loaded=True, done_steps=k + 1, outline=True)
    count(k + 1, 12, 0)
    cv.snap(f"k={k}   12 FMAs: c(r,v) += a(r,{k}) × b({k},v)", f"step {k + 1} of {KB}: 4 broadcasts + 3 loads", 800)

paint(loaded=True, done_steps=KB, outline=True)
count(KB, 12, 12)
cv.snap("store the C block: 12 vector stores, once", "after the k loop", 3500)

cv.save("register_block")
