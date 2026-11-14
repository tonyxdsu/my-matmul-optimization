"""Run: python3 docs/guide/g5_alias_test.py (writes the GIF next to this file).

Fix 2, aliasing test 2: if C[j] were the same memory as B[j+3], source order and vector order disagree.

Row update C[j] += a * B[j], j = 0..7. Memory cells 0..10: B[j] at cell j, C[j] at cell j + 3.
"""

from gifkit import COLORS, Canvas, blend

CELLS, SHIFT, VEC = 11, 3, 8
W, H = 8.0, 4.7

cv = Canvas(W, H, "Why GCC tests whether C starts 1 to 7 doubles after B",
            "The row update is C[j] += a × B[j]. Suppose C[j] were the same memory as B[j+3], so writing C[0] changes B[3].\n"
            "One element at a time, the source order, B[3] is read after C[0] was written. Eight at a time, it is read before.\n"
            "The two orders give different results, so the vector loop is only correct when this overlap cannot happen.")

old_write = blend(COLORS["cell"], COLORS["write"], 0.45)
x0, cw, chh = 1.4, 0.5, 0.32


class Strip:
    def __init__(self, title, top):
        cv.text(0.25, top, title, size=10, color="ink", ha="left", fontweight="bold")
        self.note = cv.text(W - 0.25, top, "", size=8.5, color="alert", ha="right")
        y = top - 0.6
        self.cells = [cv.rect(x0 + i * cw, y, cw, chh) for i in range(CELLS)]
        for i in range(CELLS):
            cv.text(x0 + (i + 0.5) * cw, y + chh + 0.11, f"B[{i}]", size=7.5, color="ink2" if i < VEC else "muted")
            if i >= SHIFT:
                cv.text(x0 + (i + 0.5) * cw, y - 0.11, f"C[{i - SHIFT}]", size=7.5, color="ink2")
        cv.text(x0 - 0.1, y + chh / 2, "memory", size=7.5, color="muted", ha="right")

    def paint(self, read=(), write=(), written=(), alert=()):
        for i, rect in enumerate(self.cells):
            if i in write:
                face = COLORS["write"]
            elif i in read:
                face = COLORS["read"]
            elif i in written:
                face = old_write
            else:
                face = COLORS["cell"]
            rect.set_facecolor(face)
            hot = i in alert
            rect.set_edgecolor(COLORS["alert"] if hot else COLORS["edge"])
            rect.set_linewidth(2.5 if hot else 1)
            rect.set_zorder(3 if hot else 2)


src = Strip("Source order: one element at a time", 3.45)
vec = Strip("Vector order: 8 at a time", 2.25)
summary = cv.text(W / 2, 0.82, "", size=9, color="ink", fontweight="bold")

src.paint()
vec.paint()
cv.snap("C[j] += a × B[j], with C[j] at the same address as B[j+3]", "", 1800)

written = set()
for j in range(VEC):
    b, c = j, j + SHIFT
    stale = b in written
    src.paint(read={b}, write={c}, written=written, alert={b} if stale else ())
    if stale:
        src.note.set_text(f"B[{b}] was overwritten as C[{b - SHIFT}]: this read sees the new value")
    note = f"   B[{b}] is C[{b - SHIFT}], already updated" if stale else ""
    cv.snap(f"j={j}   C[{j}] += a × B[{j}]{note}", f"source order, element {j + 1} of {VEC}",
            1600 if j == SHIFT else 800)
    written.add(c)
src.paint(written=written, alert=set(range(SHIFT, VEC)))
src.note.set_text("B[3] to B[7] were read after being overwritten")

vec.paint(read=set(range(VEC)))
cv.snap("load B[0..7]: one vector, before anything is stored", "vector order, step 1 of 3", 1600)
vec.paint(read=set(range(VEC)), alert=set(range(SHIFT, VEC)))
vec.note.set_text("B[3] to B[7] were read before C[0] to C[4] were stored: old values")
cv.snap("load C[0..7] and do 8 FMAs", "vector order, step 2 of 3", 1600)
vec.paint(write=set(range(SHIFT, SHIFT + VEC)), alert=set(range(SHIFT, VEC)))
cv.snap("store C[0..7]", "vector order, step 3 of 3", 1600)
vec.paint(written=set(range(SHIFT, SHIFT + VEC)), alert=set(range(SHIFT, VEC)))
summary.set_text("C[3] to C[7] come out different, so GCC runs the scalar copy if C starts 1 to 7 doubles after B")
cv.snap("test 2 checks this distance at run time", "aliasing: two names, one address", 5000)

cv.save("alias_test")
