#!/usr/bin/env python3
"""Draws alternative GIFs for the v1 section of README.md. Each shows v0 and v1 in one picture,
on one clock, with a row of 8 doubles: what one zmm register holds on this CPU.

    python3 docs/v1/alternatives.py                   # writes every GIF next to this file
    python3 docs/v1/alternatives.py same_work_wide    # writes only the ones named

same_work.gif       The loop nests of v0 and v1 stepping through the same 4 x 8 tile. A frame is
                    one pass of v0's inner loop; v1 makes one pass for every 8 of v0's.
same_work_wide.gif  The same on a 4 x 32 tile, so v1's j loop has to loop too: 4 passes of 8
                    doubles to cross one row of C.
lanes.gif           Inside the inner loop, one instruction per frame: v0 moves one double through
                    registers that could hold 8, v1 fills all 8 lanes.

The look (palette, frame capture, GIF writer) is docs/guide/gifkit.py, the same as traversal.py.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "guide"))

import gifkit  # noqa: E402
from gifkit import COLORS, Canvas, blend, sub  # noqa: E402

gifkit.OUT = HERE

ROWS, DEPTH, LANES = 4, 4, 8          # extents of i, k and j: C's tile is 4 x 8, the k block is 4
MONO = 0.602 / 72                     # width of one monospace character, inches per point of size


def shade(done):
    """Orange for something that has received `done` (0 to 1) of its multiply-adds."""
    return blend(COLORS["cell"], COLORS["write"], 0.15 + 0.7 * done)


class Cells:
    """A grid of cells that is repainted every frame. label_size adds a text to each cell. group
    and gap leave a space of `gap` after every `group` columns."""

    def __init__(self, cv, x0, top, rows, cols, cell, label_size=None, group=None, gap=0.0):
        self.rects, self.texts = {}, {}
        for r in range(rows):
            for c in range(cols):
                x = x0 + c * cell + (c // group * gap if group else 0)
                self.rects[r, c] = cv.rect(x, top - (r + 1) * cell, cell, cell)
                if label_size:
                    self.texts[r, c] = cv.text(x + 0.5 * cell, top - (r + 0.5) * cell, "",
                                               size=label_size, color="muted")
        self.x0, self.top, self.height = x0, top, rows * cell
        self.width = cols * cell + ((cols - 1) // group * gap if group else 0)

    def paint(self, faces=None, outline=(), labels=None, dim=()):
        """faces maps a cell to its fill; outlined cells get a white edge; labels maps a cell to
        its text; dim cells are drawn as unused."""
        faces, labels = faces or {}, labels or {}
        for cell, rect in self.rects.items():
            lit = cell in faces
            rect.set_facecolor(faces.get(cell, COLORS["surface"] if cell in dim else COLORS["cell"]))
            rect.set_edgecolor(COLORS["ink"] if cell in outline else COLORS["edge"])
            rect.set_linewidth(2 if cell in outline else 1)
            rect.set_zorder(3 if cell in outline else 2)
            if cell in self.texts:
                self.texts[cell].set_text(labels.get(cell, ""))
                self.texts[cell].set_color(COLORS["ink"] if lit else COLORS["muted"])


class Code:
    """A loop nest in monospace. Each frame highlights lines and shows loop variables beside them."""

    def __init__(self, cv, x, top, lines, value_x, size=7.5, pitch=0.19):
        width = max(len(line) for line in lines) * MONO * size + 0.1
        self.bars, self.values = [], []
        for n, line in enumerate(lines):
            y = top - n * pitch
            self.bars.append(cv.rect(x - 0.05, y - pitch / 2 + 0.01, width, pitch - 0.02, face=COLORS["surface"],
                                     edge=COLORS["surface"], lw=0, z=1))
            cv.text(x, y, line, size=size, color="ink2", ha="left", family="monospace")
            self.values.append(cv.text(x + value_x, y, "", size=size, color="ink", ha="left", family="monospace"))

    def show(self, active, values):
        for n, bar in enumerate(self.bars):
            bar.set_facecolor(active.get(n, COLORS["surface"]))
            self.values[n].set_text(values.get(n, ""))


def same_work():
    """v0 and v1 on the same 4 x 8 tile. A frame is one pass of v0's inner loop."""
    total = ROWS * LANES * DEPTH
    cv = Canvas(8.0, 5.7, "The same tile, two loop orders",
                f"Both do the same {total} multiply-adds on a {ROWS} x {LANES} tile of C. v0's inner loop does one per "
                f"pass. v1's does a\nwhole row of {LANES}, which is one vector instruction, so it needs "
                f"{ROWS * DEPTH} passes where v0 needs {total}.")
    cell = 0.21
    xs = {"code": 0.25, "A": 2.6, "B": 3.8, "C": 6.03}
    gap_mid = (xs["B"] + LANES * cell + xs["C"]) / 2
    read_bar, write_bar = COLORS["trail"], blend(COLORS["surface"], COLORS["write"], 0.5)

    def band(top, name, lines, value_x, notes, with_sum):
        cv.text(0.25, top, name, size=10.5, color="ink", ha="left", fontweight="bold")
        stat = cv.text(7.75, top, "", size=8.5, color="ink2", ha="right")
        grid_top = top - 0.52
        mid = grid_top - ROWS * cell / 2
        code = Code(cv, xs["code"], grid_top - 0.05, lines, value_x)
        grids = {}
        for key, cols in (("A", DEPTH), ("B", LANES), ("C", LANES)):
            grids[key] = Cells(cv, xs[key], grid_top, ROWS, cols, cell)
            cv.text(xs[key] + cols * cell / 2, grid_top + 0.14, key, size=10, color="ink", fontweight="bold")
            if key in notes:
                cv.text(xs[key] + cols * cell / 2, grid_top - ROWS * cell - 0.15, notes[key], size=7.3,
                        color="muted")
        cv.text((xs["A"] + DEPTH * cell + xs["B"]) / 2, mid, "×", size=13, color="muted")
        total_box = None
        if with_sum:
            total_box = Cells(cv, gap_mid - cell / 2, mid + cell / 2, 1, 1, cell)
            cv.text(gap_mid, mid + cell / 2 + 0.11, "sum", size=7.3, color="muted")
        else:
            cv.text(gap_mid, mid, "→", size=13, color="muted")
        return stat, code, grids, total_box

    v0_stat, v0_code, v0_grids, v0_sum = band(
        4.52, "v0: i-j-k",
        ["for i", "  for j", "    for k", "      sum += A[i][k] * B[k][j]", "    C[i][j] += sum"], 0.95,
        {"B": "down a column: each read is a row away", "C": "one element, after 4 passes"}, True)
    cv.ax.plot([0.25, 7.75], [2.66, 2.66], color=COLORS["edge"], linewidth=1)
    v1_stat, v1_code, v1_grids, _ = band(
        2.38, "v1: i-k-j",
        ["for i", "  for k", "    for j", "      C[i][j] += A[i][k] * B[k][j]"], 0.95,
        {"B": "along a row: 8 adjacent doubles", "C": "a whole row, on every pass"}, False)

    for frame in range(total):
        # v0: pass number `frame` of the i-j-k nest.
        i, j, k = frame // (LANES * DEPTH), frame // DEPTH % LANES, frame % DEPTH
        last = k == DEPTH - 1
        v0_code.show({3: read_bar, **({4: write_bar} if last else {})}, {0: f"i = {i}", 1: f"j = {j}", 2: f"k = {k}"})
        v0_grids["A"].paint({**{(i, c): COLORS["trail"] for c in range(DEPTH)}, (i, k): COLORS["read"]})
        v0_grids["B"].paint({**{(r, j): COLORS["trail"] for r in range(DEPTH)}, (k, j): COLORS["read"]})
        finished = i * LANES + j + (1 if last else 0)
        v0_grids["C"].paint({(n // LANES, n % LANES): shade(1) for n in range(finished)}, outline={(i, j)})
        v0_sum.paint({(0, 0): shade((k + 1) / DEPTH)})
        v0_stat.set_text(f"{frame + 1} passes, {frame + 1} multiply-adds")

        # v1: one pass of the i-k-j nest for every LANES frames.
        step = frame // LANES
        i, k = step // DEPTH, step % DEPTH
        v1_code.show({3: read_bar}, {0: f"i = {i}", 1: f"k = {k}", 2: "j = 0..7 at once"})
        v1_grids["A"].paint({**{(i, c): COLORS["trail"] for c in range(DEPTH)}, (i, k): COLORS["read"]})
        v1_grids["B"].paint({(k, c): COLORS["read"] for c in range(LANES)})
        faces = {(r, c): shade(1) for r in range(i) for c in range(LANES)}
        faces.update({(i, c): shade((k + 1) / DEPTH) for c in range(LANES)})
        v1_grids["C"].paint(faces, outline={(i, c) for c in range(LANES)})
        v1_stat.set_text(f"{step + 1} passes, {(step + 1) * LANES} multiply-adds")

        cv.snap("one frame = one pass of v0's inner loop", f"{frame + 1} of {total}",
                2500 if frame == total - 1 else 110)
    cv.save("same_work")


def same_work_wide():
    """v0 and v1 on the same 4 x 32 tile: wide enough that v1's j loop makes 4 passes per row."""
    width = 32                                      # extent of j
    groups = width // LANES                         # passes of v1's j loop to cross one row
    total = ROWS * width * DEPTH
    cv = Canvas(8.0, 6.75, "The same tile, two loop orders",
                f"Both do the same {total} multiply-adds on a {ROWS} x {width} tile of C. v0's inner loop does one "
                f"per pass. v1's does {LANES},\none 512-bit instruction, so its j loop crosses a row in {groups} "
                f"passes: {ROWS * DEPTH * groups} passes where v0 needs {total}.")
    cell, gap = 0.13, 0.05                          # the gap sets off each run of 8 columns
    xs = {"code": 0.25, "A": 2.62, "B": 3.42}
    a_mid = xs["A"] + DEPTH * cell / 2
    op_x = (xs["A"] + DEPTH * cell + xs["B"]) / 2   # where the × and → sit, between A and B
    read_bar, write_bar = COLORS["trail"], blend(COLORS["surface"], COLORS["write"], 0.5)

    def band(top, name, lines, notes, with_sum):
        cv.text(0.25, top, name, size=10.5, color="ink", ha="left", fontweight="bold")
        stat = cv.text(7.75, top, "", size=8.5, color="ink2", ha="right")
        b_top = top - 0.62
        c_top = b_top - ROWS * cell - 0.42
        c_mid = c_top - ROWS * cell / 2
        code = Code(cv, xs["code"], b_top - 0.05, lines, 0.95)
        grids = {"A": Cells(cv, xs["A"], b_top, ROWS, DEPTH, cell)}
        cv.text(a_mid, b_top + 0.14, "A", size=10, color="ink", fontweight="bold")
        for key, grid_top in (("B", b_top), ("C", c_top)):
            grids[key] = Cells(cv, xs["B"], grid_top, ROWS, width, cell, group=LANES, gap=gap)
            cv.text(xs["B"], grid_top + 0.14, key, size=10, color="ink", ha="left", fontweight="bold")
            cv.text(xs["B"] + 0.22, grid_top + 0.135, notes[key], size=7.3, color="muted", ha="left")
        cv.text(op_x, b_top - ROWS * cell / 2, "×", size=13, color="muted")
        cv.text(op_x, c_mid, "→", size=13, color="muted")
        total_box = None
        if with_sum:
            box = 0.21
            total_box = Cells(cv, a_mid - box / 2, c_mid + box / 2, 1, 1, box)
            cv.text(a_mid, c_mid + box / 2 + 0.11, "sum", size=7.3, color="muted")
        ticks = [cv.text(xs["B"] + g * (LANES * cell + gap) + LANES * cell / 2, c_top - ROWS * cell - 0.13,
                         f"j = {g * LANES}..{(g + 1) * LANES - 1}", size=7.3, color="muted") for g in range(groups)]
        return stat, code, grids, total_box, ticks

    v0_stat, v0_code, v0_grids, v0_sum, v0_ticks = band(
        5.55, "v0: i-j-k",
        ["for i", "  for j", "    for k", "      sum += A[i][k] * B[k][j]", "    C[i][j] += sum"],
        {"B": "down a column: each read is a row away", "C": "one element, after 4 passes"}, True)
    cv.ax.plot([0.25, 7.75], [3.12, 3.12], color=COLORS["edge"], linewidth=1)
    v1_stat, v1_code, v1_grids, _, v1_ticks = band(
        2.85, "v1: i-k-j",
        ["for i", "  for k", "    for j", "      C[i][j] += A[i][k] * B[k][j]"],
        {"B": "along a row: 8 adjacent doubles per pass", "C": "8 elements per pass, a whole row in 4"}, False)

    def mark(ticks, active):
        for g, tick in enumerate(ticks):
            tick.set_color(COLORS["ink"] if g == active else COLORS["muted"])

    for frame in range(total):
        # v0: pass number `frame` of the i-j-k nest.
        i, j, k = frame // (width * DEPTH), frame // DEPTH % width, frame % DEPTH
        last = k == DEPTH - 1
        v0_code.show({3: read_bar, **({4: write_bar} if last else {})}, {0: f"i = {i}", 1: f"j = {j}", 2: f"k = {k}"})
        v0_grids["A"].paint({**{(i, c): COLORS["trail"] for c in range(DEPTH)}, (i, k): COLORS["read"]})
        v0_grids["B"].paint({**{(r, j): COLORS["trail"] for r in range(DEPTH)}, (k, j): COLORS["read"]})
        finished = i * width + j + (1 if last else 0)
        v0_grids["C"].paint({(n // width, n % width): shade(1) for n in range(finished)}, outline={(i, j)})
        v0_sum.paint({(0, 0): shade((k + 1) / DEPTH)})
        v0_stat.set_text(f"{frame + 1} passes, {frame + 1} multiply-adds")
        mark(v0_ticks, j // LANES)

        # v1: one pass of the i-k-j nest for every LANES frames; g counts the passes of its j loop.
        step = frame // LANES
        i, k, g = step // (DEPTH * groups), step // groups % DEPTH, step % groups
        lo = g * LANES
        v1_code.show({3: read_bar}, {0: f"i = {i}", 1: f"k = {k}", 2: f"j = {lo}..{lo + LANES - 1}"})
        v1_grids["A"].paint({**{(i, c): COLORS["trail"] for c in range(DEPTH)}, (i, k): COLORS["read"]})
        v1_grids["B"].paint({**{(k, c): COLORS["trail"] for c in range(width)},
                             **{(k, c): COLORS["read"] for c in range(lo, lo + LANES)}})
        faces = {(r, c): shade(1) for r in range(i) for c in range(width)}
        for c in range(width):
            updates = k + (1 if c < lo + LANES else 0)      # multiply-adds this element of row i has had
            if updates:
                faces[i, c] = shade(updates / DEPTH)
        v1_grids["C"].paint(faces, outline={(i, c) for c in range(lo, lo + LANES)})
        v1_stat.set_text(f"{step + 1} passes, {(step + 1) * LANES} multiply-adds")
        mark(v1_ticks, g)

        cv.snap("one frame = one pass of v0's inner loop", f"{frame + 1} of {total}",
                2500 if frame == total - 1 else 60)
    cv.save("same_work_wide")


def lanes():
    """Inside the inner loop, one instruction per frame, with the registers drawn as 8 lanes."""
    cv = Canvas(8.0, 5.7, "Inside the inner loop: one lane, or all eight",
                "One frame is one instruction. A register holds 8 doubles. v0 sends one double at a time through "
                "it;\nv1 fills every lane, so about the same number of instructions does 8 times the work.")
    cell = 0.26
    cols = [0.45, 2.96, 5.47]                       # left edge of the three columns
    lane_row = [(0, c) for c in range(LANES)]

    def band(top, name, memory, registers):
        """memory and registers: per column, (label, number of cells) or None."""
        cv.text(0.25, top, name, size=10.5, color="ink", ha="left", fontweight="bold")
        rows = []
        for y, spec, kind in ((top - 0.45, memory, "in memory"), (top - 1.12, registers, "register")):
            strips = []
            for x, entry in zip(cols, spec):
                if entry is None:
                    strips.append(None)
                    continue
                label, count = entry
                strips.append(Cells(cv, x, y, 1, count, cell, label_size=6.6))
                cv.text(x, y + 0.11, f"{label}  ({kind})" if kind == "in memory" else f"{kind} {label}",
                        size=7.3, color="muted", ha="left")
            rows.append(strips)
        instruction = cv.text(cols[0], top - 1.6, "", size=9, color="ink", ha="left", family="monospace")
        note = cv.text(cols[2], top - 1.25, "", size=8.5, color="ink2", ha="left")
        return rows[0], rows[1], instruction, note

    v0_mem, v0_reg, v0_ins, v0_note = band(
        4.55, "v0: scalar", [("A[i][k]", 1), ("B[k][j], a row from the last one", 1), ("C[i][j]", 1)],
        [("xmm0, the low lane of zmm0", LANES), ("xmm1 = sum", LANES), None])
    cv.ax.plot([0.25, 7.75], [2.73, 2.73], color=COLORS["edge"], linewidth=1)
    v1_mem, v1_reg, v1_ins, v1_note = band(
        2.47, "v1: vector", [("A[i][k]", 1), ("row k of B: 8 adjacent doubles", LANES), ("row i of C", LANES)],
        [("zmm3", LANES), ("zmm1", LANES), None])
    unused = set(lane_row[1:])

    frames = 4 * DEPTH
    v0_done = 0
    for frame in range(frames):
        # v0: load, multiply, add for each k; after the last k, one frame to store the sum.
        cycle = DEPTH * 3 + 1
        j, phase = frame // cycle, frame % cycle
        storing = phase == cycle - 1
        k, op = (DEPTH - 1, 3) if storing else (phase // 3, phase % 3)
        a, b, c = f"a{sub(0, k)}", f"b{sub(k, j)}", f"c{sub(0, j)}"
        summed = k + (1 if op >= 2 else 0)            # adds that have gone into sum so far
        if op == 2:
            v0_done += 1
        v0_mem[0].paint({(0, 0): COLORS["read"]} if op == 0 else {}, labels={(0, 0): a})
        v0_mem[1].paint({(0, 0): COLORS["read"]} if op == 1 else {}, labels={(0, 0): b})
        v0_mem[2].paint({(0, 0): shade(1)} if storing else {}, outline={(0, 0)} if storing else (), labels={(0, 0): c})
        v0_reg[0].paint({(0, 0): COLORS["read"] if op < 2 else COLORS["trail"]} if not storing else {},
                        labels={(0, 0): a if op == 0 else "a·b"} if not storing else {}, dim=unused)
        v0_reg[1].paint({(0, 0): shade(summed / DEPTH)} if summed else {}, outline={(0, 0)} if op == 2 else (),
                        labels={(0, 0): "sum"}, dim=unused)
        v0_ins.set_text(("vmovsd  xmm0 = A[i][k]", "vmulsd  xmm0 = xmm0 × B[k][j]",
                         "vaddsd  sum = sum + xmm0    (waits for the add before it)",
                         "then    C[i][j] += sum")[op])
        v0_note.set_text(f"{v0_done} multiply-add{'' if v0_done == 1 else 's'} done\n7 of 8 lanes unused")

        # v1: broadcast, load, multiply-add, store for each k.
        k, op = frame // 4, frame % 4
        a = f"a{sub(0, k)}"
        stored = k + (1 if op == 3 else 0)            # row updates that have reached C in memory
        v1_mem[0].paint({(0, 0): COLORS["read"]} if op == 0 else {}, labels={(0, 0): a})
        v1_mem[1].paint({cell_: COLORS["read"] for cell_ in lane_row} if op == 1 else {},
                        labels={(0, c): f"b{sub(k, c)}" for c in range(LANES)})
        v1_mem[2].paint({cell_: shade(stored / DEPTH) for cell_ in lane_row} if stored else {},
                        outline=lane_row if op >= 2 else (), labels={(0, c): f"c{sub(0, c)}" for c in range(LANES)})
        v1_reg[0].paint({cell_: COLORS["read"] if op in (0, 2) else COLORS["trail"] for cell_ in lane_row},
                        labels={cell_: a for cell_ in lane_row})
        if op == 0:
            v1_reg[1].paint()
        elif op == 1:
            v1_reg[1].paint({cell_: COLORS["read"] for cell_ in lane_row},
                            labels={(0, c): f"b{sub(k, c)}" for c in range(LANES)})
        else:
            v1_reg[1].paint({cell_: shade((k + 1) / DEPTH) for cell_ in lane_row},
                            outline=lane_row if op == 2 else (), labels={(0, c): f"c{sub(0, c)}" for c in range(LANES)})
        v1_ins.set_text(("vbroadcastsd  zmm3 = A[i][k] in all 8 lanes", "vmovupd       zmm1 = row k of B",
                         "vfmadd213pd   zmm1 = zmm3 × zmm1 + row i of C", "vmovupd       row i of C = zmm1")[op])
        v1_note.set_text(f"{stored * LANES} multiply-adds done\nevery lane in use")

        cv.snap("one frame = one instruction", f"{frame + 1} of {frames}", 3000 if frame == frames - 1 else 900)
    cv.save("lanes")


if __name__ == "__main__":
    drawings = {f.__name__: f for f in (same_work, same_work_wide, lanes)}
    for name in sys.argv[1:] or drawings:
        drawings[name]()
