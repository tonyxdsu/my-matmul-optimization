#!/usr/bin/env python3
"""Draws the pictures in ASSIGNMENT.md, one per exercise, into img/ next to this file.

    python3 exercises/register_blocking/visuals.py

Uses the look of the README and guide GIFs (docs/guide/gifkit.py). Numbers quoted in the pictures
are from one run of solution.cpp; the timelines are schematic: FMA latency 4 cycles, at most 2 FMAs
and 2 vector loads started per cycle.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "docs" / "guide"))

import gifkit  # noqa: E402
from gifkit import COLORS, Canvas, blend, sub  # noqa: E402

gifkit.OUT = HERE / "img"
LIGHT_BLUE = "#8fb8ee"


def save_png(cv, name):
    """Writes the canvas as one still image, sharper than a GIF frame."""
    gifkit.OUT.mkdir(exist_ok=True)
    path = gifkit.OUT / f"{name}.png"
    cv.fig.savefig(path, dpi=200, facecolor=COLORS["surface"])
    gifkit.plt.close(cv.fig)
    print(f"wrote {path}")


# =============================================================================================
# Exercise 1: axpy, lane by lane
# =============================================================================================

def ex1_lanes():
    N, SHOWN = 19, 24  # 19 elements, plus 5 cells past the end for the masked tail
    W, H = 8.0, 4.75
    cv = Canvas(W, H, "axpy: 8 doubles per instruction",
                "A zmm register holds 8 doubles, one per lane. Lane j works on element i + j of the arrays.\n"
                "n = 19: two whole vectors of 8, then 3 left over.")
    cw, ch, x0 = 0.28, 0.27, 0.95

    def cx(col):
        return x0 + col * cw

    # The arrays, with index numbers above x.
    arr_y = {"x": 3.28, "y": 2.94}
    cells, texts = {}, {}
    for name, y in arr_y.items():
        cv.text(x0 - 0.12, y + ch / 2, name, size=9, color="ink", ha="right", family="monospace")
        for col in range(SHOWN):
            past = col >= N
            cells[name, col] = cv.rect(cx(col), y, cw, ch, face=COLORS["surface"] if past else None,
                                       edge=blend(COLORS["edge"], COLORS["surface"], 0.4) if past else None)
            if past:
                cells[name, col].set_linestyle((0, (2, 2)))
            texts[name, col] = cv.text(cx(col) + cw / 2, y + ch / 2, "" if past else name + sub(col), size=6.5,
                                       color="muted" if past else "ink2")
    for col in range(0, N, 8):
        cv.text(cx(col) + cw / 2, arr_y["x"] + ch + 0.09, str(col), size=6.5, color="muted")
    cv.text(cx(N) + 2.5 * cw, arr_y["x"] + ch + 0.09, "past the end", size=6.5, color="muted")

    # The registers: 8 lanes each, moved under the elements they work on.
    reg_y = {"av": 2.25, "xv": 1.91, "yv": 1.57, "r": 1.23}
    lanes, lane_text, frames_ = {}, {}, {}
    for name, y in reg_y.items():
        cv.text(x0 - 0.12, y + ch / 2, name, size=9, color="ink", ha="right", family="monospace")
        for j in range(8):
            lanes[name, j] = cv.rect(cx(j), y, cw, ch)
            lane_text[name, j] = cv.text(cx(j) + cw / 2, y + ch / 2, "", size=6.5, color="ink")
        frames_[name] = cv.rect(cx(0), y, 8 * cw, ch, face="none", edge=COLORS["ink2"], lw=1.4, z=4)
    lane_no = [cv.text(cx(j) + cw / 2, reg_y["r"] - 0.12, f"lane {j}" if j == 0 else str(j), size=6,
                       color="muted") for j in range(8)]
    mask_label = cv.text(x0 - 0.12, 0.79 + ch / 2, "m", size=9, color="ink", ha="right", family="monospace")
    mask_cells = [cv.rect(cx(j), 0.79, cw, ch) for j in range(8)]
    mask_text = [cv.text(cx(j) + cw / 2, 0.79 + ch / 2, "", size=7, color="ink") for j in range(8)]

    def show_mask(on, bits=None):
        mask_label.set_visible(on)
        for j in range(8):
            mask_cells[j].set_visible(on)
            mask_text[j].set_visible(on)
            if on:
                b = bits[j]
                mask_text[j].set_text(str(b))
                mask_cells[j].set_facecolor(blend(COLORS["cell"], COLORS["ink2"], 0.35) if b else COLORS["cell"])

    def place(i, visible=True):
        for name in reg_y:
            frames_[name].set_x(cx(i))
            frames_[name].set_visible(visible)
            for j in range(8):
                lanes[name, j].set_x(cx(i + j))
                lanes[name, j].set_visible(visible)
                lane_text[name, j].set_x(cx(i + j) + cw / 2)
                lane_text[name, j].set_visible(visible)
        for j in range(8):
            lane_no[j].set_x(cx(i + j) + cw / 2)
            lane_no[j].set_visible(visible)
        for j in range(8):
            mask_cells[j].set_x(cx(i + j))
            mask_text[j].set_x(cx(i + j) + cw / 2)

    def fill(name, values, color, off=()):
        for j in range(8):
            if j in off:
                lanes[name, j].set_facecolor(COLORS["surface"])
                lane_text[name, j].set_text(values[j])
                lane_text[name, j].set_color(COLORS["muted"])
            else:
                lanes[name, j].set_facecolor(color if values[j] else COLORS["cell"])
                lane_text[name, j].set_text(values[j])
                lane_text[name, j].set_color(COLORS["ink"])

    def clear_regs(keep_a=True):
        for name in reg_y:
            if keep_a and name == "av":
                continue
            fill(name, [""] * 8, COLORS["cell"])

    def mark(name, cols, color):
        for col in cols:
            cells[name, col].set_facecolor(color)

    read_tint = blend(COLORS["read"], COLORS["surface"], 0.35)
    written = blend(COLORS["write"], COLORS["surface"], 0.3)
    done_y = set()

    def reset_reads():
        for col in range(N):
            cells["x", col].set_facecolor(COLORS["cell"])
            cells["y", col].set_facecolor(written if col in done_y else COLORS["cell"])

    show_mask(False)
    place(0)
    clear_regs(keep_a=False)
    cv.snap("n = 19 elements of x and y", "", 1600)

    fill("av", ["a"] * 8, LIGHT_BLUE)
    cv.snap("av = _mm512_set1_pd(a)             a in all 8 lanes, once", "before the loop", 1600)

    for i, ms in ((0, 1250), (8, 800)):
        place(i)
        clear_regs()
        reset_reads()
        cols = range(i, i + 8)
        mark("x", cols, read_tint)
        fill("xv", ["x" + sub(c) for c in cols], COLORS["read"])
        cv.snap(f"xv = _mm512_loadu_pd(x + {i})", f"i = {i}", ms)
        mark("y", cols, read_tint)
        fill("yv", ["y" + sub(c) for c in cols], COLORS["read"])
        cv.snap(f"yv = _mm512_loadu_pd(y + {i})", f"i = {i}", ms)
        fill("r", ["r" + sub(c) for c in cols], COLORS["write"])
        cv.snap("r  = _mm512_fmadd_pd(av, xv, yv)   8 multiply-adds: r = a·x + y", f"i = {i}", ms + 300)
        done_y.update(cols)
        reset_reads()
        mark("y", cols, written)
        cv.snap(f"_mm512_storeu_pd(y + {i}, r)", f"i = {i}", ms)

    place(16, visible=False)
    for col in range(16, N):
        reset_reads()
        mark("x", [col], read_tint)
        done_y.add(col)
        mark("y", [col], written)
        cv.snap(f"y[{col}] += a * x[{col}]              the last 19 % 8 = 3, one at a time", "plain loop", 900)
    reset_reads()
    cv.snap("done: 2 vector passes and 3 single steps", "", 2200)

    # Bonus: the same tail with a mask.
    for col in range(16, N):
        done_y.discard(col)
    reset_reads()
    place(16)
    clear_regs()
    off = range(3, 8)
    show_mask(True, [1, 1, 1, 0, 0, 0, 0, 0])
    cv.snap("bonus: m = 0b00000111, bit j = lane j: lanes 0-2 on, 3-7 off", "masked, i = 16", 2200)
    mark("x", range(16, N), read_tint)
    fill("xv", ["x" + sub(c) for c in range(16, N)] + ["0"] * 5, COLORS["read"], off=off)
    cv.snap("xv = _mm512_maskz_loadu_pd(m, x + 16)   off lanes: not read, 0", "masked, i = 16", 1800)
    mark("y", range(16, N), read_tint)
    fill("yv", ["y" + sub(c) for c in range(16, N)] + ["0"] * 5, COLORS["read"], off=off)
    cv.snap("yv = _mm512_maskz_loadu_pd(m, y + 16)", "masked, i = 16", 1400)
    fill("r", ["r" + sub(c) for c in range(16, N)] + ["0"] * 5, COLORS["write"], off=off)
    cv.snap("r  = _mm512_fmadd_pd(av, xv, yv)", "masked, i = 16", 1200)
    done_y.update(range(16, N))
    reset_reads()
    mark("y", range(16, N), written)
    cv.snap("_mm512_mask_storeu_pd(y + 16, m, r)   only lanes 0-2 are written", "masked, i = 16", 4000)
    cv.save("ex1_lanes")


# =============================================================================================
# Exercise 2: chains and loads, cycle by cycle
# =============================================================================================

def ex2_pipeline():
    T, PIPES, LOAD_PORTS, LAT = 24, 2, 2, 4
    W, H = 8.0, 6.35
    panels_spec = [
        ("sumSquares<1>: 1 accumulator, 1 load per FMA", 1, 1,
         "1 FMA per 4 cycles = 4 FLOP/cycle"),
        ("sumSquares<8>: 8 accumulators, 1 load per FMA", 8, 1,
         "2 FMAs per cycle = 32 FLOP/cycle"),
        ("dot<8>: 8 accumulators, 2 loads per FMA", 8, 2,
         "loads full at 2 per cycle: 1 FMA per cycle = 16 FLOP/cycle"),
    ]
    cv = Canvas(W, H, "Two limits: chains and loads",
                "Each row is one accumulator's chain; a bar is one FMA in flight (4 cycles before its result is ready).\n"
                "Each cycle the core starts at most 2 FMAs and 2 vector loads. An FMA needs the previous result of its\n"
                "own chain, and its loads.")
    x0, xs = 1.75, 5.9 / T
    row_h, row_gap = 0.085, 0.025

    def schedule(chains, loads_per_fma):
        ready, starts = [0] * chains, []
        fmas, loads = [0] * T, [0] * T
        for t in range(T):
            for c in sorted((c for c in range(chains) if ready[c] <= t), key=lambda c: (ready[c], c)):
                if fmas[t] == PIPES or loads[t] + loads_per_fma > LOAD_PORTS:
                    break
                starts.append((c, t))
                ready[c] = t + LAT
                fmas[t] += 1
                loads[t] += loads_per_fma
        return starts, fmas, loads

    def strip(y, label):
        cv.text(x0 - 0.08, y + 0.05, label, size=7, color="muted", ha="right")
        return [cv.rect(x0 + t * xs, y, xs, 0.1, lw=0.3) for t in range(T)]

    panels, top = [], H - 1.5
    first_top = top
    for title, chains, lpf, rate in panels_spec:
        p = {"chains": chains, "rate": rate}
        p["starts"], p["fmas"], p["loads"] = schedule(chains, lpf)
        cv.text(0.25, top, title, size=9.5, color="ink", ha="left", fontweight="bold")
        p["count"] = cv.text(W - 0.25, top, "", size=8.5, color="ink2", ha="right")
        p["y"] = {}
        for c in range(chains):
            y = top - 0.22 - (c + 1) * (row_h + row_gap)
            p["y"][c] = y
            cv.rect(x0, y, T * xs, row_h, face=blend(COLORS["cell"], COLORS["surface"], 0.3), edge=COLORS["surface"],
                    lw=0.3, z=1)
            cv.text(x0 - 0.08, y + row_h / 2, f"acc {c}", size=7, color="muted", ha="right")
        fy = top - 0.22 - chains * (row_h + row_gap) - 0.17
        p["fma_strip"] = strip(fy, "FMA starts")
        p["load_strip"] = strip(fy - 0.15, "vector loads")
        p["dyn"] = []
        p["bottom"] = fy - 0.15
        panels.append(p)
        top = p["bottom"] - 0.32

    axis_y = panels[-1]["bottom"] - 0.15
    for t in range(0, T + 1, 4):
        cv.text(x0 + t * xs, axis_y, str(t), size=7, color="muted")
    cv.text(x0 + T * xs / 2, axis_y - 0.17, "cycle", size=7, color="muted")
    cursor = cv.ax.plot([x0, x0], [axis_y + 0.08, first_top - 0.12], color=COLORS["ink"], lw=1, zorder=6)[0]
    shades = [COLORS["cell"], blend(COLORS["cell"], COLORS["ink2"], 0.45), COLORS["ink2"]]

    def paint(p, now):
        for patch in p["dyn"]:
            patch.remove()
        p["dyn"] = []
        for c, t in p["starts"]:
            if t <= now:
                end = min(t + LAT, now + 1, T)
                p["dyn"].append(cv.rect(x0 + t * xs, p["y"][c], (end - t) * xs, row_h,
                                        face=blend(COLORS["read"], COLORS["surface"], 0.35), edge=COLORS["surface"],
                                        lw=0.6, z=3))
                p["dyn"].append(cv.rect(x0 + t * xs, p["y"][c], xs, row_h, face=COLORS["read"],
                                        edge=COLORS["surface"], lw=0.6, z=4))
        for t in range(T):
            p["fma_strip"][t].set_facecolor(shades[p["fmas"][t]] if t <= now else COLORS["cell"])
            p["load_strip"][t].set_facecolor(shades[p["loads"][t]] if t <= now else COLORS["cell"])
        p["count"].set_text(f"FMAs started: {sum(p['fmas'][: now + 1])}")

    for now in range(T):
        for p in panels:
            paint(p, now)
        cursor.set_xdata([x0 + (now + 1) * xs] * 2)
        cv.snap(f"cycle {now:2d}", "strips: dark = 0, grey = 1, light = 2 started", 190)
    for p in panels:
        p["count"].set_text(p["rate"])
    cursor.set_visible(False)
    cv.snap("steady state: chains limit the first, loads limit the third", "peak: 2 FMAs per cycle = 32 FLOP", 6000)
    cv.save("ex2_pipeline")


# =============================================================================================
# Exercise 3: Horner is one chain; short chains overlap
# =============================================================================================

def ex3_horner():
    W, H = 8.0, 4.9
    cv = Canvas(W, H, "Horner's rule is one chain",
                "Every step needs the result of the step before, so each one waits for the last FMA to finish (4 cycles).\n"
                "The 8 lanes are 8 points evaluated side by side inside each instruction; they do not add chains.")
    # The chain, drawn for degree 4.
    steps = ["acc = c" + sub(4), "acc·x + c" + sub(3), "acc·x + c" + sub(2), "acc·x + c" + sub(1),
             "acc·x + c" + sub(0), "store y"]
    bw, bh, gap, y = 1.0, 0.5, 0.3, 3.2
    for i, label in enumerate(steps):
        x = 0.35 + i * (bw + gap)
        face = LIGHT_BLUE if i == 0 else (COLORS["write"] if i == len(steps) - 1 else COLORS["read"])
        cv.rect(x, y, bw, bh, face=blend(face, COLORS["surface"], 0.25))
        for lane in range(8):  # 8 thin lanes inside each box: 8 points, one instruction
            cv.rect(x + 0.06 + lane * (bw - 0.12) / 8, y + 0.05, (bw - 0.12) / 8 - 0.015, 0.07,
                    face=blend(face, COLORS["ink"], 0.35), edge=COLORS["surface"], lw=0.3, z=3)
        cv.text(x + bw / 2, y + bh / 2 + 0.06, label, size=8, color="ink")
        if i < len(steps) - 1:
            cv.ax.annotate("", xy=(x + bw + gap - 0.02, y + bh / 2), xytext=(x + bw + 0.02, y + bh / 2),
                           arrowprops=dict(arrowstyle="-|>", color=COLORS["ink2"], lw=1.1))
            if 1 <= i <= len(steps) - 2:  # after each FMA: the next step waits for its result
                cv.text(x + bw + gap / 2, y - 0.13, "4 cycles", size=6.5, color="muted")
    cv.text(0.35, y - 0.42, "Degree 4 shown: 4 FMAs, each waiting for the one before. At degree 256 a chain is about "
                            "1,000 cycles long.", size=8, color="ink2", ha="left")

    # Overlap: chains in flight = measured FLOP/cycle ÷ 4 (one chain alone gives 4).
    cv.text(0.35, 2.32, "The next point vector's chain does not depend on this one, so the core starts it while this one "
                        "waits, as far as its", size=8, color="ink2", ha="left")
    cv.text(0.35, 2.12, "out-of-order window (a few hundred instructions) reaches. Short chains fit more of them in the "
                        "window.", size=8, color="ink2", ha="left")
    x0, scale, span = 1.85, 5.8 / 1100, 1100
    rows = [("degree 64", 256, 9.6, 1.62), ("degree 256", 1024, 5.4, 0.92)]
    for label, length, measured, top in rows:
        in_flight = measured / 4
        step = length / in_flight
        cv.text(x0 - 0.1, top - 0.18, label, size=8.5, color="ink", ha="right", fontweight="bold")
        cv.text(x0 - 0.1, top - 0.36, f"{measured} FLOP/cycle", size=7.5, color="muted", ha="right")
        n, start = 0, 0.0
        while start < span:
            lane = n % 3
            end = min(start + length, span)
            cv.rect(x0 + start * scale, top - 0.1 - lane * 0.13, (end - start) * scale, 0.1,
                    face=blend(COLORS["read"], COLORS["surface"], 0.3 + 0.15 * (n % 2)), edge=COLORS["surface"], lw=0.5)
            n += 1
            start += step
        cv.text(x0 - 0.1, top - 0.54, f"about {in_flight:.1f} chains in flight", size=7.5, color="ink2", ha="right")
    for t in range(0, span + 1, 250):
        cv.text(x0 + t * scale, 0.42, str(t), size=7, color="muted")
    cv.text(x0 + span * scale / 2, 0.27, "cycle (each bar is one point vector's chain; schematic: in flight = "
                                         "measured ÷ 4)", size=7, color="muted")
    cv.caption.set_text("")
    save_png(cv, "ex3_horner")


# =============================================================================================
# Exercise 4: a block of accumulators, and the register budget
# =============================================================================================

def ex4_block():
    P, V, DEG = 2, 4, 4
    W, H = 8.0, 4.9
    cv = Canvas(W, H, "A block of results in registers: P = 2 polynomials × V = 4 vectors of points",
                "The point vectors are loaded once per block. At each Horner step every polynomial's coefficient is\n"
                "broadcast once and used by V FMAs; every point vector is used by P FMAs. Illustration: degree 4.")
    bx, bw, bgap = 2.25, 0.9, 0.08  # columns of the block
    xy, xh = 3.48, 0.32
    xboxes, xtext = [], []
    for v in range(V):
        x = bx + v * (bw + bgap)
        xboxes.append(cv.rect(x, xy, bw, xh))
        xtext.append(cv.text(x + bw / 2, xy + xh / 2, f"x[{8 * v}..{8 * v + 7}]", size=7.5, color="muted"))
    cv.text(bx - 0.1, xy + xh / 2, "points", size=8, color="ink2", ha="right")

    ay, ah, agap = 2.9, 0.42, 0.1
    acc, acctext = {}, {}
    for p in range(P):
        y = ay - p * (ah + agap)
        for v in range(V):
            x = bx + v * (bw + bgap)
            acc[p, v] = cv.rect(x, y, bw, ah)
            acctext[p, v] = cv.text(x + bw / 2, y + ah / 2, "acc" + sub(p, v), size=8.5, color="muted")
    # Coefficients: one row per polynomial, highest first, lined up with its accumulators.
    cw = 0.27
    coef, coeftext = {}, {}
    for p in range(P):
        y = ay - p * (ah + agap)
        for i, k in enumerate(range(DEG, -1, -1)):
            coef[p, k] = cv.rect(0.45 + i * cw, y + 0.06, cw - 0.02, ah - 0.12)
            coeftext[p, k] = cv.text(0.45 + i * cw + (cw - 0.02) / 2, y + ah / 2, "c" + sub(k), size=7, color="ink2")
        cv.text(0.45, y + ah + 0.07, f"poly {p}", size=7, color="muted", ha="left")
    arrows = [cv.ax.annotate("", xy=(bx - 0.06, ay - p * (ah + agap) + ah / 2),
                             xytext=(0.45 + (DEG + 1) * cw + 0.05, ay - p * (ah + agap) + ah / 2),
                             arrowprops=dict(arrowstyle="-|>", color=COLORS["ink2"], lw=1.0)) for p in range(P)]
    for a in arrows:
        a.set_visible(False)
    ytop, yh = ay - P * (ah + agap) - 0.12, 0.2
    ybox, ytext = {}, {}
    for p in range(P):
        y = ytop - (p + 1) * yh - p * 0.05
        cv.text(bx - 0.1, y + yh / 2, f"y, poly {p}", size=7.5, color="ink2", ha="right")
        for v in range(V):
            ybox[p, v] = cv.rect(bx + v * (bw + bgap), y, bw, yh)
            ytext[p, v] = cv.text(bx + v * (bw + bgap) + bw / 2, y + yh / 2, "", size=7, color="ink")

    # Counters and the register file.
    tx = 6.25
    names = ["point vector loads", "broadcasts", "FMAs", "stores"]
    rows_y = [3.66, 3.46, 3.26, 3.06]
    cv.text(tx, 3.88, "so far, this block", size=8, color="muted", ha="left")
    vals = []
    for y, n in zip(rows_y, names):
        cv.text(tx, y, n, size=8, color="ink2", ha="left")
        vals.append(cv.text(W - 0.25, y, "0", size=8, color="ink", ha="right", family="monospace"))
    cv.text(tx, 2.72, "zmm registers in use", size=8, color="ink2", ha="left")
    uses = ["acc"] * (P * V) + ["x"] * V + ["b"] + ["free"] * (32 - P * V - V - 1)
    face = {"acc": blend(COLORS["cell"], COLORS["write"], 0.6), "x": COLORS["read"], "b": LIGHT_BLUE,
            "free": COLORS["cell"]}
    for i, use in enumerate(uses):
        row, col = divmod(i, 8)
        cv.rect(tx + col * 0.19, 2.42 - row * 0.19, 0.16, 0.16, face=face[use])
    for i, (use, label) in enumerate([("acc", "8 accumulators"), ("x", "4 point vectors"), ("b", "1 broadcast"),
                                      ("free", "19 free")]):
        cv.rect(tx, 1.55 - i * 0.18 - 0.05, 0.1, 0.1, face=face[use])
        cv.text(tx + 0.16, 1.55 - i * 0.18, label, size=7, color="ink2", ha="left")

    def counters(loads, bcasts, fmas, stores):
        for t, val in zip(vals, (loads, bcasts, fmas, stores)):
            t.set_text(str(val))

    def paint(k=None, steps=0, loaded=False, started=False, stored=False):
        for v in range(V):
            xboxes[v].set_facecolor(COLORS["read"] if loaded else COLORS["cell"])
            xtext[v].set_color(COLORS["ink"] if loaded else COLORS["muted"])
            xboxes[v].set_edgecolor(COLORS["ink"] if (k is not None and k < DEG) else COLORS["edge"])
        for (p, kk), rect in coef.items():
            current = k is not None and kk == k
            if current:
                rect.set_facecolor(LIGHT_BLUE)
            elif k is not None and kk > k:
                rect.set_facecolor(COLORS["trail"])
            else:
                rect.set_facecolor(COLORS["cell"])
            coeftext[p, kk].set_color(COLORS["surface"] if current else COLORS["ink2"])
        for a in arrows:
            a.set_visible(k is not None)
        for (p, v), rect in acc.items():
            if started:
                rect.set_facecolor(blend(COLORS["cell"], COLORS["write"], 0.2 + 0.6 * steps / DEG))
                acctext[p, v].set_color(COLORS["ink"])
            else:
                rect.set_facecolor(COLORS["cell"])
                acctext[p, v].set_color(COLORS["muted"])
        for key, rect in ybox.items():
            rect.set_facecolor(blend(COLORS["write"], COLORS["surface"], 0.3) if stored else COLORS["cell"])
            ytext[key].set_text("stored" if stored else "")

    paint()
    counters(0, 0, 0, 0)
    cv.snap("one block: 2 polynomials × 4 vectors of 8 points", "", 1500)
    paint(loaded=True)
    counters(V, 0, 0, 0)
    cv.snap("load the block's 4 point vectors once; they stay in registers", "before the loop", 1700)
    paint(k=DEG, loaded=True, started=True)
    counters(V, P, 0, 0)
    cv.snap(f"start: every accumulator in row p = broadcast of c[p][{DEG}]", "2 broadcasts", 1700)
    for step, k in enumerate(range(DEG - 1, -1, -1), start=1):
        paint(k=k, steps=step, loaded=True, started=True)
        counters(V, P * (step + 1), P * V * step, 0)
        cv.snap(f"k={k}: 2 broadcasts, 8 FMAs   acc[p][v] = acc[p][v] × x[v] + c[p][{k}]", f"step {step} of {DEG}", 1150)
    paint(steps=DEG, loaded=True, started=True, stored=True)
    counters(V, P * (DEG + 1), P * V * DEG, P * V)
    cv.snap("store the 8 results once, after the loop", "after the loop", 1700)
    cv.snap("per step: P broadcasts and P × V FMAs; nothing goes to memory until the block is done", "", 4500)
    cv.save("ex4_block")


def ex4_registers():
    W, H = 8.0, 4.0
    cv = Canvas(W, H, "The register budget: 32 zmm registers",
                "A P × V block needs P × V accumulators, V point vectors and 1 register for the broadcast. Past 32 the\n"
                "compiler parks some values on the stack (it picks which), and each one costs memory traffic again.")
    shapes = [((1, 8), "30.8"), ((4, 4), "31.8"), ((6, 4), "31.6"), ((8, 4), "31.0: 5 spill stores per 32 FMAs, hidden"),
              ((4, 8), "22.6: 10 stores + 6 reloads per step")]
    face = {"acc": blend(COLORS["cell"], COLORS["write"], 0.6), "x": COLORS["read"], "b": LIGHT_BLUE}
    x0, cwid, top = 1.0, 0.105, 2.72
    limit_x = x0 + 32 * cwid
    for r, ((p, v), note) in enumerate(shapes):
        y = top - r * 0.42
        cv.text(x0 - 0.12, y + 0.08, f"{p} × {v}", size=9, color="ink", ha="right", fontweight="bold")
        uses = ["acc"] * (p * v) + ["x"] * v + ["b"]
        for i, use in enumerate(uses):
            over = i >= 32  # which values the compiler spills is its choice, so these only count
            c = cv.rect(x0 + i * cwid, y, cwid - 0.015, 0.17, face=COLORS["surface"] if over else face[use],
                        edge=COLORS["alert"] if over else COLORS["surface"], lw=1.0 if over else 0.3)
            if over:
                c.set_linestyle((0, (2, 1.5)))
        for i in range(len(uses), 32):
            cv.rect(x0 + i * cwid, y, cwid - 0.015, 0.17, face=COLORS["cell"], edge=COLORS["surface"], lw=0.3)
        cv.text(x0 + max(len(uses), 32) * cwid + 0.12, y + 0.08, f"{len(uses)} needed", size=7.5, color="ink2",
                ha="left")
        cv.text(x0, y - 0.1, f"FLOP/cycle {note}", size=7, color="muted", ha="left")
    cv.ax.plot([limit_x, limit_x], [top + 0.25, top - 4 * 0.42 - 0.17], color=COLORS["alert"], lw=1.4, zorder=6)
    cv.text(limit_x, top - 4 * 0.42 - 0.29, "32: the register file", size=7.5, color="alert")
    lx = 1.0
    for use, label in [("acc", "accumulators"), ("x", "point vectors"), ("b", "broadcast")]:
        cv.rect(lx, 0.3, 0.1, 0.1, face=face[use])
        cv.text(lx + 0.15, 0.35, label, size=7.5, color="ink2", ha="left")
        lx += 1.15
    over = cv.rect(lx, 0.3, 0.1, 0.1, face=COLORS["surface"], edge=COLORS["alert"])
    over.set_linestyle((0, (2, 1.5)))
    cv.text(lx + 0.15, 0.35, "over 32: that many spill to the stack", size=7.5, color="ink2", ha="left")
    cv.caption.set_text("")
    save_png(cv, "ex4_registers")


# =============================================================================================
# Exercise 5: the edges, region by region
# =============================================================================================

def ex5_edges():
    POLYS, POINTS, P, V = 10, 45, 4, 2
    pmain, nmain, nvec = POLYS - POLYS % P, POINTS - POINTS % (8 * V), POINTS - POINTS % 8
    W, H = 8.0, 4.7
    cv = Canvas(W, H, "Edges: who computes which result",
                f"{POLYS} polynomials × {POINTS} points with a {P} × {V} block ({P} polynomials × {8 * V} points). Whole blocks end at\n"
                f"row {pmain} and column {nmain}; whole vectors of 8 at column {nvec}. Every result is computed exactly once.")
    cw, ch, x0, ytop = 0.14, 0.19, 1.15, H - 1.05
    cells = {}
    for p in range(POLYS):
        y = ytop - (p + 1) * ch
        cv.text(x0 - 0.08, y + ch / 2, f"p{p}", size=6.5, color="muted", ha="right")
        for n in range(POINTS):
            cells[p, n] = cv.rect(x0 + n * cw, y, cw, ch, lw=0.4)
    for n in (0, 16, nmain, nvec, POINTS):
        cv.text(x0 + n * cw, ytop + 0.1, str(n), size=6.5, color="muted")
    region = {"A": COLORS["write"], "B": COLORS["read"], "C": COLORS["alert"], "D": COLORS["ink2"]}
    lines = [cv.ax.plot([x0 + n * cw] * 2, [ytop + 0.02, ytop - POLYS * ch - 0.02], color=COLORS["ink"], lw=1.3,
                        zorder=6, linestyle=(0, (3, 2)))[0] for n in (nmain, nvec)]
    lines.append(cv.ax.plot([x0 - 0.02, x0 + POINTS * cw + 0.02], [ytop - pmain * ch] * 2, color=COLORS["ink"], lw=1.3,
                            zorder=6, linestyle=(0, (3, 2)))[0])
    for ln in lines:
        ln.set_visible(False)
    # Legend.
    legend_y = ytop - POLYS * ch - 0.35
    labels = {"A": f"A  polyBlock<{P}, {V}> on the {pmain} × {nmain} main rectangle",
              "B": f"B  polyBlock<1, 1> on {POLYS} × {nvec - nmain}, from column {nmain}",
              "C": f"C  polyBlock<1, {V}> on {POLYS - pmain} × {nmain}, from row {pmain}; the corner is B's and D's",
              "D": f"D  scalar Horner on the last {POINTS % 8} points of every row"}
    legend = {}
    for i, key in enumerate("ABCD"):
        y = legend_y - i * 0.2
        sw = cv.rect(x0, y - 0.05, 0.12, 0.12, face=region[key])
        t = cv.text(x0 + 0.2, y, labels[key], size=8, color="ink2", ha="left", family="monospace")
        legend[key] = (sw, t)
        sw.set_visible(False)
        t.set_visible(False)

    def fill(rows, cols, key, edge=False):
        for p in rows:
            for n in cols:
                cells[p, n].set_facecolor(blend(region[key], COLORS["surface"], 0.25))
        if edge:
            cv.rect(x0 + cols[0] * cw, ytop - (rows[-1] + 1) * ch, len(cols) * cw, len(rows) * ch, face="none",
                    edge=COLORS["ink"], lw=1.2, z=5)

    def show(key):
        for part in legend[key]:
            part.set_visible(True)

    cv.snap(f"{POLYS} × {POINTS} results to compute", "", 1600)
    for ln in lines:
        ln.set_visible(True)
    cv.snap(f"pMain = {pmain}, nMain = {nmain}, nVec = {nvec}: work these out first", "the plan", 2200)
    show("A")
    for i, (r0, c0) in enumerate([(0, 0), (0, 16), (4, 0), (4, 16)]):
        fill(range(r0, r0 + P), range(c0, c0 + 8 * V), "A", edge=True)
        cv.snap(f"A: one {P} × {V} block: {P * V} accumulators, {P} × {8 * V} results", f"block {i + 1} of 4", 900)
    show("B")
    for half in (range(0, 5), range(5, POLYS)):
        fill(half, range(nmain, nvec), "B")
        cv.snap(f"B: polyBlock<1, 1>(coef, degree, {POLYS}, x + {nmain}, {nvec - nmain}, y + {nmain}, ldy)",
                "the right strip, whole vectors", 1300)
    show("C")
    fill(range(pmain, POLYS), range(0, nmain), "C")
    cv.snap(f"C: polyBlock<1, {V}>(coef + {pmain}*(degree+1), degree, {POLYS - pmain}, x, {nmain}, y + {pmain}*ldy, ldy)",
            "", 1900)
    show("D")
    for half in (range(0, 5), range(5, POLYS)):
        fill(half, range(nvec, POINTS), "D")
        cv.snap(f"D: y[p][n] for n = {nvec} to {POINTS - 1}, one point at a time", "the last 45 % 8 = 5 points", 1100)
    cv.snap(f"{pmain * nmain} + {POLYS * (nvec - nmain)} + {(POLYS - pmain) * nmain} + {POLYS * (POINTS - nvec)} = "
            f"{POLYS * POINTS} = {POLYS} × {POINTS}: every result exactly once", "", 5000)
    cv.save("ex5_edges")


if __name__ == "__main__":
    ex1_lanes()
    ex2_pipeline()
    ex3_horner()
    ex4_block()
    ex4_registers()
    ex5_edges()
