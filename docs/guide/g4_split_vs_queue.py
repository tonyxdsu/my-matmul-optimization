"""Run: python3 docs/guide/g4_split_vs_queue.py (writes the GIF next to this file).

AOCL-BLAS section: a fixed split against a shared queue when two of 8 threads run at half speed.

Model from the guide: two threads share one physical core (each at half speed) while another core idles.
1 time unit = one tile on a full-speed thread. 64 tiles, 8 threads.
"""

from gifkit import COLORS, Canvas, blend

THREADS, TILES, SLOW = 8, 64, (6, 7)
W, H = 8.0, 4.7
speed = [2.0 if w in SLOW else 1.0 for w in range(THREADS)]   # time units per tile


def fixed_split():
    plan = {}
    for w in range(THREADS):
        for i in range(TILES // THREADS):
            plan[w * 8 + i] = (w, i * speed[w], (i + 1) * speed[w])
    return plan


def shared_queue():
    plan, free = {}, [0.0] * THREADS
    for tile in range(TILES):
        w = min(range(THREADS), key=lambda x: (free[x], x))
        plan[tile] = (w, free[w], free[w] + speed[w])
        free[w] += speed[w]
    return plan


PLANS = [fixed_split(), shared_queue()]
ENDS = [max(e for _, _, e in p.values()) for p in PLANS]
TMAX = max(ENDS)

cv = Canvas(W, H, "Two half-speed threads: a fixed split waits, a shared queue adapts",
            "8 threads multiply 64 tiles of C. Threads 6 and 7 share one core, so each runs at half speed:\n"
            "the model that fits AOCL-BLAS's slow calls on 8 threads. Fixed split: each thread owns 8 tiles.\n"
            "Shared queue (your pool): a thread takes the next tile whenever it finishes one.")

fast_done, slow_done = "#2f5f99", blend(COLORS["cell"], COLORS["write"], 0.5)
idle = "#4a4a46"
cell, gtop, lane_h, lane_gap = 0.18, 3.2, 0.15, 0.03
lane_w = 1.75
scale = lane_w / TMAX


class Panel:
    def __init__(self, x0, title, plan, end):
        self.plan, self.end, self.x0 = plan, end, x0
        cv.text(x0, gtop + 0.2, title, size=10, color="ink", ha="left", fontweight="bold")
        self.tiles = {}
        for t in range(TILES):
            r, c = divmod(t, 8)
            self.tiles[t] = cv.rect(x0 + c * cell, gtop - (r + 1) * cell, cell, cell)
        self.lx = x0 + 8 * cell + 0.42
        self.bars = []
        for w in range(THREADS):
            y = gtop - (w + 1) * (lane_h + lane_gap) + lane_gap
            cv.rect(self.lx, y, lane_w, lane_h, face=COLORS["cell"], edge=COLORS["edge"], lw=0.6)
            cv.text(self.lx - 0.06, y + lane_h / 2, f"T{w}", size=7, color="alert" if w in SLOW else "muted",
                    ha="right")
        base = gtop - THREADS * (lane_h + lane_gap) + lane_gap
        for tick in (0, 4, 8, 12, 16):
            cv.text(self.lx + tick * scale, base - 0.1, str(tick), size=7, color="muted")
        cv.text(self.lx + lane_w / 2, base - 0.27, "time", size=7, color="muted")
        self.status = cv.text(x0, base - 0.42, "", size=9, color="ink2", ha="left")
        self.dyn = []

    def paint(self, now):
        for patch in self.dyn:
            patch.remove()
        self.dyn = []
        for t, (w, s, e) in self.plan.items():
            slow = w in SLOW
            if now >= e:
                face = slow_done if slow else fast_done
            elif now >= s:
                face = COLORS["write"] if slow else COLORS["read"]
            else:
                face = COLORS["cell"]
            self.tiles[t].set_facecolor(face)
            if now > s:   # busy segment on its thread's lane
                y = gtop - (w + 1) * (lane_h + lane_gap) + lane_gap
                seg = min(now, e) - s
                self.dyn.append(cv.rect(self.lx + s * scale, y, seg * scale, lane_h,
                                        face=COLORS["write"] if slow else COLORS["read"], edge=COLORS["surface"],
                                        lw=0.5, z=3))
        for w in range(THREADS):   # idle time: finished its share while the call still runs
            last = max(e for (ww, _, e) in self.plan.values() if ww == w)
            stop = min(now, self.end)
            if stop > last:
                y = gtop - (w + 1) * (lane_h + lane_gap) + lane_gap
                self.dyn.append(cv.rect(self.lx + last * scale, y, (stop - last) * scale, lane_h, face=idle,
                                        edge=COLORS["surface"], lw=0.5, z=3))
        done = sum(1 for (_, _, e) in self.plan.values() if now >= e)
        if now >= self.end:
            self.status.set_text(f"all 64 tiles done at t = {self.end:g}")
            self.status.set_color(COLORS["ink"])
        else:
            self.status.set_text(f"{done} of 64 tiles done")
        return done


left = Panel(0.35, "Fixed split (AOCL-BLAS)", PLANS[0], ENDS[0])
right = Panel(4.2, "Shared queue (your pool)", PLANS[1], ENDS[1])

legend = [(COLORS["read"], "full-speed thread working"), (COLORS["write"], "half-speed thread working"),
          (idle, "idle, waiting for the call to end")]
lx = 0.35
for face, label in legend:
    cv.rect(lx, 0.74, 0.12, 0.12, face=face)
    cv.text(lx + 0.18, 0.8, label, size=7.5, color="ink2", ha="left")
    lx += 2.45

steps = int(TMAX / 0.5) + 1
for i in range(steps):
    now = i * 0.5
    a = left.paint(now)
    b = right.paint(now)
    cv.snap(f"t = {now:4.1f}   fixed: {a:2d} of 64 tiles   queue: {b:2d} of 64",
            "1 time unit = 1 tile on a full-speed thread", 260)
cv.snap(f"fixed: t = {ENDS[0]:g}, 2× the full-speed 8   queue: t = {ENDS[1]:g}",
        "the queue loses only the slow threads' share", 4500)

cv.save("split_vs_queue")
print("ends", ENDS)
