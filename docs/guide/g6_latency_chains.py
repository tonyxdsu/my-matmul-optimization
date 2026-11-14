"""Run: python3 docs/guide/g6_latency_chains.py (writes the GIF next to this file).

Fix 2's floor and fix 3's accumulator count: dependency chains against Zen 5's 2 FMA pipes.

From the guide's machine limits: FMA latency 4, 2 FMAs start per cycle; a value kept in memory takes a
14-cycle round trip (load, add, store, reload). v2 at tile 48 updates 6 C vectors per row.
Greedy issue: each cycle, up to 2 chains whose previous update has finished start their next one.
"""

from gifkit import COLORS, Canvas, blend

T, PIPES = 32, 2
W, H = 8.0, 6.1
PANELS = [
    ("v2: C in memory, 6 vectors per row at tile 48", 6, 14, COLORS["write"], "6 per 14 cycles = 0.43 per cycle"),
    ("Registers, 4 independent accumulators", 4, 4, COLORS["read"], "4 per 4 cycles = 1 per cycle: half the pipes"),
    ("Registers, 8 independent accumulators", 8, 4, COLORS["read"], "2 per cycle: both pipes full"),
]

cv = Canvas(W, H, "A chain waits on itself: enough independent chains keep the FMA pipes full",
            "Each row is one chain of updates, and a bar is one update in flight. 2 FMAs start per cycle at most.\n"
            "v2 keeps C in memory: each update is a 14-cycle round trip (load, FMA, store, reload) before the next.\n"
            "In registers the wait is the FMA's 4 cycles, so 4 cycles × 2 pipes = 8 chains keep both pipes busy.")

x0, xs = 1.45, 6.0 / T
row_h, row_gap = 0.085, 0.025


def schedule(chains, lat):
    ready, starts = [0] * chains, []
    for t in range(T):
        go = sorted((c for c in range(chains) if ready[c] <= t), key=lambda c: (ready[c], c))[:PIPES]
        for c in go:
            starts.append((c, t))
            ready[c] = t + lat
    return starts


class Panel:
    def __init__(self, top, title, chains, lat, color, rate, row_name):
        self.chains, self.lat, self.color, self.rate = chains, lat, color, rate
        self.starts = schedule(chains, lat)
        cv.text(0.25, top, title, size=9.5, color="ink", ha="left", fontweight="bold")
        self.count = cv.text(W - 0.25, top, "", size=8.5, color="ink2", ha="right")
        self.y = {}
        for c in range(chains):
            y = top - 0.22 - (c + 1) * (row_h + row_gap)
            self.y[c] = y
            cv.rect(x0, y, T * xs, row_h, face=blend(COLORS["cell"], COLORS["surface"], 0.3), edge=COLORS["surface"],
                    lw=0.3, z=1)
            cv.text(x0 - 0.08, y + row_h / 2, f"{row_name} {c}", size=7, color="muted", ha="right")
        self.uy = top - 0.22 - chains * (row_h + row_gap) - 0.17
        self.util = [cv.rect(x0 + t * xs, self.uy, xs, 0.1, lw=0.3) for t in range(T)]
        cv.text(x0 - 0.08, self.uy + 0.05, "FMA starts", size=7, color="muted", ha="right")
        self.bottom = self.uy
        self.dyn = []

    def paint(self, now):
        for p in self.dyn:
            p.remove()
        self.dyn = []
        per_cycle = [0] * T
        for c, t in self.starts:
            if t <= now:
                per_cycle[t] += 1
                end = min(t + self.lat, now + 1, T)
                if end > t:
                    self.dyn.append(cv.rect(x0 + t * xs, self.y[c], (end - t) * xs, row_h,
                                            face=blend(self.color, COLORS["surface"], 0.35), edge=COLORS["surface"],
                                            lw=0.6, z=3))
                    self.dyn.append(cv.rect(x0 + t * xs, self.y[c], xs, row_h, face=self.color, edge=COLORS["surface"],
                                            lw=0.6, z=4))
        for t, rect in enumerate(self.util):
            if t > now:
                rect.set_facecolor(COLORS["cell"])
            else:
                rect.set_facecolor([COLORS["cell"], blend(COLORS["cell"], COLORS["ink2"], 0.45),
                                    COLORS["ink2"]][per_cycle[t]])
        started = sum(per_cycle[: now + 1])
        self.count.set_text(f"FMAs started: {started}")


panels, top = [], H - 1.32
first_top = top
for title, chains, lat, color, rate in PANELS:
    p = Panel(top, title, chains, lat, color, rate, "C vec" if lat > 4 else "acc")
    panels.append(p)
    top = p.bottom - 0.3

axis_y = panels[-1].bottom - 0.15
for t in range(0, T + 1, 4):
    cv.text(x0 + t * xs, axis_y, str(t), size=7, color="muted")
cv.text(x0 + T * xs / 2, axis_y - 0.17, "cycle", size=7, color="muted")
cursor = cv.ax.plot([x0, x0], [axis_y + 0.08, first_top - 0.12], color=COLORS["ink"], lw=1, zorder=6)[0]

for now in range(T):
    for p in panels:
        p.paint(now)
    cursor.set_xdata([x0 + (now + 1) * xs] * 2)
    cv.snap(f"cycle {now:2d}", "bright = the cycle an update starts", 170)

for p in panels:
    p.count.set_text(p.rate)
cursor.set_visible(False)
cv.snap("steady state, FMAs started per cycle", "peak: 2 per cycle = 32 FLOP", 5000)

cv.save("latency_chains")
