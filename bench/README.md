# Benchmarks

How the numbers in the [top-level README](../README.md) are produced: setup, commands, methodology,
hardware counters and charts. The suite is built on
[Google Benchmark](https://github.com/google/benchmark), with OpenBLAS as a reference.

| Question | Benchmarks | Chart |
|---|---|---|
| How much faster is cache blocking than the naive loop, and why? | `BM_Naive`, `BM_Blocked` over n = 100 to 4000 | `throughput_vs_size`, `cache_misses` |
| How well does the threadpool version scale? Do 16 workers help on 8 cores? | `BM_BlockedParallel`, 1 to 16 workers at n = 2000 | `thread_scaling` |
| Which tile size is best, and why? | `BM_Blocked` and `BM_BlockedParallel` (8 workers), tile edge 8 to 512 | `tile_sweep` |
| How far is this from a tuned library? | `BM_OpenBLAS`, same sizes and worker counts; `BM_AOCL` in its own suite (`build/bench_aocl`) | `headline` |

## Setup

```sh
sudo apt install libbenchmark-dev libbenchmark-tools libopenblas-dev
python3 -m pip install --user --break-system-packages matplotlib scipy   # charts; compare.py uses scipy
```

OpenBLAS is optional. Without it, `make bench` builds everything except `BM_OpenBLAS`.

Ubuntu 24.04 refuses `pip install --user` without `--break-system-packages`; that flag installs
into `~/.local` and leaves system packages alone (a virtual environment also works). Avoid
`apt install python3-matplotlib` if NumPy 2 is installed with pip: Ubuntu's matplotlib is built
against NumPy 1 and will not import next to it.

### AOCL-BLAS reference (optional)

OpenBLAS 0.3.26 has no kernels for Zen 5: on this CPU it picks Intel's Cooper Lake AVX-512 code.
AMD's AOCL-BLAS, its fork of BLIS, has kernels written for Zen 5 and is the stronger reference.
Build it from source into your home directory, no sudo:

```sh
git clone --depth 1 --branch 5.3.2 https://github.com/amd/blis ~/opt/src/aocl-blis-5.3.2
cd ~/opt/src/aocl-blis-5.3.2
./configure --prefix=$HOME/opt/aocl-blis-5.3.2 --enable-cblas --enable-threading=openmp zen5
make -j16 && make install                     # about 20 seconds
ln -sfn aocl-blis-5.3.2 ~/opt/aocl-blis       # the Makefile's default AOCL_DIR
```

AOCL-BLAS exports the same BLAS functions as OpenBLAS, so one program cannot link both.
`bench/aocl_bench.cpp` is therefore its own suite, `build/bench_aocl`, with `BM_AOCL` at the same
sizes and worker counts as `BM_OpenBLAS`. Each results file records the AOCL-BLAS version and the
CPU configuration it picked (`config=zen5`). Both suites time on the raw clock, so their files
can be compared directly. Run them back to back all the same, so that heat and background load
are alike.

## Running

| Command | What it does | Time |
|---|---|---|
| `make bench` | Build `build/bench` | |
| `make bench-quick` | One repetition without n = 3000 and 4000. Prints only. | ~2 min |
| `make bench-run` | Five interleaved repetitions, saved to `bench/results/<label>.json` | ~20 min |
| `make bench-plot` | Charts and `summary.md` from that file into `bench/plots/`, and the results block of the top-level README. `BASE=<label>` also draws an earlier run for comparison. | seconds |
| `make bench-compare BASE=<label>` | Compare two results files with a significance test | seconds |
| `make bench-aocl` | Build the AOCL-BLAS suite, `build/bench_aocl` (needs AOCL-BLAS, see Setup) | |
| `make bench-aocl-run` | Five interleaved repetitions of `BM_AOCL`, saved to `bench/results/<label>-aocl.json` | ~2 min |

`<label>` defaults to `git describe --always --dirty`, so a run is named after the commit it
measured (`22af068`, or `22af068-dirty` with uncommitted changes). Name a run yourself with
`LABEL=simd-v1`. Pass extra flags with `ARGS`, for example
`make bench-run ARGS=--benchmark_filter=BM_Blocked/` to run one family, and set the number of
repetitions with `BENCH_REPS=9`. `TILE=192` runs with another default tile than `kBlockSize`.

For clean numbers: close other programs, use the Windows "Best performance" power mode, pause
OneDrive syncing (this repository is inside OneDrive, which uploads every file written during
the run), and leave the machine alone. Then check the CV column; see "Noise" below.

## Methodology

- **Wall-clock time, on the raw clock.** The thread that calls `multiplyBlockedParallel` sleeps
  in `waitAll()` while the workers compute, so its CPU time would make the parallel version look
  many times faster than it is. Each iteration is timed with `bench::IterationTimer`, which reads
  `CLOCK_MONOTONIC_RAW` and reports to Google Benchmark through `UseManualTime()`. Google
  Benchmark's own wall clock is the one the system adjusts, and under WSL2 it has run from 7%
  fast to 3% slow; the raw clock matched the Windows clock to 0.01%. The CPU column is the CPU
  time of the whole process (`MeasureProcessCPUTime()`), so CPU / Time is the number of busy
  cores, and 1.00 on a single-threaded benchmark.
- **What is timed.** The same job for every implementation, the BLAS references included:
  allocate a zeroed result, compute the product into it, free the previous result. The public
  functions here return a new matrix, so callers pay for that. Inputs use fixed seeds and are
  built outside the timed loop. The pool is created outside it, so thread start-up is not timed.
- **No page faults in the timed loop.** By default glibc hands large freed blocks back to the
  kernel, and the next result then pays a page fault for every page. Whether that happened
  depended on the allocator's state, not on the code: at n = 2000 the same loop took 0 faults per
  call at tile 48 and about 5,100 at tile 256, which cost 3% on one thread and 25% on 16.
  `bench::recycleFreedMemory()` makes glibc keep freed memory, and
  `bench::warmUpResultMemory()` touches the result's blocks before the loop. The `faults/call`
  counter shows it worked: it should read 0.
- **Every result is checked.** After timing, `bench::productLooksCorrect` verifies C = A x B
  with Freivalds' algorithm: for a random vector x, A(Bx) must equal Cx up to a rounding-error
  bound. That takes O(n^2) time instead of O(n^3). A wrong result shows up as `ERROR OCCURRED`
  in place of a time, so a fast but broken kernel can never post a number.
- **GFLOP/s** = 2n^3 / time for every variant: one multiply and one add per inner step.
- **Repetitions are interleaved.** `bench-run` runs each benchmark five times in a random
  order across the whole suite, so slow drift (heat, background load) spreads over all
  benchmarks instead of landing on one. Results are medians.
- **No power-of-two sizes.** In calibration runs the current kernels were 5-6x slower at
  n = 1024 and 2048, with 50x more L1 misses, most likely because column walks of B that step
  by a power of two keep landing in the same few cache sets. That would hide every other effect.
  See `TODO.md` for a benchmark of it.
- **Speedup baselines.** Parallel speedup is measured against the best serial code
  (`multiplyBlocked`), not against the parallel code on one worker. The one-worker point shows
  the pool's own overhead. Efficiency divides speedup by the physical cores in use (at most 8).
- **The tile size** used everywhere except the tile sweep is `kBlockSize` in `bench_common.hpp`,
  picked from the tile sweep. Pick it again whenever `multiplyBlock` changes: `make bench-plot`
  names the best tile for one thread and for the pool, and warns when the default is more than
  5% behind. `make bench-run TILE=192` tries another tile without a rebuild.
- **OpenBLAS.** After a multithreaded call its idle threads busy-wait for about 60 ms, which
  would slow down whatever runs next, so its benchmark waits that out untimed. Its CPU kernel
  (`core=Cooperlake` on Zen 5, which is AVX-512) is recorded in each results file. OpenBLAS has
  no kernel written for Zen 5; its README says to use the Cooper Lake one there.

### Noise

Each results file has the coefficient of variation (CV = stddev / mean) for every benchmark,
and `summary.md` shows it; the charts draw the slowest-to-fastest range behind each point. Treat
a difference smaller than about twice the CV as noise. What the first full run on this machine
showed:

- The blocked kernel is steady: CV about 1% single-threaded, 2-8% multithreaded.
- Naive at n = 3000 and 4000 runs at one of two speeds about 25% apart, with the same cache-miss
  counts but different IPC, so the misses themselves got slower. The likely cause is address
  translation in the VM, since naive touches a new 4 KiB page on almost every load. The median
  hides this; treat those two points as +-20%.
- Runs that use all 16 hardware threads suffer most from background activity, because one
  delayed thread holds up the whole call. OpenBLAS at 16 threads had a CV of 33%. The pool's
  16-worker runs had 2%, because a shared task queue lets the other workers pick up the slack.
- The `GHz` counter explains some outliers: one slow repetition ran at 4.4 GHz instead of 5.1.

**Runs up to v3 were timed on the adjusted clock: their GFLOP/s shift between sessions, their
FLOP/cycle does not.** Until 2026-11-16 the benchmarks used Google Benchmark's wall clock, whose
rate WSL2 adjusts. Measured after the v2 run, against the Windows clock over 40 seconds, it ran
2.0% slow while process CPU time matched to 0.01%, and `adjtimex` showed `tick` moving between
9740 and 9990 (nominal 10000); on 2026-11-16 it ran 3.1 to 3.8% fast. CPU / Time on a
single-threaded benchmark shows the size of the shift: 0.93 in v1's run and 1.02 in v2's, so the
unchanged OpenBLAS kernel reads 121 and then 135 GFLOP/s at 26 FLOP/cycle. Compare those runs
with each other, and with newer ones, by FLOP/cycle. Those runs also timed OpenBLAS writing into
a result that already existed, and the matmul.hpp calls with whatever page faults the allocator
handed them. `make bench-plot` prints each file's clock and its CPU / wall ratio.

### The machine

AMD Ryzen 7 9800X3D: 8 Zen 5 cores with 2 hardware threads each (SMT), 48 KiB L1D and 1 MiB L2
per core, and 96 MiB of L3 shared by all cores (3D V-Cache). All 8 cores share one L3, so there
are no cross-cluster effects. Each core can start two 512-bit FMAs per cycle, which is
32 double-precision FLOP/cycle: about 160 GFLOP/s per core at 5 GHz. The `FLOP/cycle` counter
compares against that.

**WSL2:** Linux runs in a Hyper-V VM, and Windows decides which physical core runs each virtual
CPU. Pinning inside WSL does not change that: 8 threads pinned to CPUs 0-7 (which Linux reports
as 4 cores x 2 threads) ran as fast as 8 threads on the even CPUs (8 separate cores). So with 8
workers you cannot guarantee one thread per physical core, and some SMT sharing may be in the
numbers.

**Google Benchmark warning:** Ubuntu's package prints `Library was built as DEBUG`. That slows
only the library's own per-iteration bookkeeping, which is nanoseconds against iterations of
14 microseconds to 40 seconds here.

## Hardware counters

Every matmul benchmark reports these per FMA (multiply-add), read with `perf_event_open`
around the timed loop (`perf_counters.hpp`). Counting follows the pool's worker threads.
OpenBLAS reports them only single-threaded, because its threads exist before the counters do.

| Counter | Meaning |
|---|---|
| `GHz` | Cycles / CPU time: the average clock of the busy cores. The cycles are user-mode only and the CPU time includes the kernel's, so time spent in the kernel (page faults, a contended lock) reads as a lower clock. |
| `faults/call` | Page faults per timed call, from the kernel's bookkeeping. Should be 0; see Methodology. |
| `IPC` | Instructions per cycle, per hardware thread |
| `FLOP/cycle` | Single-threaded runs only. Zen 5 peak is 32. |
| `L1D_miss/FMA` | L1 data-cache read misses |
| `L2_miss/FMA` | Requests that missed L2 (AMD event 0x64; includes instruction fetches) |
| `L3_fill/FMA` | L1D fills served by L3 (AMD raw event 0x44, unit mask 0x02) |
| `DRAM_fill/FMA` | L1D fills served by DRAM (AMD raw event 0x44, unit mask 0x08) |

The L2, L3 and DRAM events are AMD-specific and only enabled on Zen 3 or newer. To check what
they count, a random pointer chase (each load depends on the previous one, so prefetchers cannot
help) was run over working sets of increasing size:

| Working set | L1D miss / load | L2 miss / load | L3 fill / load | DRAM fill / load | Cycles / load |
|---:|---:|---:|---:|---:|---:|
| 16 KiB | 0.00 | 0.00 | 0.00 | 0.00 | 4 |
| 256 KiB | 1.00 | 0.00 | 0.00 | 0.00 | 14 |
| 4 MiB | 1.00 | 0.92 | 0.93 | 0.00 | 57 |
| 32 MiB | 1.00 | 1.02 | 1.00 | 0.01 | 98 |
| 512 MiB | 1.00 | 2.50 | 1.48 | 0.99 | 551 |

Each level turns on exactly when the working set outgrows the cache before it. Counts above 1
at 512 MiB come from page-table walks, which miss too.

## Comparing versions (for the SIMD work)

1. Commit the version you want as the baseline and run `make bench-run`. The results land in
   `bench/results/<commit>.json`.
2. Change `multiplyBlock` and run `make bench-run` again, which writes
   `bench/results/<commit>-dirty.json` (or name it with `LABEL=`).
3. `make bench-compare BASE=<commit> LABEL=<commit>-dirty`

`compare.py` prints the change in time for every benchmark present in both files, with a
Mann-Whitney U test p-value; it wants at least 9 repetitions (`BENCH_REPS=9`) for that test.

For the strictest A/B test, run both builds in the same session, alternating, so drift hits
both equally:

```sh
git worktree add ../threadpool-base <baseline-commit>
make -C ../threadpool-base bench
python3 /usr/share/benchmark/compare.py benchmarks ../threadpool-base/build/bench ./build/bench \
    --benchmark_filter=BM_Blocked/ --benchmark_repetitions=9
```

Benchmarks are matched by name, and the name includes the tile size (`bs:48`). Compare at the
same tile size first, then re-run the tile sweep for the new kernel.

## Charts

`make bench-plot` writes each chart as `<name>.png` on a dark background, and
`bench/plots/summary.md` shows every chart above the numbers behind it as Markdown tables. A
chart that needs a benchmark the run did not include (for example `BM_Naive`, when filtered out)
is skipped, and so are table columns with no data.

To compare with an earlier version, add `BASE=<label>`, for example
`make bench-plot LABEL=v1 BASE=v0` (`--baseline` for `plot.py`). The charts then also draw that
run's blocked kernels as hollow markers and dashed lines, and the tables gain "vs v0" speedup
columns.

The collapsed results block in the top-level README is the same summary. `make bench-plot` rewrites
everything between the `<!-- results:start -->` and `<!-- results:end -->` lines (invisible on
GitHub) and nothing else, so edit outside them. To show results somewhere else, move the two
lines. Plotting with `python3 bench/plot.py` directly leaves the README alone unless you pass
`--readme README.md`.

## Files

| File | Contents |
|---|---|
| `matmul_bench.cpp` | Naive, blocked and parallel benchmarks, and `main()` |
| `openblas_bench.cpp` | OpenBLAS reference, built only if OpenBLAS is installed |
| `aocl_bench.cpp` | AOCL-BLAS reference and its own `main()`: the separate `build/bench_aocl` suite |
| `bench_common.hpp/.cpp` | Sizes, tile and worker lists, the Freivalds check, metric reporting, results-file context |
| `perf_counters.hpp/.cpp` | Hardware counter wrapper around `perf_event_open` |
| `plot.py` | Charts and `summary.md` from a results file |
| `results/` | One JSON file per run |
| `plots/` | Charts and `summary.md` from the last `make bench-plot` |
