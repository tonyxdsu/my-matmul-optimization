# Benchmarks

Measures the matrix multiply implementations in `include/matmul.hpp` with
[Google Benchmark](https://github.com/google/benchmark), with hardware performance counters
and OpenBLAS as a reference.

| Question | Benchmarks | Chart |
|---|---|---|
| How much faster is cache blocking than the naive loop, and why? | `BM_Naive`, `BM_Blocked` over n = 100 to 4000 | `throughput_vs_size`, `cache_misses` |
| How well does the threadpool version scale? Do 16 workers help on 8 cores? | `BM_BlockedParallel`, 1 to 16 workers at n = 2000 | `thread_scaling` |
| Which tile size is best, and why? | `BM_Blocked` and `BM_BlockedParallel` (8 workers), tile edge 8 to 512 | `tile_sweep` |
| How far is this from a tuned library? | `BM_OpenBLAS`, same sizes and worker counts; `BM_AOCL` in its own suite (`build/bench_aocl`) | `headline` |

## v1: loop order i-k-j

v1 changes one thing in `multiplyBlock`: the order of the loops inside a tile. The innermost loop
now walks along a row of C instead of summing down a column of B. For each k block of a tile:

```
v0:  for i, for j:   sum = 0;  for k: sum += A[i][k] * B[k][j];  C[i][j] += sum
v1:  for i, for k:   for j: C[i][j] += A[i][k] * B[k][j]
```

At n = 2000 and tile 48, single-threaded `multiplyBlocked` went from 7.04 to 12.3 GFLOP/s
(1.74x), and `multiplyBlockedParallel` on 16 workers from 59.9 to 99.4 GFLOP/s (1.66x). The
results below draw v0 as hollow markers and dashed lines. This run used v0's settings: the full
suite, 5 interleaved repetitions, the same flags (`-falign-loops=64` aligns code, not data) and
tile 48. The unchanged naive and OpenBLAS kernels came out within 3% of v0's run, except
OpenBLAS on 16 threads, which is always noisy.

| One thread, n = 2000, tile 48 | v0 | v1 |
|---|---:|---:|
| GFLOP/s | 7.04 | 12.3 |
| FLOP/cycle (peak 32) | 1.49 | 2.58 |
| Instructions per multiply-add | 9.35 | 1.95 |
| Cycles per multiply-add | 1.34 | 0.78 |
| IPC | 6.95 | 2.52 |

### 1. The inner loop vectorizes

GCC left v0's inner loop scalar for three reasons: the tail `break` gave it an early exit, it
summed into one variable (vectorizing that reorders the additions, which GCC does only with
`-ffast-math`), and it read B down a column, a whole row apart in memory. v1's inner loop reads a
row of B and a row of C, both contiguous, multiplies by A[i][k], which is the same for every j,
and gives each j its own element of C, so no addition is reordered. GCC compiles it to 512-bit
vector instructions, 8 doubles each.

![v0 visiting one 4 x 4 tile, one multiply-add at a time](docs/v1/traversal_v0.gif)

![v1 visiting one 4 x 4 tile, one row of C at a time](docs/v1/traversal_v1.gif)

The hot loops, from `objdump -d -C build/bench` (`sd` = scalar double, `pd` = packed double,
`zmm` = 512 bits). v0, one multiply-add per pass:

```
inc     %rax                      # k++
add     %rcx,%rdx                 # B pointer += one row
cmp     %rsi,%rax                 # the tail check
jae     ...
vmovsd  (%r15,%rax,8),%xmm0       # A[i][k]
vmulsd  (%rdx),%xmm0,%xmm0        # × B[k][j]
vaddsd  %xmm0,%xmm1,%xmm1         # sum += ...   (xmm1 on every pass: the chain)
cmp     %rdi,%rax                 # end of the k block?
jne     ...
```

v1, eight multiply-adds per pass:

```
vbroadcastsd (%r8),%zmm3                # before the loop: A[i][k] in all 8 lanes
vmovupd      (%r10,%r9,1),%zmm1         # 8 doubles of B's row k
vfmadd213pd  (%rdx,%r9,1),%zmm3,%zmm1   # × A[i][k], + 8 doubles of C's row i
vmovupd      %zmm1,(%rdx,%r9,1)         # store them to C
add          $0x40,%r9                  # next 64 bytes
cmp          %r9,0x110(%rsp)
jne          ...
```

### 2. The add chain is (semi-)removed

In v0 every add had to wait 2 cycles for the previous add's result in `sum`. In v1, consecutive
multiply-adds go to different elements of C, so they can be in flight at the same time. Each
element of C still gets its multiply-adds one after another, now as a load, an FMA and a store
per k, but 47 other elements are updated between two updates of the same one.

On its own this did not make v1 faster. v0's chains were only 48 adds long, and the out-of-order
core already overlapped neighbouring ones: 1.34 cycles per multiply-add, not the 2 of a single
chain. Built without SIMD (`-fno-tree-vectorize -fno-tree-slp-vectorize`,
`bench/results/v1-nosimd.json`), v1 drops to 3.04 GFLOP/s, below v0's 7.04, mostly because GCC's
scalar code for it keeps its loop pointers on the stack. The speedup comes from point 1; what
point 2 contributes is independent elements of C, which let point 1 happen without reordering.

### Notes

- **IPC fell from 6.95 to 2.52, as expected.** A vector instruction does 8 multiply-adds, so v1
  runs 4.8x fewer instructions per multiply-add, in 1.7x fewer cycles. Compare versions by
  FLOP/cycle, not IPC.
- **FMA came for free.** GCC's Zen tuning splits v0's multiply-add into `vmulsd` + `vaddsd` to
  keep the 4-cycle FMA out of the add chain. With no chain, v1 gets `vfmadd213pd`.
- **Alignment holds v1 back.** `std::vector` puts these matrices 16 bytes past a 64-byte
  boundary, so every 64-byte load and store of B and C straddles two cache lines. At tile 48, 2 of
  v1's 5 repetitions ran at 16.8-16.9 GFLOP/s, when C happened to land on a boundary, and 3 at
  12.0-12.3, hence the 17% CV. With all three matrices aligned, a scratch test ran v1 at 4.27
  FLOP/cycle instead of 2.60. Fixing that is v2.
- **v1 wants a bigger tile.** The comparison keeps v0's tile, 48, but v1's tile sweep peaks at
  256: 15.6 GFLOP/s on one thread and 119 on 8 workers, largely because a longer row spreads each
  row's setup over more vector FMAs (0.98 instructions per multiply-add instead of 1.95).
  `kBlockSize` should be re-picked for the next version.
- **Next bottleneck:** C still makes a round trip through memory for every k: two loads and a
  store per vector FMA.

<!-- results:start -->

## Benchmark results: v1

- Machine: AMD Ryzen 7 9800X3D 8-Core Processor; 8 cores, 16 hardware threads; L1d 48 KiB and L2 1 MiB per core, L3 96 MiB shared by all 8 cores
- Build: gcc 13.3.0, `-std=c++17 -Wall -Wextra -Iinclude -pthread -O3 -march=native -g -falign-loops=64`, git v1
- Reference: OpenBLAS 0.3.26 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=64 core=Cooperlake
- Run: 2026-10-24T17:31:44-07:00, median of 5 repetitions, wall-clock time, every result checked with Freivalds' algorithm
- Baseline: v0, run 2026-10-04T00:47:28-07:00. The charts draw its blocked kernels as hollow markers and dashed lines; "vs v0" columns are speedups over it.

### Headline: n = 2000, tile 48

![GFLOP/s of each implementation at n = 2000](bench/plots/headline.png)

| Implementation | Threads | Time (ms) | GFLOP/s | vs naive | vs v0 | vs OpenBLAS, same threads | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| multiplyNaive | 1 | 4,222 | 3.79 | 1x | - | 3.1% | 0.5% |
| multiplyBlocked | 1 | 1,305 | 12.3 | 3.23x | 1.74x | 10.1% | 17.3% |
| multiplyBlockedParallel | 8 | 217 | 73.7 | 19.4x | 1.56x | 14.1% | 2.5% |
| multiplyBlockedParallel | 16 | 161 | 99.4 | 26.2x | 1.66x | 15.6% | 1.5% |
| OpenBLAS dgemm | 1 | 132 | 121 | 32x | - | 100.0% | 0.7% |
| OpenBLAS dgemm | 8 | 30.7 | 522 | 138x | - | 100.0% | 2.4% |
| OpenBLAS dgemm | 16 | 25.1 | 637 | 168x | - | 100.0% | 29.9% |

### By matrix size, one thread

![GFLOP/s of naive and blocked by matrix size](bench/plots/throughput_vs_size.png)

![Cache misses per multiply-add by matrix size](bench/plots/cache_misses.png)

GFLOP/s, then hardware-counter events per multiply-add (N = naive, B = blocked).

| n | Naive | Blocked | Blocked / naive | Blocked vs v0 | OpenBLAS | L1D miss N | L1D miss B | L2 miss N | L2 miss B | DRAM fill N | DRAM fill B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 6.04 | 12.4 | 2.06x | 1.73x | 138 | 0.132 | 0.0216 | 7.93e-05 | 4.44e-05 | 1.22e-06 | 6.16e-07 |
| 200 | 5.63 | 13.1 | 2.33x | 1.84x | 107 | 0.13 | 0.0179 | 0.000567 | 0.000734 | 5.3e-06 | 9.38e-07 |
| 300 | 5.16 | 13.6 | 2.63x | 1.91x | 109 | 0.133 | 0.0167 | 0.000802 | 0.00126 | 1.73e-05 | 5.41e-06 |
| 500 | 4.81 | 13.3 | 2.76x | 1.84x | 116 | 0.158 | 0.0173 | 0.0272 | 0.00172 | 3.72e-05 | 1.49e-05 |
| 700 | 4.81 | 13.3 | 2.77x | 1.87x | 116 | 0.54 | 0.0153 | 0.0491 | 0.00241 | 7.51e-05 | 3.85e-05 |
| 1000 | 3.98 | 14.3 | 3.60x | 2.01x | 118 | 1.13 | 0.016 | 0.027 | 0.00422 | 9.75e-05 | 5.47e-05 |
| 1500 | 3.8 | 13.9 | 3.66x | 1.95x | 119 | 1.13 | 0.0152 | 0.0491 | 0.00503 | 2.24e-05 | 7.89e-05 |
| 2000 | 3.79 | 12.3 | 3.23x | 1.74x | 121 | 1.09 | 0.015 | 0.0661 | 0.00531 | 0.000199 | 0.000102 |
| 3000 | 1.79 | 12 | 6.70x | 1.70x | 122 | 1.13 | 0.0156 | 0.0864 | 0.00545 | 0.0135 | 0.000589 |
| 4000 | 1.11 | 11.9 | 10.77x | 1.71x | 123 | 1.16 | 0.0157 | 0.635 | 0.00532 | 0.0636 | 0.00101 |

### Thread scaling: n = 2000, tile 48

![Speedup by number of workers at n = 2000](bench/plots/thread_scaling.png)

Speedup is against single-threaded multiplyBlocked; efficiency divides it by the physical cores in use (at most 8). GHz is the average clock of the busy cores; IPC is per hardware thread.

| Workers | Time (ms) | GFLOP/s | vs v0 | Speedup | Efficiency | GHz | IPC | L1D miss/FMA | OpenBLAS GFLOP/s | OpenBLAS speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,426 | 11.2 | 1.57x | 0.915 | 92% | 5.06 | 2.22 | 0.0147 | 121 | 1 |
| 2 | 733 | 21.8 | 1.62x | 1.78 | 89% | 5.03 | 2.18 | 0.0207 | 230 | 1.9 |
| 4 | 386 | 41.5 | 1.60x | 3.39 | 85% | 5.09 | 2.04 | 0.0271 | 397 | 3.27 |
| 6 | 265 | 60.3 | 1.60x | 4.92 | 82% | 5.08 | 1.98 | 0.0315 | 506 | 4.18 |
| 8 | 217 | 73.7 | 1.56x | 6.01 | 75% | 5.03 | 1.83 | 0.0475 | 522 | 4.31 |
| 10 | 188 | 85.1 | 1.61x | 6.95 | 87% | 4.96 | 1.73 | 0.0624 | 606 | 5 |
| 12 | 176 | 91.1 | 1.59x | 7.43 | 93% | 4.98 | 1.54 | 0.0893 | 688 | 5.68 |
| 14 | 166 | 96.3 | 1.63x | 7.85 | 98% | 4.92 | 1.41 | 0.113 | 787 | 6.5 |
| 16 | 161 | 99.4 | 1.66x | 8.11 | 101% | 4.71 | 1.34 | 0.13 | 637 | 5.26 |

### Tile sweep: n = 2000

![GFLOP/s by tile size at n = 2000](bench/plots/tile_sweep.png)

| Tile | 1 thread GFLOP/s | vs v0 | IPC | L1D miss/FMA | L2 miss/FMA | 8 workers GFLOP/s | vs v0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5.49 | 0.82x | 4.78 | 0.0495 | 0.0154 | 18.2 | 0.66x |
| 16 | 8.04 | 1.14x | 3.73 | 0.0218 | 0.0106 | 35 | 0.92x |
| 24 | 9.83 | 1.40x | 3.29 | 0.0181 | 0.0105 | 49.2 | 1.13x |
| 32 | 10.9 | 1.56x | 2.95 | 0.0177 | 0.0083 | 60.9 | 1.30x |
| 48 | 12.3 | 1.74x | 2.52 | 0.015 | 0.00531 | 73.7 | 1.56x |
| 64 | 12.6 | 1.94x | 2.21 | 0.0539 | 0.00443 | 82 | 1.93x |
| 96 | 12 | 1.93x | 1.71 | 0.156 | 0.00335 | 87.7 | 2.16x |
| 128 | 13 | 2.19x | 1.67 | 0.149 | 0.00344 | 103 | 2.61x |
| 192 | 14.9 | 2.81x | 1.68 | 0.142 | 0.00489 | 115 | 3.23x |
| 256 | 15.6 | 3.05x | 1.6 | 0.138 | 0.0053 | 119 | 3.61x |
| 384 | 15.4 | 3.21x | 1.48 | 0.137 | 0.0356 | 106 | 3.66x |
| 512 | 15.2 | 3.49x | 1.38 | 0.138 | 0.0796 | 107 | 4.00x |

<!-- results:end -->

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
CPU configuration it picked (`config=zen5`). For a fair comparison, run the suites back to back
in one session rather than comparing files from different days.

## Running

| Command | What it does | Time |
|---|---|---|
| `make bench` | Build `build/bench` | |
| `make bench-quick` | One repetition without n = 3000 and 4000. Prints only. | ~2 min |
| `make bench-run` | Five interleaved repetitions, saved to `bench/results/<label>.json` | ~20 min |
| `make bench-plot` | Charts and `summary.md` from that file into `bench/plots/`, and the results section of this README. `BASE=<label>` also draws an earlier run for comparison. | seconds |
| `make bench-compare BASE=<label>` | Compare two results files with a significance test | seconds |
| `make bench-aocl` | Build the AOCL-BLAS suite, `build/bench_aocl` (needs AOCL-BLAS, see Setup) | |
| `make bench-aocl-run` | Five interleaved repetitions of `BM_AOCL`, saved to `bench/results/<label>-aocl.json` | ~2 min |

`<label>` defaults to `git describe --always --dirty`, so a run is named after the commit it
measured (`22af068`, or `22af068-dirty` with uncommitted changes). Name a run yourself with
`LABEL=simd-v1`. Pass extra flags with `ARGS`, for example
`make bench-run ARGS=--benchmark_filter=BM_Blocked/` to run one family, and set the number of
repetitions with `BENCH_REPS=9`.

For clean numbers: close other programs, use the Windows "Best performance" power mode, pause
OneDrive syncing (this repository is inside OneDrive, which uploads every file written during
the run), and leave the machine alone. Then check the CV column; see "Noise" below.

## Methodology

- **Wall-clock time.** Every benchmark uses `UseRealTime()`. The thread that calls
  `multiplyBlockedParallel` sleeps in `waitAll()` while the workers compute, so its CPU time
  would make the parallel version look many times faster than it is. The CPU column is the CPU
  time of the whole process (`MeasureProcessCPUTime()`), so CPU / Time is roughly the number of
  busy cores.
- **What is timed.** The public call, including allocating its result, because callers pay
  for that too. Inputs use fixed seeds and are built outside the timed loop. The pool is
  created outside it, so thread start-up is not timed.
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
  picked from the tile sweep. Pick it again whenever `multiplyBlock` changes.
- **OpenBLAS** computes into a result allocated once, the way BLAS is normally called. After a
  multithreaded call its idle threads busy-wait for about 60 ms, which would slow down whatever
  runs next, so its benchmark waits that out untimed. Its CPU kernel (`core=Cooperlake` on Zen
  5, which is AVX-512) is recorded in each results file.

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
| `GHz` | Cycles / CPU time: the average clock of the busy cores |
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

The results section near the top of this README is the same summary. `make bench-plot` rewrites
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
