<div align="center">

# Matrix multiplication, one optimization at a time

A learning project: a hand-written thread pool, and a matmul kernel taken from the textbook
triple loop toward BLAS speed, one measured change per version.

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![GCC 13.3](https://img.shields.io/badge/GCC-13.3-555555?style=flat-square&logo=gnu&logoColor=white)
![AVX-512](https://img.shields.io/badge/SIMD-AVX--512-3987e5?style=flat-square)
![Zen 5](https://img.shields.io/badge/CPU-Zen%205-d95926?style=flat-square&logo=amd&logoColor=white)
![Progress](https://img.shields.io/badge/progress-v4%20in%20progress-2ea043?style=flat-square)

[At a glance](#versions-at-a-glance) ·
[v0](#v0-cache-blocking) ·
[v1](#v1-loop-interchange) ·
[v2](#v2-cache-line-aligned-storage) ·
[v3](#v3-hoist-loop-invariants) ·
[v4*](#v4-register-blocking-in-progress) ·
[Thread pool](#the-thread-pool) ·
[Results](#results) ·
[Run it](#run-it) ·
[Method](#how-it-is-measured)

</div>

![FLOP per cycle of each version on one thread: naive 0.80, v0 1.49, v1 2.58, v2 4.31, v3 6.10, v4* 29.1, OpenBLAS 26.0, against a limit of 32. v4* is in progress and is measured at n = 2016, tile 192, where OpenBLAS runs at 25.8](docs/progress.png)

v3, the last finished version, is about **7.5x** faster than the naive loop on one thread and
**58x** faster on the pool's 16 workers, at about a quarter of OpenBLAS's speed.

**v4\*** is in progress. Its kernel so far handles only matrices made of whole register blocks. On
those it runs at **28.5 to 29.9 FLOP/cycle** on one thread, 3.9x v3's kernel at n = 2016. From
n = 192 to 4032 it is 6 to 33% faster than single-threaded OpenBLAS.
[What the asterisk covers](#v4-register-blocking-in-progress) is spelled out below.

Every number comes from [Google Benchmark](https://github.com/google/benchmark) with hardware
performance counters, on a Ryzen 7 9800X3D.

## Versions at a glance

One thread, n = 2000, tile 48. FLOP/cycle counts multiplies and adds per CPU cycle; this core's
limit is 32.

| | What changed | FLOP/cycle | Step | Instructions per multiply-add | IPC |
|---|---|---:|---:|---:|---:|
| naive | Textbook i-j-k loop | 0.80 | | 7.01 | 2.82 |
| [v0](#v0-cache-blocking) | Work in 48 x 48 tiles, so the data in use stays in cache | 1.49 | 1.85x | 9.35 | 6.95 |
| [v1](#v1-loop-interchange) | Loop order i-k-j, which lets GCC vectorize with AVX-512 | 2.58 | 1.73x | 1.95 | 2.52 |
| [v2](#v2-cache-line-aligned-storage) | Matrices start on a 64-byte cache line | 4.31 | 1.67x | 1.70 | 3.67 |
| [v3](#v3-hoist-loop-invariants) | Loop invariants computed once, outside the loops | 6.10 | 1.42x | 1.33 | 4.05 |
| [v4\*](#v4-register-blocking-in-progress) | A 6 x 32 block of C stays in registers for a whole k block | 29.1\* | 3.89x\* | 0.20 | 2.91 |
| OpenBLAS | Reference: hand-written AVX-512 kernels | 26.0 | | 0.23 | 2.94 |

\* v4 is in progress and handles whole register blocks only, which n = 2000 and tile 48 are not.
Its row is n = 2016, tile 192, where v3's kernel runs at 7.48 FLOP/cycle and OpenBLAS at 25.8.

Each section below has the same parts: what changed, what it bought, why, and the bottleneck it
left for the next version.

## v0: cache blocking

`multiplyNaive` reads a whole column of B for every element of C, so from about n = 1000 nearly
every multiply-add misses L1.

![The naive loop working through a 12 x 12 multiply, one element of C at a time](docs/v0/naive.gif)

`multiplyBlocked` finishes C one tile at a time instead. Each step adds one tile of A times one
tile of B into a tile of C, and at tile 48 a tile is 18 KiB. `multiplyBlockedParallel` runs the
same tiles of C as tasks on the thread pool.

![v0 working through the same 12 x 12 multiply with tile edge 4, one tile of C at a time](docs/v0/blocking.gif)

**Result.** At n = 2000, L1 misses per multiply-add drop from 1.09 to 0.014, and FLOP/cycle rises
from 0.80 to 1.49 (1.85x). Blocked holds about 7 GFLOP/s from n = 100 to 4000, while naive falls
from 6.22 to 1.12.

> [!NOTE]
> **Next bottleneck.** The inner loop is scalar. It does one multiply-add per pass, and each add
> waits for the one before it. IPC is a high 6.95, but that buys only 1.49 FLOP/cycle.

## v1: loop interchange

One change in `multiplyBlock`: the order of the loops inside a tile. The innermost loop now walks
along a row of C instead of summing down a column of B.

```
v0:  for i, for j:   sum = 0;  for k: sum += A[i][k] * B[k][j];  C[i][j] += sum
v1:  for i, for k:   for j: C[i][j] += A[i][k] * B[k][j]
```

![v0 and v1 stepping through the same 4 x 32 tile: v0 makes 512 passes of its inner loop, one multiply-add each, while v1's j loop crosses each row in 4 passes of 8, 64 passes in all](docs/v1/same_work_wide.gif)

**Result.** 1.49 to 2.58 FLOP/cycle (1.73x) on one thread, and 59.9 to 99.4 GFLOP/s on 16 workers.

**Why.**

- **The inner loop vectorizes.** GCC left v0's loop scalar for three reasons: the tail `break`
  gave it an early exit, it summed into one variable (vectorizing that reorders the additions,
  which GCC does only with `-ffast-math`), and it read B down a column, a whole row apart in
  memory. v1 reads a row of B and a row of C, both contiguous, and gives each j its own element of
  C, so nothing is reordered. GCC compiles it to 512-bit instructions, 8 doubles each.
- **The add chain is gone, but that alone bought nothing.** v0's chains were only 48 adds long and
  the out-of-order core already overlapped them. Built without SIMD, v1 is slower than v0: 3.04
  against 7.04 GFLOP/s. What the independent elements of C give is permission to vectorize.
- **IPC fell from 6.95 to 2.52, and that is fine.** One vector instruction does 8 multiply-adds,
  so v1 runs 4.8x fewer instructions per multiply-add in 1.7x fewer cycles. Compare versions by
  FLOP/cycle, not IPC.

<details>
<summary><b>The two hot loops, from <code>objdump</code></b></summary>

`sd` = scalar double, `pd` = packed double, `zmm` = 512 bits. v0 does one multiply-add per pass.
GCC's Zen tuning splits it into `vmulsd` + `vaddsd` to keep the 4-cycle FMA out of the add chain:

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

v1 does eight per pass, and with no chain it gets a real FMA:

```
vbroadcastsd (%r8),%zmm3                # before the loop: A[i][k] in all 8 lanes
vmovupd      (%r10,%r9,1),%zmm1         # 8 doubles of B's row k
vfmadd213pd  (%rdx,%r9,1),%zmm3,%zmm1   # × A[i][k], + 8 doubles of C's row i
vmovupd      %zmm1,(%rdx,%r9,1)         # store them to C
add          $0x40,%r9                  # next 64 bytes
cmp          %r9,0x110(%rsp)
jne          ...
```

</details>

> [!NOTE]
> **Next bottleneck.** `std::vector` puts these matrices 16 bytes past a 64-byte boundary, so
> every 64-byte load and store straddles two cache lines. Of v1's 5 repetitions, 2 ran at
> 16.8 GFLOP/s, when C happened to land on a boundary, and 3 at 12.1.

## v2: cache-line-aligned storage

No change to the source of `multiplyBlock`, only to where the matrices live. `Matrix::data` now
uses an `AlignedAllocator` ([include/matmul.hpp](include/matmul.hpp)) that asks aligned
`operator new` for 64-byte alignment, so every matrix starts on a cache line.

![v1's 64-byte loads cross cache lines because the data starts at page + 16; v2's stay inside one line because the data starts at page + 64](docs/v2/alignment.png)

**Result.** 2.58 to 4.31 FLOP/cycle (1.67x). The run-to-run variation (CV) drops from 17.3% to
0.8%: v1's two speeds are gone.

**Why.**

- **Where the 16 bytes came from.** glibc's `malloc` puts a 16-byte header just before the
  pointer it returns, and it gives a large block fresh pages from `mmap`. So the data starts at
  page + 16, which is 16 bytes into a cache line.
- **What that costs.** The kernel loads 8 doubles at a time, and from that offset every 64-byte
  load needs two cache accesses. The core can start two aligned loads per cycle but only one
  split load (measured: 0.50 and 1.00 cycles per load), and v1 does two loads per vector FMA.
- **Why v1 ran at two speeds.** After the first large `free`, glibc raises its `mmap` threshold,
  and later matrices come from the ordinary heap at whatever multiple of 16 its history leaves. C
  alternated between 0 and 16 mod 64.
- **The check.** The same v2 object files, linked with an `operator new` that returns 16 mod 64
  on purpose, ran at 2.61 FLOP/cycle against 4.30, with identical instruction counts.
- **Not every size gets the full gain.** All rows start on a line only when n is a multiple of 8.
  At n = 100, 300, 500, 700 and 1500 every other row starts 32 bytes into a line, and those sizes
  reach 2.8-3.3 FLOP/cycle against 4.0-4.4 elsewhere.

> [!NOTE]
> **Next bottleneck.** Each pass of the k loop updates one row of a tile: 48 multiply-adds. v2
> runs 81 instructions for it, and only 36 of them are the vector loop. The rest is setup that
> GCC repeats every time.

## v3: hoist loop invariants

`multiplyBlock` computes the same thing in the same loop order. Values that stay the same through
an inner loop are now worked out once, before it:

- the data pointers of A, B and C and the widths of A and B, read from the `Matrix` objects once
- the end of each k block, computed once with `std::min` instead of an `if (...) break` on every k
- the row pointers of A and C, once per i
- `A[i][k]` and the row pointer of B, once per k, before the j loop

![Instructions in one row update at n = 2000, tile 48: v2 runs 36 in the vector loop and 45 of setup, 81 in all; v3 runs the same 36 and 27 of setup, 63 in all](docs/v3/bookkeeping.png)

**Result.** 4.31 to 6.10 FLOP/cycle (1.42x) on one thread, and 167 to 233 GFLOP/s on 16 workers.

**Why.** The vector loop is the same 6 instructions per 8 columns as in v1 and v2. What shrank is
the setup GCC runs around it once per row update (one i and one k, which at tile 48 is 48
multiply-adds, or 6 passes of the vector loop):

| Instructions per row update | v2 | v3 |
|---|---:|---:|
| Vector loop, 6 passes of 6 | 36 | 36 |
| Overlap checks | 14 | 5 |
| Reloads and spills | 9 | 2 |
| Length checks | 8 | 6 |
| Address arithmetic | 7 | 3 |
| Loop control | 6 | 3 |
| Load `A[i][k]` into all 8 lanes | 1 | 2 |
| Padding (`nop`) | 0 | 6 |
| **Total** | **81** | **63** |
| Total / 48 multiply-adds | 1.69 | 1.31 |
| Measured per multiply-add | 1.70 | 1.33 |

- **Overlap checks, 14 to 5.** GCC cannot prove that storing to C's row leaves `A[i][k]` and B's
  row unchanged, so before the vector loop it checks that they don't overlap. Copied into a
  local, `A[i][k]` needs no check, which leaves only B's row against C's.
- **Reloads and spills, 9 to 2.** On every row update v2 read `&A` and `&B` from the stack, then
  `A.data`, `B.data` and `B.cols` from the `Matrix` objects. v3 reads the objects once per call.
- **Address arithmetic and loop control, 13 to 6.** v2 multiplied k by B's width on every row
  update and tested two ways out of the k loop. v3 steps its pointers by a fixed amount and has
  one test.
- **Cycles fell further than instructions:** 30% fewer cycles per multiply-add against 22% fewer
  instructions. The setup that went away had been holding the FMAs up, not running beside them.

The 6 `nop`s are padding from `-falign-loops=64`, which starts the vector loop on a 64-byte
boundary.

> [!NOTE]
> **Next bottleneck.** C still makes a round trip through memory for every k: two loads and a
> store per vector FMA. v3 reaches 6.10 of a possible 32 FLOP/cycle.

## v4: register blocking (in progress)

`multiplyBlock` is now written with AVX-512 intrinsics. It keeps a 6 x 32 block of C in 24 `zmm`
registers for a whole k block: loaded once, updated by 24 FMAs for each k, and stored once.

```
v3:  for i, for k:
         for j, 8 at a time:   C[i][j] += A[i][k] * B[k][j]        C is loaded and stored for every FMA

v4:  for i, 6 at a time:   for j, 32 at a time:
         load the 6 x 32 block of C into 24 registers
         for k:   every register += its row's A[i][k] * its 8 columns of B[k]
         store the 24 registers
```

**Result\*.** On matrices made of whole register blocks, 28.5 to 29.9 FLOP/cycle on one thread,
3.8 to 4.0x v3's kernel from n = 288 up, and 864 GFLOP/s on 8 workers. One thread, tile 192:

| n | v3, FLOP/cycle | v4\*, FLOP/cycle | OpenBLAS, FLOP/cycle | v4\* against OpenBLAS, by time |
|---|---:|---:|---:|---:|
| 96 | 8.90 | 29.51 | 30.02 | 0.99x |
| 192 | 10.45 | 29.37 | 21.66 | 1.33x |
| 480 | 7.71 | 29.91 | 24.36 | 1.21x |
| 960 | 7.63 | 29.85 | 24.99 | 1.17x |
| 2016 | 7.48 | 29.11 | 25.75 | 1.09x |
| 4032 | 7.50 | 28.50 | 26.22 | 1.06x |

The lead over OpenBLAS is smaller by time than by FLOP/cycle (1.09x against 1.13x at n = 2016),
because the counters read a clock about 3% lower during v4's runs: 4.79 against 4.92 GHz.

**Why.**

- **C stays out of memory.** v3 did two loads and a store for every vector FMA. One pass of v4's k
  loop does 24 FMAs with 10 loads and no store: 6 values of A, one per row of the block, and 4
  vectors of B, one per 8 columns.
- **Fewer instructions.** That pass is 38 instructions for 192 multiply-adds, 0.20 each, against
  v3's 1.33 at tile 48 and 0.90 at tile 192.
- **Enough FMAs in flight.** The core can start 2 FMAs per cycle and each takes 4 cycles, so it
  needs at least 8 that do not wait on each other. The 24 registers are 24 separate chains.
- **It just fits.** 24 registers of C, 6 of A and 1 of B make 31 of the 32 `zmm` registers. The
  compiled k loop never touches the stack.

<details>
<summary><b>The k loop, from <code>objdump</code></b></summary>

One pass, for one k. `zmm0` is the vector of B, `zmm1` to `zmm6` are the 6 values of A, and the
other 24 registers are the block of C:

```
vmovupd      zmm0,[rcx]              # B[k], columns j to j+7
vbroadcastsd zmm6,[rsi+rax*8]        # column k of A in all 8 lanes, for one row of the block
vbroadcastsd zmm5,[r13+rax*8]        # and for the other 5 rows
vbroadcastsd zmm4,[r10+rax*8]
vbroadcastsd zmm3,[r11+rax*8]
vbroadcastsd zmm2,[rbx+rax*8]
vbroadcastsd zmm1,[r12+rax*8]
inc          rax                     # k++
vfmadd231pd  zmm12,zmm6,zmm0         # 6 rows of C, columns j to j+7
vfmadd231pd  zmm9,zmm5,zmm0
vfmadd231pd  zmm23,zmm4,zmm0
vfmadd231pd  zmm24,zmm3,zmm0
vfmadd231pd  zmm18,zmm2,zmm0
vfmadd231pd  zmm25,zmm1,zmm0
vmovupd      zmm0,[rcx+0x40]         # B[k], columns j+8 to j+15
...                                  # 6 more FMAs, and the same again for j+16 and j+24
add          rcx,r9                  # B pointer += one row
cmp          rdx,rax                 # end of the k block?
jne          ...
```

</details>

> [!WARNING]
> **What the asterisk covers.**
> - **Whole register blocks only.** The kernel is correct when every tile's height is a multiple
>   of 6 and its width a multiple of 32, so n and the tile must be multiples of 96. The code for
>   the leftover rows and columns is not written yet. At any other size `multiplyBlocked` and
>   `multiplyBlockedParallel` on this branch return wrong results, and `make test` fails.
> - **Sizes chosen to fit.** The suite's usual sizes are not multiples of 96, so v4\*'s run uses
>   n = 96 to 4032 in multiples of 96, with tile 192
>   ([the command](bench/README.md#running)). v3's kernel was rerun at the same sizes as the
>   baseline, `v3-same-sizes`. The numbers at n = 2000 and tile 48 will come with the edge code.
> - **Small matrices.** At n = 96 OpenBLAS is level or ahead, 30.0 against 29.5 FLOP/cycle.
> - **OpenBLAS on many threads.** Its multi-thread results are erratic under WSL2, so read the
>   pool's lead over it with care. See [the thread pool](#the-thread-pool).
> - **A second check agrees.** `make v4-fullblock`
>   ([experiments/v4_fullblock.cpp](experiments/v4_fullblock.cpp)) times the kernel and OpenBLAS
>   pinned to one core, both adding A x B into a result that is already allocated. At n = 2016 it
>   reads 28.4 against 25.6 FLOP/cycle.

**Still to do.** The edge code, then the suite at its usual sizes and a new default tile.

## The thread pool

`Threadpool` ([include/threadpool.hpp](include/threadpool.hpp)) is a fixed set of workers that
take `std::function` tasks from one shared queue, with `enqueue` and `waitAll`.
`multiplyBlockedParallel` enqueues one task per tile of C, so no two tasks write the same elements.

![Speedup by number of workers at n = 2016](bench/plots/thread_scaling.png)

With v4\*'s kernel at n = 2016 and tile 192, 8 workers run 6.2x faster than the single-threaded
code and 16 workers 7.1x, on 8 cores with 2 hardware threads each: 864 and 987 GFLOP/s. v3's
kernel scaled about the same (6.0x and 7.1x), so the pool carries the kernel's 3.9x through to
every worker count.

- **The second hardware thread adds little.** Going from 8 to 16 workers adds 14%. Both threads of
  a core share its two FMA units, and one thread of this kernel already keeps them 91% busy.
- **Against OpenBLAS.** In this run OpenBLAS reaches 542 GFLOP/s on 8 threads and 649 on 16.
  Its 16-thread repetitions ranged from 455 to 702, and the v3 run measured 842, so that number
  is not a steady one on this machine. On one thread, where it is steady, the pool's kernel leads
  by 9%.

## Results

![GFLOP/s of each implementation at n = 2016](bench/plots/headline.png)

This is v4\*'s run, on the sizes its kernel handles: n = 96 to 4032 in multiples of 96, at tile
192. The hollow markers are v3's kernel rerun at the same sizes and tile. v3's own run at
n = 2000 and tile 48 is in [bench/results/v3.json](bench/results/v3.json).

- **The tile matters less than it did.** On one thread, tiles 96, 192 and 288 are within 2% of
  each other, and larger tiles fall up to 13% behind. With 8 workers tile 192 is the fastest.
- **Size barely matters.** One thread holds 137 to 145 GFLOP/s from n = 96 to 4032.
- **Ignore "vs naive" here.** The naive loop is unusually slow at n = 2016: 2.11 GFLOP/s, against
  4.02 at n = 2000. Compare v4\* with v3 and OpenBLAS instead.

<details>
<summary><b>Every chart and table from the latest run</b> (by matrix size, thread scaling, tile sweep)</summary>

<!-- results:start -->

## Benchmark results: v4-partial

- Machine: AMD Ryzen 7 9800X3D 8-Core Processor; 8 cores, 16 hardware threads; L1d 48 KiB and L2 1 MiB per core, L3 96 MiB shared by all 8 cores
- Build: gcc 13.3.0, `-std=c++20 -Wall -Wextra -Iinclude -pthread -O3 -march=native -g -falign-loops=64`, git v4-partial
- Reference: OpenBLAS 0.3.26 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=64 core=Cooperlake
- Run: 2026-11-23T20:31:16-07:00, median of 5 repetitions, wall-clock time on the raw clock (CPU / wall on one thread: 1.00), every result checked with Freivalds' algorithm
- Baseline: v3-same-sizes, run 2026-11-23T20:46:07-07:00, wall-clock time on the raw clock (CPU / wall on one thread: 1.00). The charts draw its blocked kernels as hollow markers and dashed lines; "vs v3-same-sizes" columns are speedups over it in time, so they carry any difference between the two clocks. FLOP/cycle does not.

### Headline: n = 2016, tile 192

![GFLOP/s of each implementation at n = 2016](bench/plots/headline.png)

| Implementation | Threads | Time (ms) | GFLOP/s | vs naive | vs v3-same-sizes | vs OpenBLAS, same threads | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| multiplyNaive | 1 | 7,763 | 2.11 | 1x | - | 1.7% | 19.9% |
| multiplyBlocked | 1 | 118 | 138 | 65.6x | 3.78x | 109.0% | 2.0% |
| multiplyBlockedParallel | 8 | 19 | 864 | 409x | 3.93x | 159.3% | 3.6% |
| multiplyBlockedParallel | 16 | 16.6 | 987 | 467x | 3.79x | 152.1% | 1.3% |
| OpenBLAS dgemm | 1 | 129 | 127 | 60.1x | - | 100.0% | 1.2% |
| OpenBLAS dgemm | 8 | 30.2 | 542 | 257x | - | 100.0% | 2.0% |
| OpenBLAS dgemm | 16 | 25.3 | 649 | 307x | - | 100.0% | 19.6% |

### By matrix size, one thread

![GFLOP/s of naive and blocked by matrix size](bench/plots/throughput_vs_size.png)

![Cache misses per multiply-add by matrix size](bench/plots/cache_misses.png)

GFLOP/s, then hardware-counter events per multiply-add (N = naive, B = blocked).

| n | Naive | Blocked | Blocked / naive | Blocked vs v3-same-sizes | OpenBLAS | L1D miss N | L1D miss B | L2 miss N | L2 miss B | DRAM fill N | DRAM fill B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 6.71 | 145 | 21.57x | 3.37x | 147 | 0.132 | 0.0276 | 0.000261 | 9.48e-06 | 2.03e-06 | 1.53e-08 |
| 192 | 3.39 | 142 | 41.98x | 2.96x | 107 | 1.03 | 0.0326 | 0.000911 | 0.000173 | 2.69e-06 | 2.07e-08 |
| 288 | 4.68 | 142 | 30.24x | 3.90x | 116 | 1.03 | 0.028 | 0.00319 | 0.000334 | 1.33e-05 | 1.28e-07 |
| 480 | 4.64 | 145 | 31.24x | 3.84x | 120 | 1.04 | 0.0281 | 0.0888 | 0.000332 | 0.000192 | 7.39e-07 |
| 672 | 4.57 | 143 | 31.38x | 3.83x | 121 | 1.04 | 0.028 | 0.0609 | 0.000333 | 0.000227 | 3.02e-06 |
| 960 | 3.01 | 143 | 47.58x | 3.89x | 122 | 1.03 | 0.0265 | 0.0194 | 0.000348 | 5.3e-05 | 1.5e-05 |
| 1440 | 4.33 | 142 | 32.85x | 3.84x | 125 | 1.05 | 0.028 | 0.0618 | 0.000375 | 4.29e-05 | 2.72e-05 |
| 2016 | 2.11 | 138 | 65.58x | 3.78x | 127 | 1.16 | 0.0283 | 0.105 | 0.000311 | 0.00102 | 7.42e-05 |
| 2880 | 1.7 | 139 | 81.68x | 3.82x | 130 | 1.21 | 0.0264 | 0.492 | 0.00031 | 0.0189 | 0.000325 |
| 4032 | 1.36 | 137 | 100.74x | 3.72x | 129 | 1.21 | 0.0265 | 0.723 | 0.000306 | 0.0937 | 0.000454 |

### Thread scaling: n = 2016, tile 192

![Speedup by number of workers at n = 2016](bench/plots/thread_scaling.png)

Speedup is against single-threaded multiplyBlocked; efficiency divides it by the physical cores in use (at most 8). GHz is the average clock of the busy cores; IPC is per hardware thread.

| Workers | Time (ms) | GFLOP/s | vs v3-same-sizes | Speedup | Efficiency | GHz | IPC | L1D miss/FMA | OpenBLAS GFLOP/s | OpenBLAS speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 117 | 140 | 3.87x | 1.01 | 101% | 4.81 | 2.89 | 0.0283 | 127 | 1 |
| 2 | 60.9 | 269 | 3.79x | 1.94 | 97% | 4.71 | 2.87 | 0.0283 | 236 | 1.86 |
| 4 | 32.3 | 508 | 3.75x | 3.67 | 92% | 4.64 | 2.78 | 0.0283 | 406 | 3.2 |
| 6 | 22.8 | 719 | 3.95x | 5.19 | 87% | 4.56 | 2.71 | 0.0283 | 503 | 3.96 |
| 8 | 19 | 864 | 3.93x | 6.24 | 78% | 4.51 | 2.49 | 0.0283 | 542 | 4.27 |
| 10 | 17.9 | 914 | 3.74x | 6.6 | 83% | 4.44 | 2.18 | 0.0283 | 615 | 4.85 |
| 12 | 17.1 | 959 | 3.78x | 6.93 | 87% | 4.37 | 1.94 | 0.0283 | 713 | 5.62 |
| 14 | 16.6 | 988 | 3.78x | 7.13 | 89% | 4.32 | 1.76 | 0.0283 | 790 | 6.23 |
| 16 | 16.6 | 987 | 3.79x | 7.13 | 89% | 4.09 | 1.66 | 0.0283 | 649 | 5.11 |

### Tile sweep: n = 2016

![GFLOP/s by tile size at n = 2016](bench/plots/tile_sweep.png)

Best tile: 1 thread 288 (141 GFLOP/s, +2% against tile 192); 8 workers 192 (864 GFLOP/s, +0% against tile 192).

| Tile | 1 thread GFLOP/s | vs v3-same-sizes | IPC | L1D miss/FMA | L2 miss/FMA | 8 workers GFLOP/s | vs v3-same-sizes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 96 | 140 | 3.99x | 2.93 | 0.0262 | 0.000471 | 839 | 4.02x |
| 192 | 138 | 3.78x | 2.91 | 0.0283 | 0.000311 | 864 | 3.93x |
| 288 | 141 | 3.98x | 2.92 | 0.0272 | 0.000312 | 769 | 3.99x |
| 384 | 130 | 4.44x | 2.7 | 0.0265 | 0.000634 | 763 | 4.55x |
| 480 | 125 | 4.39x | 2.62 | 0.0262 | 0.000728 | 680 | 4.71x |
| 576 | 125 | 4.43x | 2.6 | 0.0262 | 0.000715 | 665 | 5.01x |
| 768 | 122 | 4.39x | 2.53 | 0.0259 | 0.000873 | 601 | 4.54x |

<!-- results:end -->

</details>

## Run it

```sh
sudo apt install libgtest-dev libbenchmark-dev libbenchmark-tools libopenblas-dev

make && make run      # the demo app, build/app
make test             # GoogleTest suites for the pool and every multiply
make bench-quick      # one repetition of the benchmarks, about 2 minutes
make bench-run        # the full run, about 20 minutes, saved to bench/results/<label>.json
make bench-plot       # charts and tables from that file
```

OpenBLAS is optional. [bench/README.md](bench/README.md) has the rest: every command and option,
the AOCL-BLAS reference suite, comparing two versions, and what each counter means.

## How it is measured

- **Machine.** AMD Ryzen 7 9800X3D: 8 Zen 5 cores, 16 hardware threads, 48 KiB L1D and 1 MiB L2
  per core, 96 MiB of shared L3. GCC 13.3 with `-O3 -march=native`, Ubuntu under WSL2.
- **The limit.** A core can start two 512-bit FMAs per cycle. Each does 8 multiplies and 8 adds,
  which makes 32 FLOP/cycle.
- **Timing.** Wall-clock time on `CLOCK_MONOTONIC_RAW`, the median of 5 repetitions interleaved
  across the whole suite. GFLOP/s = 2n³ / time.
- **Sizes.** v0 to v3 are quoted at n = 2000 and tile 48. v4\* cannot run those yet, so it is
  quoted at n = 2016 and tile 192, with v3 and OpenBLAS measured again at that size.
- **Every result is checked** with Freivalds' algorithm after timing, so a fast but broken kernel
  cannot post a number.
- **Counters** come from `perf_event_open` around the timed loop, reported per multiply-add:
  instructions, cycles, L1 and L2 misses, L3 and DRAM fills.
- **Compare versions by FLOP/cycle.** Runs up to v3 used a clock that WSL2 adjusts, so their
  GFLOP/s shift by up to 10% between sessions. Their FLOP/cycle does not.
- **No power-of-two sizes.** At n = 1024 and 2048 these kernels ran 5-6x slower, most likely from
  cache-set conflicts, which would hide every other effect.

The full methodology and its noise notes are in [bench/README.md](bench/README.md).

## Repository layout

| Path | Contents |
|---|---|
| [include/](include/), [src/](src/) | `Threadpool`, `Matrix` and the multiply variants |
| [tests/](tests/) | GoogleTest suites |
| [bench/](bench/) | Benchmark harness, hardware counters, `plot.py`, results and charts |
| [docs/](docs/) | The GIFs and diagrams on this page, each with the script that draws it |
| [experiments/](experiments/) | Standalone programs that each measure one CPU effect |
| [exercises/register_blocking/](exercises/register_blocking/) | Toy problems that lead up to v4 |
