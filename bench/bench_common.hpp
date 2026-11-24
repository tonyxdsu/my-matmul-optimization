#ifndef BENCH_COMMON_HPP
#define BENCH_COMMON_HPP

#include "matmul.hpp"
#include "perf_counters.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

/**
 * Settings and helpers shared by every benchmark, so all variants run on the same sizes and
 * report the same metrics.
 */
namespace bench {

/**
 * Matrix sizes for the size sweep. None is a power of two: at n = 1024 or 2048 the column walks
 * of B touch only a few cache sets, which slows the current kernels 5-6x and would hide every
 * other effect. Three n x n matrices of doubles outgrow a 1 MiB L2 at n ~ 200 and a 96 MiB L3
 * at n ~ 2000.
 */
constexpr std::int64_t kSizes[] = {100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000};

/**
 * Matrix size for the thread-scaling benchmark and the tile sweep.
 */
constexpr std::int64_t kFixedN = 2000;

/**
 * Default tile edge for everything except the tile sweep. Pick it again from the tile sweep
 * whenever multiplyBlock changes: `make bench-plot` prints the best tiles and warns when this one
 * has fallen behind. With the scalar i-j-k kernel at n = 2000, 32 and 48 tied within noise
 * (about 1%) for 1 and 8 workers. 48 makes fewer tasks, and its 18 KiB tile of B still fits in
 * the 48 KiB L1 when two SMT threads share a core.
 */
constexpr std::int64_t kBlockSize = 48;

/**
 * @return The tile edge of this run: the BENCH_TILE environment variable if it holds a positive
 *         integer (`make bench-run TILE=192`), otherwise kBlockSize. Lets a run try
 *         another tile without a rebuild. The value is recorded in the results file.
 */
std::int64_t blockSize();

/**
 * Tile edges for the tile sweep.
 */
constexpr std::int64_t kTileSizes[] = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};

/**
 * The three functions below let one run use other sizes without a rebuild, for a kernel that
 * only handles some shapes (`make bench-run SIZES=96,192,480 FIXED_N=2016 TILES=96,192,384`).
 * Each list is comma-separated positive integers. The sizes that ran are in the results file as
 * the benchmarks' own arguments.
 */

/**
 * @return Matrix sizes for the size sweep: the BENCH_SIZES environment variable if it holds a
 *         list, otherwise kSizes.
 */
const std::vector<std::int64_t>& sizes();

/**
 * @return Matrix size for the thread-scaling benchmark and the tile sweep: the BENCH_FIXED_N
 *         environment variable if it holds a positive integer, otherwise kFixedN.
 */
std::int64_t fixedN();

/**
 * @return Tile edges for the tile sweep: the BENCH_TILES environment variable if it holds a list,
 *         otherwise kTileSizes.
 */
const std::vector<std::int64_t>& tileSizes();

/**
 * Worker count for the parallel tile sweep: one per physical core of the 9800X3D.
 */
constexpr std::int64_t kTileSweepWorkers = 8;

/**
 * @return Worker counts for the scaling benchmark: 1, 2, 4, then every even count up to the
 *         number of hardware threads, which is included.
 */
std::vector<std::int64_t> workerCounts();

/**
 * @return Seconds on CLOCK_MONOTONIC_RAW, the clock every benchmark is timed with. Google
 *         Benchmark's own wall clock (std::chrono::steady_clock, which is CLOCK_MONOTONIC) gets its
 *         rate adjusted to track the host: under WSL2 it has run from 7% fast to 3% slow, changing
 *         within minutes, which moved every GFLOP/s figure by that much. The raw clock counts the
 *         CPU's timestamp counter at a fixed rate and matched the Windows clock to 0.01%.
 */
double rawSeconds();

/**
 * Times one iteration on rawSeconds() and reports it to Google Benchmark. Create it at the top of
 * a block that holds exactly the work to time; the benchmark must be registered with
 * UseManualTime().
 */
class IterationTimer {
public:
    /**
     * Starts timing.
     *
     * @param state The state of the benchmark whose iteration this is.
     */
    explicit IterationTimer(benchmark::State& state) : state(state), start(rawSeconds()) {}

    /**
     * Stops timing and adds the elapsed time to the benchmark's manual time.
     */
    ~IterationTimer() { state.SetIterationTime(rawSeconds() - start); }

    IterationTimer(const IterationTimer&) = delete;
    IterationTimer& operator=(const IterationTimer&) = delete;

private:
    /**
     * The benchmark being timed.
     */
    benchmark::State& state;

    /**
     * rawSeconds() when this iteration started.
     */
    double start;
};

/**
 * Makes the C library keep freed memory and hand it out again, so a result allocated inside a
 * timed loop lands on pages the process has already touched. Call first in main().
 *
 * By default glibc returns large freed blocks to the kernel, and every page of the next one
 * costs a page fault when it is first written. Whether that happens depends on the allocator's
 * state, not on the code being measured: at n = 2000 the same loop took 0 page faults per
 * iteration at tile 48 and about 5,100 at tile 256, worth 3% of the time on one thread and 25% on
 * 16. A program that multiplies repeatedly reaches the recycling state on its own.
 */
void recycleFreedMemory();

/**
 * Touches the memory that a timed loop's results will come from. A loop of `C = multiply(...)`
 * cycles through up to four blocks of the result's size, so this allocates four zeroed matrices
 * of that size and frees them again; with recycleFreedMemory() the loop then reuses their pages.
 * Call before starting the counters. Without it, the first iterations of a benchmark that runs
 * only one or two would still pay for fresh pages.
 *
 * @param rows Rows of the result.
 * @param cols Columns of the result.
 */
void warmUpResultMemory(std::size_t rows, std::size_t cols);

/**
 * @return True if value equals an element of range.
 */
template <class Range, class T>
bool contains(const Range& range, const T& value) {
    return std::find(std::begin(range), std::end(range), value) != std::end(range);
}

/**
 * Checks that C equals A * B in O(n^2) time, using Freivalds' algorithm: for a random vector
 * x, A(Bx) must equal Cx up to rounding error. An indexing bug, a missed tile or a data race
 * changes Cx by many orders of magnitude more than rounding can, so this catches wrong
 * results without an O(n^3) reference multiply.
 *
 * @param A Left operand, M x K.
 * @param B Right operand, K x N.
 * @param C The result to check, M x N.
 * @return True if C has the right shape and passes the check.
 */
bool productLooksCorrect(const Matrix& A, const Matrix& B, const Matrix& C);

/**
 * Records the compiler, the CPU model, the build flags, the default tile and the open hardware
 * counters in the results file's context, so every file says what it measured. Call after
 * benchmark::Initialize.
 */
void addContext();

/**
 * Ends a matmul benchmark. Checks the last result with productLooksCorrect and marks the
 * benchmark as failed if it is wrong. Otherwise reports FLOP/s and page faults per call, which
 * should be 0 (see recycleFreedMemory). If the hardware events saw every working thread, it also
 * reports IPC, the average clock speed of the busy cores, and cache misses per multiply-add
 * (FMA). FLOP/cycle is reported only for single-threaded runs, where all cycles belong to one
 * core.
 *
 * @param state The benchmark state, after its timing loop.
 * @param A Left operand that was multiplied.
 * @param B Right operand that was multiplied.
 * @param C The result of the last timed iteration.
 * @param counters Counters started and stopped around the timing loop.
 * @param singleThreaded True if one thread did all the work.
 * @param hardwareEvents False if some working threads existed before counters did, so its
 *                       hardware events missed them. Page faults are counted either way.
 */
void finish(benchmark::State& state, const Matrix& A, const Matrix& B, const Matrix& C,
            const PerfCounters& counters, bool singleThreaded, bool hardwareEvents = true);

}  // namespace bench

#endif
