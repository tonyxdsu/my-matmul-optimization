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
 * Tile edge used everywhere except the tile sweep. Pick it again from the tile sweep whenever
 * multiplyBlock changes. With the scalar i-j-k kernel at n = 2000, 32 and 48 tied within noise
 * (about 1%) for 1 and 8 workers. 48 makes fewer tasks, and its 18 KiB tile of B still fits in
 * the 48 KiB L1 when two SMT threads share a core.
 */
constexpr std::int64_t kBlockSize = 48;

/**
 * Tile edges for the tile sweep.
 */
constexpr std::int64_t kTileSizes[] = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};

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
 * benchmark as failed if it is wrong. Otherwise reports FLOP/s and, if counters is given,
 * IPC, the average clock speed of the busy cores, and cache misses per multiply-add (FMA).
 * FLOP/cycle is reported only for single-threaded runs, where all cycles belong to one core.
 *
 * @param state The benchmark state, after its timing loop.
 * @param A Left operand that was multiplied.
 * @param B Right operand that was multiplied.
 * @param C The result of the last timed iteration.
 * @param counters Counters started and stopped around the timing loop, or nullptr.
 * @param singleThreaded True if one thread did all the work.
 */
void finish(benchmark::State& state, const Matrix& A, const Matrix& B, const Matrix& C,
            const PerfCounters* counters, bool singleThreaded);

}  // namespace bench

#endif
