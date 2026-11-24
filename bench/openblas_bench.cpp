// Reference point: OpenBLAS's DGEMM, a hand-tuned assembly kernel, on the same inputs and with
// the same correctness check as the matmul.hpp benchmarks. Built only when OpenBLAS is
// installed (see the Makefile).
//
// Like the matmul.hpp functions, each timed call allocates a zeroed result and computes into it,
// so both sides are timed doing the same job. BLAS callers usually reuse one result, which would
// save DGEMM the allocation that the matmul.hpp functions cannot avoid.

#include "bench_common.hpp"
#include "matmul.hpp"
#include "perf_counters.hpp"

#include <benchmark/benchmark.h>
#include <cblas.h>

#include <chrono>
#include <string>
#include <thread>
#include <utility>

namespace {

// Records which OpenBLAS build and CPU kernel ran, since that decides how fast the reference is.
const bool kContextAdded = [] {
    benchmark::AddCustomContext("openblas", std::string(openblas_get_config()) + " core=" + openblas_get_corename());
    return true;
}();

// Args: n, workers.
void BM_OpenBLAS(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto workers = static_cast<int>(state.range(1));
    const auto dim = static_cast<blasint>(n);
    openblas_set_num_threads(workers);
    const Matrix A = randomMatrix(n, n, 1), B = randomMatrix(n, n, 2);
    Matrix C(0, 0);
    bench::warmUpResultMemory(n, n);

    // OpenBLAS's workers exist before these counters, so they are not counted. With one worker
    // the calling thread does all the work, so only those runs report hardware counters.
    PerfCounters counters;
    counters.start();
    for (auto _ : state) {
        {
            // The same job as the matmul.hpp calls: a new zeroed result, the product written
            // into it, and the previous result freed.
            bench::IterationTimer timer(state);
            Matrix product(n, n);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim,
                        1.0, A.data.data(), dim, B.data.data(), dim, 0.0, product.data.data(), dim);
            C = std::move(product);
        }
        benchmark::DoNotOptimize(C);
    }
    counters.stop();
    bench::finish(state, A, B, C, counters, workers == 1, workers == 1);

    // After a multithreaded call, OpenBLAS's idle workers busy-wait for about 60 ms before
    // sleeping. Wait them out, untimed, so they do not take cores from the next benchmark.
    if (workers > 1) std::this_thread::sleep_for(std::chrono::milliseconds(150));
}

// Single-threaded size sweep, plus thread scaling at fixedN().
void openblasArgs(benchmark::internal::Benchmark* b) {
    for (auto n : bench::sizes()) b->Args({n, 1});
    for (auto w : bench::workerCounts()) {
        const bool alreadyAdded = w == 1 && bench::contains(bench::sizes(), bench::fixedN());
        if (!alreadyAdded) b->Args({bench::fixedN(), w});
    }
}

BENCHMARK(BM_OpenBLAS)
    ->Apply(openblasArgs)
    ->ArgNames({"n", "workers"})
    ->UseManualTime()
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace
