// Second reference point: AMD's AOCL-BLAS DGEMM (AMD's fork of BLIS, with kernels written for
// Zen 5), on the same inputs and with the same correctness check as the matmul.hpp benchmarks.
//
// AOCL-BLAS and OpenBLAS export the same BLAS functions, so one program cannot link both. This
// file is its own suite, build/bench_aocl, with its own main(); the Makefile keeps it out of
// build/bench. Like the OpenBLAS benchmark and the matmul.hpp functions, each timed call
// allocates a zeroed result and computes into it, so every side is timed doing the same job.

#include "bench_common.hpp"
#include "matmul.hpp"
#include "perf_counters.hpp"

#include <benchmark/benchmark.h>
#include <blis.h>    // AOCL-BLAS: version, CPU configuration, thread count
#include <cblas.h>   // AOCL-BLAS's own CBLAS header, found through -I$(AOCL_DIR)/include/blis

#include <chrono>
#include <string>
#include <thread>
#include <utility>

namespace {

// Args: n, workers.
void BM_AOCL(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto workers = static_cast<dim_t>(state.range(1));
    const auto dim = static_cast<int>(n);
    bli_thread_set_num_threads(workers);
    const Matrix A = randomMatrix(n, n, 1), B = randomMatrix(n, n, 2);
    Matrix C(0, 0);
    bench::warmUpResultMemory(n, n);

    // AOCL-BLAS's OpenMP threads may already exist from an earlier benchmark, so they are not
    // always counted. With one worker the calling thread does all the work, so only those runs
    // report hardware counters.
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

    // OpenMP's idle threads spin for a while before sleeping. Wait them out, untimed, so they do
    // not take cores from the next benchmark.
    if (workers > 1) std::this_thread::sleep_for(std::chrono::milliseconds(150));
}

// The same points as BM_OpenBLAS: the single-threaded size sweep, plus thread scaling at fixedN().
void aoclArgs(benchmark::internal::Benchmark* b) {
    for (auto n : bench::sizes()) b->Args({n, 1});
    for (auto w : bench::workerCounts()) {
        const bool alreadyAdded = w == 1 && bench::contains(bench::sizes(), bench::fixedN());
        if (!alreadyAdded) b->Args({bench::fixedN(), w});
    }
}

BENCHMARK(BM_AOCL)
    ->Apply(aoclArgs)
    ->ArgNames({"n", "workers"})
    ->UseManualTime()
    ->MeasureProcessCPUTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace

int main(int argc, char** argv) {
    bench::recycleFreedMemory();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    bench::addContext();
    // Which AOCL-BLAS build ran, and which CPU configuration it picked at run time.
    benchmark::AddCustomContext("aocl", std::string(bli_info_get_version_str()) + " config=" +
                                            bli_arch_string(bli_arch_query_id()));

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
