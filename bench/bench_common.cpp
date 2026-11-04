#include "bench_common.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <utility>

#ifndef BENCH_CXXFLAGS
#define BENCH_CXXFLAGS "unknown"
#endif

namespace {

// The CPU model from /proc/cpuinfo. Google Benchmark records only core count and caches.
std::string cpuModel() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) != 0) continue;
        const auto start = line.find_first_not_of(" \t", line.find(':') + 1);
        if (start != std::string::npos) return line.substr(start);
    }
    return "unknown";
}

}  // namespace

namespace bench {

void addContext() {
#ifdef __clang__
    benchmark::AddCustomContext("compiler", "clang " __VERSION__);
#else
    benchmark::AddCustomContext("compiler", "gcc " __VERSION__);
#endif
    benchmark::AddCustomContext("cpu", cpuModel());
    benchmark::AddCustomContext("cxxflags", BENCH_CXXFLAGS);
    benchmark::AddCustomContext("block_size", std::to_string(kBlockSize));
    benchmark::AddCustomContext("perf_counters", PerfCounters().describe());
}

std::vector<std::int64_t> workerCounts() {
    const std::int64_t hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::int64_t> counts;
    for (std::int64_t w = 1; w <= hardwareThreads; w = w < 4 ? w * 2 : w + 2) {
        counts.push_back(w);
    }
    if (counts.back() != hardwareThreads) counts.push_back(hardwareThreads);
    return counts;
}

bool productLooksCorrect(const Matrix& A, const Matrix& B, const Matrix& C) {
    const std::size_t M = A.rows, K = A.cols, N = B.cols;
    if (B.rows != K || C.rows != M || C.cols != N) return false;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> x(N);
    for (double& v : x) v = dist(gen);

    // Bx, plus |B||x| for the rounding-error bound.
    std::vector<double> Bx(K, 0.0), absBx(K, 0.0);
    for (std::size_t k = 0; k < K; k++) {
        for (std::size_t j = 0; j < N; j++) {
            Bx[k] += B.at(k, j) * x[j];
            absBx[k] += std::fabs(B.at(k, j) * x[j]);
        }
    }

    // Each side carries rounding error of up to about (K + N) unit roundoffs times |A||B||x|,
    // in any summation order. Allow twice the total; a real bug is off by far more.
    const double tolerance = 2.0 * static_cast<double>(K + N) * std::numeric_limits<double>::epsilon();
    for (std::size_t i = 0; i < M; i++) {
        double ABx = 0.0, bound = 0.0, Cx = 0.0;
        for (std::size_t k = 0; k < K; k++) {
            ABx += A.at(i, k) * Bx[k];
            bound += std::fabs(A.at(i, k)) * absBx[k];
        }
        for (std::size_t j = 0; j < N; j++) Cx += C.at(i, j) * x[j];
        if (!(std::fabs(ABx - Cx) <= tolerance * bound)) return false;   // also rejects NaN
    }
    return true;
}

void finish(benchmark::State& state, const Matrix& A, const Matrix& B, const Matrix& C,
            const PerfCounters* counters, bool singleThreaded) {
    if (!productLooksCorrect(A, B, C)) {
        state.SkipWithError("wrong result: C != A * B (Freivalds check)");
        return;
    }

    const double fmasPerCall = static_cast<double>(A.rows) * static_cast<double>(A.cols) * static_cast<double>(B.cols);
    state.counters["FLOP/s"] = benchmark::Counter(2.0 * fmasPerCall, benchmark::Counter::kIsIterationInvariantRate);
    if (counters == nullptr) return;

    const double fmas = fmasPerCall * static_cast<double>(state.iterations());
    const double cycles = counters->value(PerfCounters::Cycles);
    if (cycles > 0) {
        if (counters->isOpen(PerfCounters::Instructions)) {
            state.counters["IPC"] = counters->value(PerfCounters::Instructions) / cycles;
        }
        if (counters->cpuSeconds() > 0) state.counters["GHz"] = cycles / counters->cpuSeconds() / 1e9;
        if (singleThreaded) state.counters["FLOP/cycle"] = 2.0 * fmas / cycles;
    }

    const std::pair<PerfCounters::Event, const char*> perFma[] = {
        {PerfCounters::L1DMisses, "L1D_miss/FMA"},
        {PerfCounters::L2Misses, "L2_miss/FMA"},
        {PerfCounters::L3Fills, "L3_fill/FMA"},
        {PerfCounters::DramFills, "DRAM_fill/FMA"},
    };
    for (const auto& [event, name] : perFma) {
        if (counters->isOpen(event)) state.counters[name] = counters->value(event) / fmas;
    }
}

}  // namespace bench
