// v4 in progress: how fast the register-blocked kernel runs where it already works.
//
// The v4 multiplyBlock keeps a 6 x 32 block of C (24 zmm registers) in registers for a whole k
// block. Until its edges are written it is only correct when every tile is a whole number of
// register blocks: rows a multiple of 6, columns a multiple of 32, so a tile edge that is a
// multiple of 96. The benchmark suite's sizes (100 to 4000) are not, so this program times the
// kernel on sizes that are, next to OpenBLAS's DGEMM on one thread in the same process.
//
// Both sides do the same job, C += A * B into a result that is already allocated, zeroed and
// mapped: the v4 side runs the tile loop of multiplyBlocked over multiplyBlock, and DGEMM is
// called with beta = 1. Calls are interleaved, and each row is the median of its repetitions.
// Time is CLOCK_MONOTONIC_RAW, cycles and instructions come from PerfCounters, and every
// configuration is first checked element by element against a plain i-k-j multiply.
//
// Run it with `make v4-fullblock`. Delete it once v4 handles edges and the suite can time it.

#include "matmul.hpp"
#include "perf_counters.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifdef HAVE_OPENBLAS
#include <cblas.h>
#endif

#ifndef EXP_CXXFLAGS
#define EXP_CXXFLAGS "unknown"
#endif

namespace {

// Every n and tile is a multiple of 96 = lcm(6 rows, 32 columns) of one register block.
const std::size_t kSizes[] = {96, 192, 480, 960, 1920, 2016};
const std::size_t kTiles[] = {96, 192, 384};
// 0 as a tile means OpenBLAS.
const std::size_t kOpenBLAS = 0;

double nowSeconds() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

// Plain i-k-j multiply, written here so the check does not depend on the code under test.
Matrix reference(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (std::size_t i = 0; i < A.rows; i++) {
        for (std::size_t k = 0; k < A.cols; k++) {
            const double a = A.data[i * A.cols + k];
            const double* b = &B.data[k * B.cols];
            double* c = &C.data[i * C.cols];
            for (std::size_t j = 0; j < B.cols; j++) c[j] += a * b[j];
        }
    }
    return C;
}

// C += A * B, the way multiplyBlocked walks its tiles.
void accumulateTiles(const Matrix& A, const Matrix& B, Matrix& C, std::size_t tile) {
    for (std::size_t r = 0; r < C.rows; r += tile) {
        for (std::size_t c = 0; c < C.cols; c += tile) {
            multiplyBlock(A, B, C, r, std::min(r + tile, C.rows), c, std::min(c + tile, C.cols), tile);
        }
    }
}

// C += A * B with whichever implementation `tile` names.
void accumulate(const Matrix& A, const Matrix& B, Matrix& C, std::size_t tile) {
    if (tile != kOpenBLAS) {
        accumulateTiles(A, B, C, tile);
        return;
    }
#ifdef HAVE_OPENBLAS
    const auto n = static_cast<blasint>(A.rows);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A.data.data(), n,
                B.data.data(), n, 1.0, C.data.data(), n);
#endif
}

double maxRelativeError(const Matrix& got, const Matrix& want) {
    double scale = 0, err = 0;
    for (std::size_t i = 0; i < want.data.size(); i++) {
        scale = std::max(scale, std::fabs(want.data[i]));
        err = std::max(err, std::fabs(got.data[i] - want.data[i]));
    }
    return err / scale;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Samples {
    std::vector<double> seconds, cycles, instructions;
};

}  // namespace

int main() {
    std::printf("v4 full-block check. Built with: %s\n", EXP_CXXFLAGS);
#ifdef HAVE_OPENBLAS
    openblas_set_num_threads(1);
    std::printf("OpenBLAS: %s core=%s, 1 thread\n", openblas_get_config(), openblas_get_corename());
#else
    std::printf("OpenBLAS: not built in\n");
#endif
    PerfCounters counters;
    std::printf("Counters: %s\n\n", counters.describe().c_str());
    std::printf("%6s %-9s %5s %9s %9s %11s %6s %10s  %s\n", "n", "kernel", "reps", "ms", "GFLOP/s",
                "FLOP/cycle", "IPC", "instr/FMA", "max rel err");

    bool allCorrect = true;
    for (const std::size_t n : kSizes) {
        const Matrix A = randomMatrix(n, n, 1), B = randomMatrix(n, n, 2);
        const Matrix want = reference(A, B);
        const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

        std::vector<std::size_t> configs;
        for (const std::size_t tile : kTiles) {
            if (tile <= n) configs.push_back(tile);
        }
#ifdef HAVE_OPENBLAS
        configs.push_back(kOpenBLAS);
#endif

        Matrix C(n, n);
        std::vector<double> errors;
        for (const std::size_t tile : configs) {
            std::memset(C.data.data(), 0, C.data.size() * sizeof(double));
            accumulate(A, B, C, tile);
            errors.push_back(maxRelativeError(C, want));
        }

        // About 2 seconds per configuration, at a guessed 100 GFLOP/s, and at least 7 calls.
        const int reps = std::clamp(static_cast<int>(2e11 / flops), 7, 5001) | 1;
        std::vector<Samples> samples(configs.size());
        for (int rep = 0; rep < reps; rep++) {
            for (std::size_t c = 0; c < configs.size(); c++) {
                // Untimed: the result starts at zero for both, and its pages stay mapped.
                std::memset(C.data.data(), 0, C.data.size() * sizeof(double));
                counters.start();
                const double t0 = nowSeconds();
                accumulate(A, B, C, configs[c]);
                const double t1 = nowSeconds();
                counters.stop();
                samples[c].seconds.push_back(t1 - t0);
                samples[c].cycles.push_back(counters.value(PerfCounters::Cycles));
                samples[c].instructions.push_back(counters.value(PerfCounters::Instructions));
            }
        }

        for (std::size_t c = 0; c < configs.size(); c++) {
            const double seconds = median(samples[c].seconds);
            const double cycles = median(samples[c].cycles);
            const double instructions = median(samples[c].instructions);
            // v4 does the same adds in the same order as the reference; DGEMM does not.
            const bool correct = errors[c] < 1e-12;
            allCorrect = allCorrect && correct;
            const std::string name = configs[c] == kOpenBLAS ? "OpenBLAS" : "tile " + std::to_string(configs[c]);
            std::printf("%6zu %-9s %5d %9.3f %9.1f %11.2f %6.2f %10.3f  %.1e%s\n", n, name.c_str(), reps,
                        seconds * 1e3, flops / seconds / 1e9, flops / cycles, instructions / cycles,
                        instructions / (flops / 2.0), errors[c], correct ? "" : "  <-- WRONG");
        }
        std::printf("\n");
    }
    return allCorrect ? 0 : 1;
}
