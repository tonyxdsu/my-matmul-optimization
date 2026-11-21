// Checks and times the functions of the register-blocking assignment. exercises.cpp (yours) and
// solution.cpp (the reference) both include this file at their end, so it sees their templates;
// it defines main().
//
//   make regblock            build exercises.cpp, run every exercise pinned to one core
//   make regblock EX=3       run exercise 3 only
//   make regblock-solution   the same for solution.cpp
//
// Each exercise is first checked against a plain C++ version at several sizes, including awkward
// ones. Only a version that passes is timed. Times are in core cycles from PerfCounters (the same
// counters as the benchmarks), reported as FLOP per cycle: one FMA on one double is 2 FLOP, and a
// core's peak is 32 (2 FMA units x 8 doubles x 2).

#ifndef REGBLOCK_HARNESS_HPP
#define REGBLOCK_HARNESS_HPP

#include "perf_counters.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

#ifndef EXP_CXXFLAGS
#define EXP_CXXFLAGS "unknown"
#endif

namespace harness {

// Every buffer starts on a 64-byte cache line, as Matrix does since v2. From malloc's usual
// 16-byte alignment, each 64-byte load straddles two lines and the core starts only one such
// load per cycle instead of two, which would hide the limits these exercises are about.
template <class T>
struct Aligned64 {
    using value_type = T;
    Aligned64() = default;
    template <class U>
    Aligned64(const Aligned64<U>&) {}
    T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{64})); }
    void deallocate(T* p, std::size_t) { ::operator delete(p, std::align_val_t{64}); }
    bool operator==(const Aligned64&) const = default;
};

using Buffer = std::vector<double, Aligned64<double>>;

std::mt19937 gen(42);

Buffer randomVector(std::size_t n) {
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    Buffer v(n);
    for (double& x : v) x = dis(gen);
    return v;
}

// ---- Plain C++ versions: the answers each exercise is checked against ------------------------

// GCC vectorizes this loop by itself; exercise 1 times it next to yours.
[[gnu::noinline]] void axpyPlain(double a, const double* x, double* y, std::size_t n) {
    for (std::size_t i = 0; i < n; i++) y[i] += a * x[i];
}

// GCC keeps this loop scalar: vectorizing it would reorder the additions (v0's problem).
[[gnu::noinline]] double sumSquaresPlain(const double* x, std::size_t n) {
    double s = 0;
    for (std::size_t i = 0; i < n; i++) s += x[i] * x[i];
    return s;
}

double dotPlain(const double* x, const double* y, std::size_t n) {
    double s = 0;
    for (std::size_t i = 0; i < n; i++) s += x[i] * y[i];
    return s;
}

// Horner's rule, one polynomial at one point at a time.
void polyPlain(const double* coef, int degree, int numPolys, const double* x, int numPoints,
               double* y, int ldy) {
    for (int p = 0; p < numPolys; p++) {
        const double* c = coef + static_cast<std::size_t>(p) * (degree + 1);
        for (int n = 0; n < numPoints; n++) {
            double acc = c[degree];
            for (int k = degree - 1; k >= 0; k--) acc = acc * x[n] + c[k];
            y[static_cast<std::size_t>(p) * ldy + n] = acc;
        }
    }
}

// ---- Checking --------------------------------------------------------------------------------

// Sums in a different order round differently, so results are compared with a tolerance.
bool close(double got, double want, double tol = 1e-12) {
    return std::fabs(got - want) <= tol * std::max(1.0, std::fabs(want));
}

// Fills the y buffer with this before a call, so writes outside the rows and columns asked for
// (into the padding of ldy, or past the end) are caught.
constexpr double kUntouched = 12345.0;

// Runs `fn` (a function with polyPlain's signature) on numPolys x numPoints with padding after
// every row, and compares it with polyPlain. Prints the first difference and returns false.
template <class F>
bool checkPoly(F fn, int degree, int numPolys, int numPoints, const char* name) {
    const int ldy = numPoints + 5;
    const Buffer coef = randomVector(static_cast<std::size_t>(numPolys) * (degree + 1));
    const Buffer x = randomVector(numPoints);
    Buffer want(static_cast<std::size_t>(numPolys) * ldy + 3, kUntouched);
    Buffer got = want;
    polyPlain(coef.data(), degree, numPolys, x.data(), numPoints, want.data(), ldy);
    fn(coef.data(), degree, numPolys, x.data(), numPoints, got.data(), ldy);
    for (std::size_t i = 0; i < got.size(); i++) {
        const bool inside = static_cast<int>(i % ldy) < numPoints &&
                            static_cast<int>(i / ldy) < numPolys;
        const bool ok = inside ? close(got[i], want[i]) : got[i] == kUntouched;
        if (!ok) {
            std::printf("  %-14s FAIL at %d polys x %d points, degree %d: y[%zu] (poly %zu, point %zu)"
                        " is %g, want %g%s\n",
                        name, numPolys, numPoints, degree, i, i / ldy, i % ldy, got[i],
                        inside ? want[i] : kUntouched,
                        inside ? "" : " (outside the result: nothing should write here)");
            return false;
        }
    }
    return true;
}

// ---- Timing ----------------------------------------------------------------------------------

volatile double sink;  // keeps results the compiler would otherwise throw away

// Runs `run` (flopPerRun FLOP) once to warm up, then 7 trials of about 2e8 FLOP each, and returns
// the best trial's FLOP per cycle: the one least disturbed by interrupts and other programs.
template <class F>
double flopPerCycle(PerfCounters& counters, F run, double flopPerRun) {
    const long reps = std::max(1L, static_cast<long>(2e8 / flopPerRun));
    run();
    double best = 0;
    for (int trial = 0; trial < 7; trial++) {
        counters.start();
        for (long r = 0; r < reps; r++) run();
        counters.stop();
        best = std::max(best, flopPerRun * reps / counters.value(PerfCounters::Cycles));
    }
    return best;
}

// ---- Exercise 1 ------------------------------------------------------------------------------

template <class F>
bool checkAxpy(F fn, const char* name) {
    for (std::size_t n : {0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1000, 2048}) {
        const Buffer x = randomVector(n);
        Buffer want = randomVector(n + 3);
        for (std::size_t i = n; i < n + 3; i++) want[i] = kUntouched;
        Buffer got = want;
        axpyPlain(0.75, x.data(), want.data(), n);
        fn(0.75, x.data(), got.data(), n);
        for (std::size_t i = 0; i < n + 3; i++) {
            if (!close(got[i], want[i])) {
                std::printf("  %-14s FAIL at n = %zu: y[%zu] is %g, want %g%s\n", name, n, i,
                            got[i], want[i], i >= n ? " (past the end: nothing should write here)" : "");
                return false;
            }
        }
    }
    std::printf("  %-14s PASS (n = 0 to 2048, including sizes that are not a multiple of 8)\n", name);
    return true;
}

void exercise1(PerfCounters& counters) {
    std::printf("Exercise 1: axpy, y[i] += a * x[i]\n");
    const std::size_t n = 2048;  // x and y: 32 KiB together, inside the 48 KiB L1
    Buffer x = randomVector(n), y = randomVector(n);
    const double flop = 2.0 * n;
    const double plain = flopPerCycle(counters, [&] { axpyPlain(1e-9, x.data(), y.data(), n); }, flop);
    if (checkAxpy(axpy, "axpy")) {
        const double mine = flopPerCycle(counters, [&] { axpy(1e-9, x.data(), y.data(), n); }, flop);
        std::printf("  FLOP/cycle at n = %zu: axpy %.1f, the plain loop GCC vectorized %.1f\n", n, mine,
                    plain);
    }
    std::printf("  Bonus, masked tail:\n");
    if (checkAxpy(axpyMasked, "axpyMasked")) {
        const double mine = flopPerCycle(counters, [&] { axpyMasked(1e-9, x.data(), y.data(), n); }, flop);
        std::printf("  FLOP/cycle at n = %zu: axpyMasked %.1f\n", n, mine);
    } else {
        std::printf("  (the bonus is optional)\n");
    }
}

// ---- Exercise 2 ------------------------------------------------------------------------------

template <int ACC>
void exercise2Row(PerfCounters& counters, const Buffer& a, const Buffer& b, const Buffer& c) {
    // a for sumSquares (3840 doubles, 30 KiB), b and c for dot (1920 each, 30 KiB together):
    // all of it stays in L1, so the loop's own limits show, not the memory's.
    const bool okSq = close(sumSquares<ACC>(a.data(), a.size()), sumSquaresPlain(a.data(), a.size()));
    const bool okDot = close(dot<ACC>(b.data(), c.data(), b.size()), dotPlain(b.data(), c.data(), b.size()));
    std::printf("  %3d", ACC);
    if (okSq) {
        std::printf("  %14.1f", flopPerCycle(counters, [&] { sink = sumSquares<ACC>(a.data(), a.size()); },
                                             2.0 * a.size()));
    } else {
        std::printf("  %14s", "FAIL");
    }
    if (okDot) {
        std::printf("  %10.1f\n", flopPerCycle(counters, [&] { sink = dot<ACC>(b.data(), c.data(), b.size()); },
                                               2.0 * b.size()));
    } else {
        std::printf("  %10s\n", "FAIL");
    }
}

void exercise2(PerfCounters& counters) {
    std::printf("Exercise 2: chains and loads, FLOP/cycle by number of accumulators (peak 32)\n");
    const Buffer a = randomVector(3840), b = randomVector(1920), c = randomVector(1920);
    std::printf("  plain C++ sum of squares (scalar, one chain): %.1f\n",
                flopPerCycle(counters, [&] { sink = sumSquaresPlain(a.data(), a.size()); }, 2.0 * a.size()));
    std::printf("  ACC  sumSquares<ACC>  dot<ACC>\n");
    exercise2Row<1>(counters, a, b, c);
    exercise2Row<2>(counters, a, b, c);
    exercise2Row<4>(counters, a, b, c);
    exercise2Row<8>(counters, a, b, c);
    exercise2Row<12>(counters, a, b, c);
    exercise2Row<16>(counters, a, b, c);
}

// ---- Exercises 3 and 4: the polynomial bank --------------------------------------------------

// The timed problem: 48 polynomials of degree 256 at 1536 points. 48 is a multiple of every P
// below and 1536 of every 8 * V, so no shape has leftovers. The degree is high so each chain is
// long: with short chains the out-of-order core overlaps several of them, which blurs the
// one-chain limit (degree 64 lifts exercise 3 from about 4 to about 9 FLOP/cycle).
constexpr int kDegree = 256, kPolys = 48, kPoints = 1536;

struct PolyData {
    Buffer coef = randomVector(static_cast<std::size_t>(kPolys) * (kDegree + 1));
    Buffer x = randomVector(kPoints);
    Buffer y = Buffer(static_cast<std::size_t>(kPolys) * kPoints);
};

constexpr double kPolyFlop = 2.0 * kPolys * kPoints * kDegree;  // one FMA per Horner step

template <class F>
double timePoly(PerfCounters& counters, PolyData& d, F fn) {
    return flopPerCycle(
        counters,
        [&] { fn(d.coef.data(), kDegree, kPolys, d.x.data(), kPoints, d.y.data(), kPoints); },
        kPolyFlop);
}

void exercise3(PerfCounters& counters) {
    std::printf("Exercise 3: Horner with intrinsics, one polynomial x 8 points at a time\n");
    bool ok = true;
    for (int degree : {0, 1, 5, 64}) {
        ok = ok && checkPoly(polyOne, degree, 1, 8, "polyOne") && checkPoly(polyOne, degree, 3, 24, "polyOne") &&
             checkPoly(polyOne, degree, 5, 64, "polyOne");
    }
    if (!ok) return;
    std::printf("  %-14s PASS (degrees 0 to 64, up to 5 polynomials x 64 points)\n", "polyOne");
    PolyData d;
    std::printf("  FLOP/cycle, %d polynomials of degree %d at %d points: %.1f\n", kPolys, kDegree, kPoints,
                timePoly(counters, d, polyOne));
}

template <int P, int V>
void exercise4Row(PerfCounters& counters, PolyData& d) {
    bool ok = true;
    for (int degree : {0, 1, 7, 64}) {
        ok = ok && checkPoly(polyBlock<P, V>, degree, P, 8 * V, "polyBlock") &&
             checkPoly(polyBlock<P, V>, degree, 3 * P, 16 * V, "polyBlock");
    }
    std::printf("  %2d x %-2d  %10d", P, V, P * V + V + 1);
    if (ok) {
        std::printf("  %10.1f\n", timePoly(counters, d, polyBlock<P, V>));
    } else {
        std::printf("  %10s\n", "FAIL (see above)");
    }
}

void exercise4(PerfCounters& counters) {
    std::printf("Exercise 4: a block of P polynomials x V vectors of points kept in registers\n");
    std::printf("  P x V    zmm needed  FLOP/cycle   (zmm needed = P*V accumulators + V point vectors"
                " + 1 broadcast; the core has 32)\n");
    PolyData d;
    exercise4Row<1, 1>(counters, d);
    exercise4Row<1, 2>(counters, d);
    exercise4Row<1, 4>(counters, d);
    exercise4Row<1, 8>(counters, d);
    exercise4Row<2, 4>(counters, d);
    exercise4Row<4, 2>(counters, d);
    exercise4Row<4, 4>(counters, d);
    exercise4Row<6, 4>(counters, d);
    exercise4Row<8, 4>(counters, d);
    exercise4Row<4, 8>(counters, d);
}

// ---- Exercise 5 ------------------------------------------------------------------------------

void exercise5(PerfCounters& counters) {
    std::printf("Exercise 5: any number of polynomials and points (the edges)\n");
    const int sizes[][2] = {{1, 1},  {1, 7},  {1, 8},   {2, 9},    {3, 17},    {5, 13},
                            {7, 100}, {9, 33}, {13, 64}, {45, 1003}, {48, 1536}, {50, 1537}};
    for (int degree : {0, 3, 64}) {
        for (const auto& s : sizes) {
            if (!checkPoly(polyAny, degree, s[0], s[1], "polyAny")) return;
        }
    }
    std::printf("  %-14s PASS (1 x 1 up to 50 x 1537, degrees 0, 3 and 64)\n", "polyAny");
    PolyData d;
    std::printf("  FLOP/cycle at 48 x 1536 (no edges): %.1f\n", timePoly(counters, d, polyAny));
    const double flop = 2.0 * 45 * 1003 * kDegree;
    std::printf("  FLOP/cycle at 45 x 1003 (edges on both sides): %.1f\n",
                flopPerCycle(
                    counters,
                    [&] { polyAny(d.coef.data(), kDegree, 45, d.x.data(), 1003, d.y.data(), kPoints); },
                    flop));
}

}  // namespace harness

int main(int argc, char** argv) {
    PerfCounters counters;
    if (!counters.isOpen(PerfCounters::Cycles)) {
        std::printf("Needs the cycle counter. %s\n", counters.describe().c_str());
        return 1;
    }
    const int only = argc > 1 ? std::atoi(argv[1]) : 0;
    if (only < 0 || only > 5) {
        std::printf("usage: %s [exercise 1-5]\n", argv[0]);
        return 1;
    }
    std::printf("Built with: %s\n\n", EXP_CXXFLAGS);
    void (*const exercises[])(PerfCounters&) = {harness::exercise1, harness::exercise2, harness::exercise3,
                                                harness::exercise4, harness::exercise5};
    for (int e = 1; e <= 5; e++) {
        if (only == 0 || only == e) {
            exercises[e - 1](counters);
            std::printf("\n");
        }
    }
}

#endif
