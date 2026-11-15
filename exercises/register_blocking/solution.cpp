// Register-blocking assignment: reference solution. Read it after your own version passes, or
// when you are stuck. The instructions are in ASSIGNMENT.md.
//
//   make regblock-solution          every exercise
//   make regblock-solution EX=4     one exercise

#include <immintrin.h>

#include <cstddef>

// =============================================================================================
// Exercise 1: y[i] += a * x[i]
// =============================================================================================

[[gnu::noinline]] void axpy(double a, const double* x, double* y, std::size_t n) {
    const __m512d av = _mm512_set1_pd(a);  // a in all 8 lanes, once, outside the loop
    std::size_t i = 0;
    // `i + 8 <= n`, not `i < n - 8`: n is unsigned, so n - 8 would wrap around when n < 8.
    for (; i + 8 <= n; i += 8) {
        const __m512d xv = _mm512_loadu_pd(x + i);
        const __m512d yv = _mm512_loadu_pd(y + i);
        _mm512_storeu_pd(y + i, _mm512_fmadd_pd(av, xv, yv));  // av * xv + yv, one rounding
    }
    for (; i < n; i++) y[i] += a * x[i];  // the last n % 8 elements, one at a time
}

// Bonus: the last n % 8 elements with a mask instead of a scalar loop.
[[gnu::noinline]] void axpyMasked(double a, const double* x, double* y, std::size_t n) {
    const __m512d av = _mm512_set1_pd(a);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm512_storeu_pd(y + i, _mm512_fmadd_pd(av, _mm512_loadu_pd(x + i), _mm512_loadu_pd(y + i)));
    }
    if (i < n) {
        // Bit j of the mask switches lane j on. n - i is 1 to 7 here, so this sets that many low bits.
        const __mmask8 m = static_cast<__mmask8>((1u << (n - i)) - 1);
        // Lanes that are off are not read (maskz fills them with 0) and not written, so touching
        // memory past the end of x and y is safe: a masked-off lane cannot fault.
        const __m512d xv = _mm512_maskz_loadu_pd(m, x + i);
        const __m512d yv = _mm512_maskz_loadu_pd(m, y + i);
        _mm512_mask_storeu_pd(y + i, m, _mm512_fmadd_pd(av, xv, yv));
    }
}

// =============================================================================================
// Exercise 2: chains and loads. n is a multiple of 8 * ACC.
// =============================================================================================

// Each accumulator is a separate chain: an FMA into acc[a] waits for the previous FMA into acc[a],
// but not for the others. With ACC chains, ACC FMAs can be in flight at once.
template <int ACC>
[[gnu::noinline]] double sumSquares(const double* x, std::size_t n) {
    __m512d acc[ACC];
    // The pragma makes GCC unroll the loop completely, so every acc[a] becomes its own register.
    // Without full unrolling, acc would have to live in memory, indexed at run time.
#pragma GCC unroll 16
    for (int a = 0; a < ACC; a++) acc[a] = _mm512_setzero_pd();

    for (std::size_t i = 0; i < n; i += 8 * ACC) {
#pragma GCC unroll 16
        for (int a = 0; a < ACC; a++) {
            const __m512d v = _mm512_loadu_pd(x + i + 8 * a);  // 1 load per FMA
            acc[a] = _mm512_fmadd_pd(v, v, acc[a]);
        }
    }

    // Combine the chains, then the 8 lanes of what is left.
#pragma GCC unroll 16
    for (int a = 1; a < ACC; a++) acc[0] = _mm512_add_pd(acc[0], acc[a]);
    return _mm512_reduce_add_pd(acc[0]);
}

// The same, with 2 loads per FMA. The core starts 2 vector loads per cycle, so this loop can start
// at most 1 FMA per cycle however many chains it has: half of sumSquares' best.
template <int ACC>
[[gnu::noinline]] double dot(const double* x, const double* y, std::size_t n) {
    __m512d acc[ACC];
#pragma GCC unroll 16
    for (int a = 0; a < ACC; a++) acc[a] = _mm512_setzero_pd();

    for (std::size_t i = 0; i < n; i += 8 * ACC) {
#pragma GCC unroll 16
        for (int a = 0; a < ACC; a++) {
            acc[a] = _mm512_fmadd_pd(_mm512_loadu_pd(x + i + 8 * a), _mm512_loadu_pd(y + i + 8 * a), acc[a]);
        }
    }

#pragma GCC unroll 16
    for (int a = 1; a < ACC; a++) acc[0] = _mm512_add_pd(acc[0], acc[a]);
    return _mm512_reduce_add_pd(acc[0]);
}

// =============================================================================================
// Exercise 3: Horner with intrinsics, one polynomial x 8 points at a time.
// y[p * ldy + n] = sum over k of coef[p * (degree + 1) + k] * x[n]^k. numPoints % 8 == 0.
// =============================================================================================

[[gnu::noinline]] void polyOne(const double* coef, int degree, int numPolys, const double* x,
                               int numPoints, double* y, int ldy) {
    const int stride = degree + 1;  // coefficients per polynomial
    for (int p = 0; p < numPolys; p++) {
        const double* c = coef + p * stride;
        for (int n = 0; n < numPoints; n += 8) {
            const __m512d xv = _mm512_loadu_pd(x + n);
            __m512d acc = _mm512_set1_pd(c[degree]);
            // One chain: every FMA needs the one before it, so each step waits the full FMA
            // latency (4 cycles) for 16 FLOP.
            for (int k = degree - 1; k >= 0; k--) {
                acc = _mm512_fmadd_pd(acc, xv, _mm512_set1_pd(c[k]));
            }
            _mm512_storeu_pd(y + p * ldy + n, acc);
        }
    }
}

// =============================================================================================
// Exercise 4: a block of P polynomials x V vectors of points in registers.
// numPolys % P == 0 and numPoints % (8 * V) == 0.
// =============================================================================================

// Per Horner step the block does P broadcasts and P * V FMAs: each broadcast coefficient feeds V
// FMAs, and each point vector feeds P. Registers: P * V accumulators + V point vectors + 1 for the
// broadcast. Past 32 the compiler has to spill some of them to the stack.
template <int P, int V>
[[gnu::noinline]] void polyBlock(const double* coef, int degree, int numPolys, const double* x,
                                 int numPoints, double* y, int ldy) {
    const int stride = degree + 1;
    for (int p0 = 0; p0 < numPolys; p0 += P) {
        for (int n0 = 0; n0 < numPoints; n0 += 8 * V) {
            // The block's points, loaded once and used at every step by all P polynomials.
            __m512d xv[V];
#pragma GCC unroll 16
            for (int v = 0; v < V; v++) xv[v] = _mm512_loadu_pd(x + n0 + 8 * v);

            // Each accumulator starts at its polynomial's leading coefficient.
            __m512d acc[P][V];
#pragma GCC unroll 16
            for (int p = 0; p < P; p++) {
                const __m512d c = _mm512_set1_pd(coef[(p0 + p) * stride + degree]);
#pragma GCC unroll 16
                for (int v = 0; v < V; v++) acc[p][v] = c;
            }

            // The P * V accumulators are P * V independent chains, held in registers for the whole
            // loop. Nothing is stored until the block is finished.
            for (int k = degree - 1; k >= 0; k--) {
#pragma GCC unroll 16
                for (int p = 0; p < P; p++) {
                    const __m512d c = _mm512_set1_pd(coef[(p0 + p) * stride + k]);
#pragma GCC unroll 16
                    for (int v = 0; v < V; v++) acc[p][v] = _mm512_fmadd_pd(acc[p][v], xv[v], c);
                }
            }

#pragma GCC unroll 16
            for (int p = 0; p < P; p++) {
#pragma GCC unroll 16
                for (int v = 0; v < V; v++) _mm512_storeu_pd(y + (p0 + p) * ldy + n0 + 8 * v, acc[p][v]);
            }
        }
    }
}

// =============================================================================================
// Exercise 5: any number of polynomials and points.
// =============================================================================================

// The result is cut into four rectangles, each handled by the fastest code that fits it:
//
//                 0 ........ nMain ...... nVec ... numPoints
//        0       +---------------+-----------+---+
//                |  A: whole     |  B:       | D |
//                |  P x V blocks |  1 x 1    | s |
//      pMain     +---------------+  blocks   | c |
//                |  C: 1 x V     |           | a |
//   numPolys     +---------------+-----------+-l-+
//
// polyBlock works on any rectangle whose sizes are multiples of its block: point its pointers at
// the rectangle's top-left corner and keep ldy, the distance between rows of y.
[[gnu::noinline]] void polyAny(const double* coef, int degree, int numPolys, const double* x,
                               int numPoints, double* y, int ldy) {
    constexpr int P = 4, V = 4;  // a fast shape from exercise 4
    const int stride = degree + 1;
    const int pMain = numPolys - numPolys % P;          // rows in whole blocks
    const int nMain = numPoints - numPoints % (8 * V);  // points in whole blocks
    const int nVec = numPoints - numPoints % 8;         // points in whole vectors

    // A: the main rectangle.
    polyBlock<P, V>(coef, degree, pMain, x, nMain, y, ldy);
    // B: every polynomial, the whole vectors after nMain (fewer than V of them).
    polyBlock<1, 1>(coef, degree, numPolys, x + nMain, nVec - nMain, y + nMain, ldy);
    // C: the leftover polynomials (fewer than P), over the main points.
    polyBlock<1, V>(coef + pMain * stride, degree, numPolys - pMain, x, nMain, y + pMain * ldy, ldy);
    // D: the last numPoints % 8 points of every polynomial, one point at a time.
    for (int p = 0; p < numPolys; p++) {
        const double* c = coef + p * stride;
        for (int n = nVec; n < numPoints; n++) {
            double acc = c[degree];
            for (int k = degree - 1; k >= 0; k--) acc = acc * x[n] + c[k];
            y[p * ldy + n] = acc;
        }
    }
}

#include "harness.hpp"  // checks and times the functions above; defines main()
