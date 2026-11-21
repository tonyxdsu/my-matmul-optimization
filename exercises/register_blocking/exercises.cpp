// Register-blocking assignment: your code. ASSIGNMENT.md walks through it one exercise at a time.
//
//   make regblock          build this file and run every exercise
//   make regblock EX=2     run exercise 2 only
//
// Each function below is a stub that does nothing, so it fails its check until you write it.
// The reference answers are in solution.cpp: look after your version passes, or when stuck.
//
// [[gnu::noinline]] keeps every function a separate symbol, so you can find it in
//   objdump -d -C --no-show-raw-insn build/regblock | less      (then search, e.g. /polyBlock)

#include <immintrin.h>

#include <cstddef>
#include <iostream>

// =============================================================================================
// Exercise 1: y[i] += a * x[i] for i in [0, n). Any n, including 0 and sizes that are not a
// multiple of 8. Use _mm512_set1_pd, _mm512_loadu_pd, _mm512_fmadd_pd, _mm512_storeu_pd, then
// finish the last n % 8 elements with a plain loop.
// =============================================================================================

[[gnu::noinline]] void axpy(double a, const double* x, double* y, std::size_t n) {
    size_t numIterations = n / 8; 
    
    __m512d aVec = _mm512_set1_pd(a);

    size_t it = 0;
    for (; it < numIterations; it++) { 
        __m512d xVec = _mm512_loadu_pd(x + it*8);
        __m512d yVec = _mm512_loadu_pd(y + it*8);
        __m512d resVec = _mm512_fmadd_pd(aVec, xVec, yVec);

        _mm512_storeu_pd(y + it*8, resVec);
    }

    for (size_t i = it * 8; i < n; i++) {
        y[i] += a * x[i];
    }
}

// Bonus: the same, but the last n % 8 elements with one masked load, FMA and store
// (_mm512_maskz_loadu_pd, _mm512_mask_storeu_pd) instead of a plain loop.
[[gnu::noinline]] void axpyMasked(double a, const double* x, double* y, std::size_t n) {
    size_t numIterations = n / 8; 
    
    __m512d aVec = _mm512_set1_pd(a);

    size_t it = 0;
    for (; it < numIterations; it++) { 
        __m512d xVec = _mm512_loadu_pd(x + it*8);
        __m512d yVec = _mm512_loadu_pd(y + it*8);
        __m512d resVec = _mm512_fmadd_pd(aVec, xVec, yVec);

        _mm512_storeu_pd(y + it*8, resVec);
    }

    size_t leftovers = n % 8;
    if (leftovers != 0) { // without if, mask of 0 still runs the following vector instructions
        __mmask8 mask = (__mmask8) ((1 << leftovers) - 1);
        __m512d xVec = _mm512_maskz_loadu_pd(mask, x + it*8);
        __m512d yVec = _mm512_maskz_loadu_pd(mask, y + it*8);
        __m512d resVec = _mm512_fmadd_pd(aVec, xVec, yVec);

        _mm512_mask_storeu_pd(y + it*8, mask, resVec);
    }
}

// =============================================================================================
// Exercise 2: the sum of x[i] * x[i], and the sum of x[i] * y[i], each over [0, n) with ACC
// separate accumulators (ACC __m512d vectors). n is a multiple of 8 * ACC.
// =============================================================================================

template <int ACC>
[[gnu::noinline]] double sumSquares(const double* x, std::size_t n) {
    __m512d accArr[ACC];
    #pragma GCC unroll 16
    for (int a = 0; a < ACC; a++) {
        accArr[a] = _mm512_setzero_pd();
    }

    for (size_t i = 0; i < n; i += ACC * 8) {
        
        #pragma GCC unroll 16
        for (int a = 0; a < ACC; a++) {
            __m512d xVec = _mm512_loadu_pd(x + i + a*8);
            accArr[a] = _mm512_fmadd_pd(xVec, xVec, accArr[a]);
        }
    }

    #pragma GCC unroll 16
    for (int a = 0; a < ACC - 1; a++) {
        accArr[a+1] = _mm512_add_pd(accArr[a], accArr[a+1]);
    }

    return _mm512_reduce_add_pd(accArr[ACC-1]);
}

template <int ACC>
[[gnu::noinline]] double dot(const double* x, const double* y, std::size_t n) {
    __m512d accArr[ACC];
    #pragma GCC unroll 16
    for (int a = 0; a < ACC; a++) {
        accArr[a] = _mm512_setzero_pd();
    }

    for (size_t i = 0; i < n; i += ACC * 8) {
        
        #pragma GCC unroll 16
        for (int a = 0; a < ACC; a++) {
            __m512d xVec = _mm512_loadu_pd(x + i + a*8);
            __m512d yVec = _mm512_loadu_pd(y + i + a*8);
            accArr[a] = _mm512_fmadd_pd(xVec, yVec, accArr[a]);
        }
    }

    #pragma GCC unroll 16
    for (int a = 0; a < ACC - 1; a++) {
        accArr[a+1] = _mm512_add_pd(accArr[a], accArr[a+1]);
    }

    return _mm512_reduce_add_pd(accArr[ACC-1]);
}

// =============================================================================================
// The polynomial bank, exercises 3 to 5. numPolys polynomials of the same degree, evaluated at
// numPoints points:
//
//   y[p * ldy + n] = coef[p * (degree + 1) + 0]
//                  + coef[p * (degree + 1) + 1] * x[n]
//                  + ...
//                  + coef[p * (degree + 1) + degree] * x[n]^degree
//
// for p in [0, numPolys) and n in [0, numPoints). ldy >= numPoints is the distance between rows
// of y; write nothing in a row past numPoints. Evaluate with Horner's rule, from the top:
//   acc = coef[degree];  then for k = degree - 1 down to 0:  acc = acc * x + coef[k]
// =============================================================================================

// Exercise 3: one polynomial and one vector of 8 points at a time. numPoints % 8 == 0.
[[gnu::noinline]] void polyOne(const double* coef, int degree, int numPolys, const double* x,
                               int numPoints, double* y, int ldy) {
    // TODO(3)
}

// Exercise 4: a block of P polynomials x V vectors of points (8 * V points) at a time, its
// P * V results kept in registers until the block is done.
// numPolys % P == 0 and numPoints % (8 * V) == 0.
template <int P, int V>
[[gnu::noinline]] void polyBlock(const double* coef, int degree, int numPolys, const double* x,
                                 int numPoints, double* y, int ldy) {
    // TODO(4)
}

// Exercise 5: any numPolys >= 1 and numPoints >= 1. Use polyBlock for most of the work.
[[gnu::noinline]] void polyAny(const double* coef, int degree, int numPolys, const double* x,
                               int numPoints, double* y, int ldy) {
    // TODO(5)
}

#include "harness.hpp"  // checks and times the functions above; defines main(). Leave it last.
