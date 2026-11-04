#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "threadpool.hpp"

#include <cstddef>
#include <new>
#include <vector>

/**
 * Allocator that returns memory aligned to Alignment bytes. Used so a matrix's buffer
 * starts on a cache-line boundary, letting full-width vector loads stay within one line.
 *
 * @tparam T Element type.
 * @tparam Alignment Required alignment in bytes; must be a power of two.
 */
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    // Aligned operator new has undefined behavior for an alignment that is not a power of two.
    static_assert(Alignment != 0 && (Alignment & (Alignment - 1)) == 0,
                  "Alignment must be a power of two");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

    using value_type = T;

    // Needed because Alignment is a non-type parameter, so std::allocator_traits
    // cannot work out how to rebind this allocator to another type on its own.
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
    }

    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t{Alignment});
    }
};

/**
 * Allocator equality: true means memory allocated by one allocator can be freed by the other.
 * Containers check this when they exchange buffers (move assignment, swap). Equal allocators let
 * the container take the other buffer's pointer in O(1); unequal ones force it to allocate its
 * own buffer and move the elements one by one.
 *
 * Always true here because AlignedAllocator has no state: every instance uses the same global
 * aligned operator new and delete. T and U may differ because a container can rebind its
 * allocator to another element type, and a rebound copy must compare equal to the original.
 * The alignment A must match, since allocators with different alignments are not interchangeable.
 */
template <typename T, typename U, std::size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return true; }

/**
 * Required alongside operator== by the C++17 allocator requirements.
 * TODO: switch to C++20, which generates this from operator==, then delete it.
 */
template <typename T, typename U, std::size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return false; }

/**
 * Dense, row-major matrix. Element (r, c) lives at data[r * cols + c].
 */
struct Matrix {
    /**
     * Element storage, row-major. Size is rows * cols.
     * The buffer starts on a 64-byte (cache-line) boundary. Every row starts on one only when
     * cols is a multiple of 8 (64 bytes / sizeof(double)); otherwise some rows start mid-line.
     */
    std::vector<double, AlignedAllocator<double, 64>> data;

    /**
     * Number of rows.
     */
    std::size_t rows;

    /**
     * Number of columns.
     */
    std::size_t cols;

    /**
     * Constructs a zero-initialized rows x cols matrix.
     */
    Matrix(std::size_t rows, std::size_t cols)
        : data(rows * cols, 0.0), rows(rows), cols(cols) {}

    /**
     * Accesses element (r, c). No bounds checking.
     */
    double& at(std::size_t r, std::size_t c) { return data[r * cols + c]; }

    /**
     * Accesses element (r, c). No bounds checking.
     */
    const double& at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

/**
 * Naive single-threaded triple-loop multiply. This is the correctness reference
 * and the baseline every other variant is measured against.
 *
 * @param A Left operand, M x K.
 * @param B Right operand, K x N.
 * @throws std::invalid_argument if dimensions do not match.
 * @return The M x N product.
 */
Matrix multiplyNaive(const Matrix& A, const Matrix& B);

/**
 * Cache-blocked single-threaded multiply. Compare against multiplyNaive to see the
 * cache-locality win in isolation, before any threading is involved.
 *
 * @param A Left operand, M x K.
 * @param B Right operand, K x N.
 * @param blockSize Tile edge length. Tune so three blockSize x blockSize tiles fit in L1/L2.
 * @return The M x N product.
 */
Matrix multiplyBlocked(const Matrix& A, const Matrix& B, std::size_t blockSize);

/**
 * Cache-blocked multiply parallelized across a Threadpool. Each enqueued task should own
 * a disjoint output tile of C, so no two tasks ever write the same element and no locking
 * is needed around C.
 *
 * @param A Left operand, M x K.
 * @param B Right operand, K x N.
 * @param blockSize Tile edge length, also determines how many tasks are produced.
 * @param pool The threadpool to distribute tile computations across.
 * @return The M x N product.
 */
Matrix multiplyBlockedParallel(const Matrix& A, const Matrix& B, std::size_t blockSize, Threadpool& pool);

/**
 * Adds the product A * B into one output tile of C: rows [rowStart, rowEnd) by columns
 * [colStart, colEnd), i.e. C_tile += (A * B)_tile. Like BLAS gemm with beta = 1, the
 * tile's existing contents are kept and added to, so the caller must zero the tile first
 * to get the plain product. The full K dimension is accumulated within a single call, so
 * the tile is complete when this returns and the call is self-contained enough to hand
 * straight to Threadpool::enqueue.
 *
 * @param A Left operand, M x K.
 * @param B Right operand, K x N.
 * @param C Output matrix, M x N. Only the named tile is read or written.
 * @param rowStart First row of the tile, inclusive.
 * @param rowEnd One past the last row of the tile.
 * @param colStart First column of the tile, inclusive.
 * @param colEnd One past the last column of the tile.
 * @param blockSize Tile edge length used for blocking the K loop.
 */
void multiplyBlock(const Matrix& A, const Matrix& B, Matrix& C,
                   std::size_t rowStart, std::size_t rowEnd,
                   std::size_t colStart, std::size_t colEnd,
                   std::size_t blockSize);

/**
 * Builds a matrix filled with pseudorandom values for benchmarking.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param seed Seed for the generator, so runs are reproducible.
 * @return The filled matrix.
 */
Matrix randomMatrix(std::size_t rows, std::size_t cols, unsigned int seed);

/**
 * Compares two matrices elementwise within a tolerance. Floating-point multiply reorders
 * differently between the naive and blocked variants, so exact equality will not hold.
 *
 * @param a First matrix.
 * @param b Second matrix.
 * @param tolerance Maximum permitted absolute difference per element.
 * @return True if dimensions match and every element is within tolerance.
 */
bool matricesEqual(const Matrix& a, const Matrix& b, double tolerance);

#endif
