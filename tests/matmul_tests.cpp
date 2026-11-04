#include "matmul.hpp"
#include "threadpool.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

// Tolerance for comparing results of the same product computed with different
// summation orders. randomMatrix draws from [-1000, 1000], so products are up to 1e6
// and sums reach ~1e7, where one double ulp is ~1e-9. 1e-6 allows a few hundred ulps
// of reordering error while still catching any real indexing bug.
constexpr double kTol = 1e-6;

// Independent reference so the blocked/parallel tests do not depend on multiplyNaive
// being correct. Deliberately the dumbest possible implementation.
Matrix referenceMultiply(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (std::size_t i = 0; i < A.rows; i++)
        for (std::size_t j = 0; j < B.cols; j++) {
            double sum = 0.0;
            for (std::size_t k = 0; k < A.cols; k++)
                sum += A.at(i, k) * B.at(k, j);
            C.at(i, j) = sum;
        }
    return C;
}

// Builds a rows x cols matrix from a row-major initializer list.
Matrix fromList(std::size_t rows, std::size_t cols, std::vector<double> values) {
    Matrix m(rows, cols);
    m.data.assign(values.begin(), values.end());
    return m;
}

// rows x cols matrix with every element set to value.
Matrix filled(std::size_t rows, std::size_t cols, double value) {
    Matrix m(rows, cols);
    for (double& x : m.data) x = value;
    return m;
}

// n x n identity.
Matrix identity(std::size_t n) {
    Matrix m(n, n);
    for (std::size_t i = 0; i < n; i++) m.at(i, i) = 1.0;
    return m;
}

// Elementwise comparison that reports the offending (r, c) on failure, rather than
// going through matricesEqual, which is itself under test.
void expectMatricesNear(const Matrix& actual, const Matrix& expected, double tol = kTol) {
    ASSERT_EQ(actual.rows, expected.rows);
    ASSERT_EQ(actual.cols, expected.cols);
    ASSERT_EQ(actual.data.size(), expected.data.size());
    for (std::size_t r = 0; r < actual.rows; r++)
        for (std::size_t c = 0; c < actual.cols; c++)
            EXPECT_NEAR(actual.at(r, c), expected.at(r, c), tol)
                << "mismatch at (" << r << ", " << c << ")";
}

}  // namespace

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

TEST(MatrixTest, ConstructorZeroInitializes) {
    Matrix m(3, 4);
    EXPECT_EQ(m.rows, 3u);
    EXPECT_EQ(m.cols, 4u);
    ASSERT_EQ(m.data.size(), 12u);
    for (double x : m.data) EXPECT_EQ(x, 0.0);
}

TEST(MatrixTest, ConstructorZeroSized) {
    Matrix m(0, 0);
    EXPECT_EQ(m.rows, 0u);
    EXPECT_EQ(m.cols, 0u);
    EXPECT_TRUE(m.data.empty());
}

TEST(MatrixTest, AtIsRowMajor) {
    Matrix m(2, 3);
    m.at(0, 0) = 1; m.at(0, 1) = 2; m.at(0, 2) = 3;
    m.at(1, 0) = 4; m.at(1, 1) = 5; m.at(1, 2) = 6;
    EXPECT_EQ(m.data, (decltype(m.data){1, 2, 3, 4, 5, 6}));
}

// The buffer must start on a 64-byte cache-line boundary so full-width vector loads of an
// aligned row never straddle two lines. Covers sizes that take both of glibc's allocation
// paths: small blocks from the heap and large blocks (2000 x 2000 = 32 MB) from mmap.
TEST(MatrixTest, DataIsCacheLineAligned) {
    for (std::size_t n : {1u, 7u, 100u, 2000u}) {
        Matrix m(n, n);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(m.data.data()) % 64, 0u) << "n = " << n;
    }
}

TEST(MatrixTest, ConstAtReadsSameElement) {
    Matrix m(2, 2);
    m.at(1, 0) = 7.5;
    const Matrix& cm = m;
    EXPECT_EQ(cm.at(1, 0), 7.5);
    EXPECT_EQ(cm.at(0, 1), 0.0);
}

// ---------------------------------------------------------------------------
// matricesEqual
// ---------------------------------------------------------------------------

TEST(MatricesEqualTest, IdenticalMatrices) {
    Matrix a = fromList(2, 2, {1, 2, 3, 4});
    Matrix b = fromList(2, 2, {1, 2, 3, 4});
    EXPECT_TRUE(matricesEqual(a, b, 0.0));
}

TEST(MatricesEqualTest, EmptyMatrices) {
    Matrix a(0, 0), b(0, 0);
    EXPECT_TRUE(matricesEqual(a, b, 0.0));
}

TEST(MatricesEqualTest, RowMismatchIsFalse) {
    Matrix a(2, 3), b(3, 3);
    EXPECT_FALSE(matricesEqual(a, b, 1e9));
}

TEST(MatricesEqualTest, ColMismatchIsFalse) {
    Matrix a(3, 2), b(3, 3);
    EXPECT_FALSE(matricesEqual(a, b, 1e9));
}

TEST(MatricesEqualTest, TransposedShapeIsFalse) {
    // Same element count, different shape: must not compare data alone.
    Matrix a(2, 3), b(3, 2);
    EXPECT_FALSE(matricesEqual(a, b, 1e9));
}

TEST(MatricesEqualTest, WithinToleranceIsTrue) {
    Matrix a = fromList(1, 2, {1.0, 2.0});
    Matrix b = fromList(1, 2, {1.0 + 5e-7, 2.0 - 5e-7});
    EXPECT_TRUE(matricesEqual(a, b, 1e-6));
}

TEST(MatricesEqualTest, DifferenceEqualToToleranceIsTrue) {
    Matrix a = fromList(1, 1, {1.0});
    Matrix b = fromList(1, 1, {1.5});
    EXPECT_TRUE(matricesEqual(a, b, 0.5));
}

TEST(MatricesEqualTest, OutsideToleranceIsFalse) {
    Matrix a = fromList(1, 2, {1.0, 2.0});
    Matrix b = fromList(1, 2, {1.0, 2.0 + 1e-3});
    EXPECT_FALSE(matricesEqual(a, b, 1e-6));
}

TEST(MatricesEqualTest, NegativeDifferenceIsSymmetric) {
    Matrix a = fromList(1, 1, {5.0});
    Matrix b = fromList(1, 1, {3.0});
    EXPECT_FALSE(matricesEqual(a, b, 1.0));
    EXPECT_FALSE(matricesEqual(b, a, 1.0));
    EXPECT_TRUE(matricesEqual(a, b, 2.0));
    EXPECT_TRUE(matricesEqual(b, a, 2.0));
}

TEST(MatricesEqualTest, SingleBadElementIsFalse) {
    Matrix a = filled(4, 4, 1.0);
    Matrix b = filled(4, 4, 1.0);
    b.at(3, 3) = 2.0;   // last element, so an off-by-one loop bound would miss it
    EXPECT_FALSE(matricesEqual(a, b, 1e-6));
}

// ---------------------------------------------------------------------------
// randomMatrix
// ---------------------------------------------------------------------------

TEST(RandomMatrixTest, HasRequestedDimensions) {
    Matrix m = randomMatrix(5, 7, 1);
    EXPECT_EQ(m.rows, 5u);
    EXPECT_EQ(m.cols, 7u);
    EXPECT_EQ(m.data.size(), 35u);
}

TEST(RandomMatrixTest, IsNotAllZeros) {
    Matrix m = randomMatrix(8, 8, 123);
    bool anyNonZero = false;
    for (double x : m.data) anyNonZero |= (x != 0.0);
    EXPECT_TRUE(anyNonZero);
}

TEST(RandomMatrixTest, ValuesAreFinite) {
    Matrix m = randomMatrix(8, 8, 99);
    for (double x : m.data) EXPECT_TRUE(std::isfinite(x));
}

TEST(RandomMatrixTest, SameSeedIsReproducible) {
    Matrix a = randomMatrix(6, 6, 42);
    Matrix b = randomMatrix(6, 6, 42);
    EXPECT_EQ(a.data, b.data);
}

TEST(RandomMatrixTest, DifferentSeedsDiffer) {
    Matrix a = randomMatrix(6, 6, 1);
    Matrix b = randomMatrix(6, 6, 2);
    EXPECT_NE(a.data, b.data);
}

TEST(RandomMatrixTest, ElementsAreNotAllIdentical) {
    Matrix m = randomMatrix(4, 4, 7);
    bool anyDiffer = false;
    for (double x : m.data) anyDiffer |= (x != m.data[0]);
    EXPECT_TRUE(anyDiffer);
}

// ---------------------------------------------------------------------------
// multiplyNaive
// ---------------------------------------------------------------------------

TEST(MultiplyNaiveTest, OutputHasMxNShape) {
    Matrix A(3, 5), B(5, 2);
    Matrix C = multiplyNaive(A, B);
    EXPECT_EQ(C.rows, 3u);
    EXPECT_EQ(C.cols, 2u);
    EXPECT_EQ(C.data.size(), 6u);
}

TEST(MultiplyNaiveTest, OneByOne) {
    Matrix A = fromList(1, 1, {3.0});
    Matrix B = fromList(1, 1, {-4.0});
    Matrix C = multiplyNaive(A, B);
    ASSERT_EQ(C.data.size(), 1u);
    EXPECT_DOUBLE_EQ(C.at(0, 0), -12.0);
}

TEST(MultiplyNaiveTest, KnownTwoByTwo) {
    Matrix A = fromList(2, 2, {1, 2,
                               3, 4});
    Matrix B = fromList(2, 2, {5, 6,
                               7, 8});
    Matrix expected = fromList(2, 2, {19, 22,
                                      43, 50});
    expectMatricesNear(multiplyNaive(A, B), expected, 0.0);
}

TEST(MultiplyNaiveTest, NonSquareOperands) {
    // (2x3) * (3x2) = 2x2
    Matrix A = fromList(2, 3, {1, 2, 3,
                               4, 5, 6});
    Matrix B = fromList(3, 2, {7,  8,
                               9,  10,
                               11, 12});
    Matrix expected = fromList(2, 2, {58,  64,
                                      139, 154});
    expectMatricesNear(multiplyNaive(A, B), expected, 0.0);
}

TEST(MultiplyNaiveTest, OuterProductShape) {
    // (3x1) * (1x2) = 3x2, K = 1
    Matrix A = fromList(3, 1, {1, 2, 3});
    Matrix B = fromList(1, 2, {10, 20});
    Matrix expected = fromList(3, 2, {10, 20,
                                      20, 40,
                                      30, 60});
    expectMatricesNear(multiplyNaive(A, B), expected, 0.0);
}

TEST(MultiplyNaiveTest, InnerProductShape) {
    // (1x3) * (3x1) = 1x1
    Matrix A = fromList(1, 3, {1, 2, 3});
    Matrix B = fromList(3, 1, {4, 5, 6});
    Matrix C = multiplyNaive(A, B);
    ASSERT_EQ(C.rows, 1u);
    ASSERT_EQ(C.cols, 1u);
    EXPECT_DOUBLE_EQ(C.at(0, 0), 32.0);
}

TEST(MultiplyNaiveTest, IdentityLeavesMatrixUnchanged) {
    Matrix A = randomMatrix(5, 5, 11);
    expectMatricesNear(multiplyNaive(identity(5), A), A, 0.0);
    expectMatricesNear(multiplyNaive(A, identity(5)), A, 0.0);
}

TEST(MultiplyNaiveTest, ZeroMatrixGivesZero) {
    Matrix A = randomMatrix(4, 6, 3);
    Matrix Z(6, 3);
    Matrix C = multiplyNaive(A, Z);
    for (double x : C.data) EXPECT_EQ(x, 0.0);
}

TEST(MultiplyNaiveTest, OnesGiveInnerDimension) {
    // Every element of ones(M,K) * ones(K,N) equals K.
    Matrix C = multiplyNaive(filled(3, 7, 1.0), filled(7, 4, 1.0));
    for (double x : C.data) EXPECT_DOUBLE_EQ(x, 7.0);
}

TEST(MultiplyNaiveTest, IsNotCommutativeInGeneral) {
    Matrix A = fromList(2, 2, {1, 2, 3, 4});
    Matrix B = fromList(2, 2, {0, 1, 1, 0});
    Matrix AB = multiplyNaive(A, B);
    Matrix BA = multiplyNaive(B, A);
    EXPECT_NE(AB.data, BA.data);
}

TEST(MultiplyNaiveTest, EmptyInnerDimensionGivesZeros) {
    // (3x0) * (0x2) is a well-defined 3x2 zero matrix.
    Matrix A(3, 0), B(0, 2);
    Matrix C = multiplyNaive(A, B);
    EXPECT_EQ(C.rows, 3u);
    EXPECT_EQ(C.cols, 2u);
    for (double x : C.data) EXPECT_EQ(x, 0.0);
}

TEST(MultiplyNaiveTest, EmptyOuterDimensionsGiveEmptyResult) {
    Matrix A(0, 4), B(4, 0);
    Matrix C = multiplyNaive(A, B);
    EXPECT_EQ(C.rows, 0u);
    EXPECT_EQ(C.cols, 0u);
    EXPECT_TRUE(C.data.empty());
}

TEST(MultiplyNaiveTest, MatchesReferenceOnRandomInput) {
    Matrix A = randomMatrix(9, 13, 100);
    Matrix B = randomMatrix(13, 6, 200);
    expectMatricesNear(multiplyNaive(A, B), referenceMultiply(A, B));
}

TEST(MultiplyNaiveTest, DoesNotModifyInputs) {
    Matrix A = randomMatrix(4, 4, 5);
    Matrix B = randomMatrix(4, 4, 6);
    auto aCopy = A.data, bCopy = B.data;
    multiplyNaive(A, B);
    EXPECT_EQ(A.data, aCopy);
    EXPECT_EQ(B.data, bCopy);
}

TEST(MultiplyNaiveTest, InnerDimensionMismatchThrows) {
    Matrix A(2, 3), B(4, 2);
    EXPECT_THROW(multiplyNaive(A, B), std::invalid_argument);
}

TEST(MultiplyNaiveTest, SwappedOperandsOfNonSquareProductThrow) {
    // A * B is valid but B * A is not.
    Matrix A(2, 3), B(3, 5);
    EXPECT_NO_THROW(multiplyNaive(A, B));
    EXPECT_THROW(multiplyNaive(B, A), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// multiplyBlocked
// ---------------------------------------------------------------------------

TEST(MultiplyBlockedTest, OutputHasMxNShape) {
    Matrix A(3, 5), B(5, 2);
    Matrix C = multiplyBlocked(A, B, 2);
    EXPECT_EQ(C.rows, 3u);
    EXPECT_EQ(C.cols, 2u);
}

TEST(MultiplyBlockedTest, KnownTwoByTwoBlockSizeOne) {
    Matrix A = fromList(2, 2, {1, 2, 3, 4});
    Matrix B = fromList(2, 2, {5, 6, 7, 8});
    Matrix expected = fromList(2, 2, {19, 22, 43, 50});
    expectMatricesNear(multiplyBlocked(A, B, 1), expected, 0.0);
}

TEST(MultiplyBlockedTest, KnownTwoByTwoBlockSizeEqualToDimension) {
    Matrix A = fromList(2, 2, {1, 2, 3, 4});
    Matrix B = fromList(2, 2, {5, 6, 7, 8});
    Matrix expected = fromList(2, 2, {19, 22, 43, 50});
    expectMatricesNear(multiplyBlocked(A, B, 2), expected, 0.0);
}

TEST(MultiplyBlockedTest, KnownTwoByTwoBlockSizeLargerThanDimension) {
    Matrix A = fromList(2, 2, {1, 2, 3, 4});
    Matrix B = fromList(2, 2, {5, 6, 7, 8});
    Matrix expected = fromList(2, 2, {19, 22, 43, 50});
    expectMatricesNear(multiplyBlocked(A, B, 8), expected, 0.0);
}

TEST(MultiplyBlockedTest, KnownThreeByThreeBlockSizeTwo) {
    // 3 is not a multiple of 2, so every dimension has a 1-wide tail tile.
    Matrix A = fromList(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Matrix B = fromList(3, 3, {9, 8, 7, 6, 5, 4, 3, 2, 1});
    Matrix expected = fromList(3, 3, {30, 24, 18, 84, 69, 54, 138, 114, 90});
    expectMatricesNear(multiplyBlocked(A, B, 2), expected, 0.0);
}

TEST(MultiplyBlockedTest, DimensionsAreMultiplesOfBlockSize) {
    Matrix A = randomMatrix(8, 12, 1);
    Matrix B = randomMatrix(12, 16, 2);
    expectMatricesNear(multiplyBlocked(A, B, 4), referenceMultiply(A, B));
}

TEST(MultiplyBlockedTest, DimensionsAreNotMultiplesOfBlockSize) {
    // 7, 10, 9 with blockSize 4 exercises tail tiles in M, K and N.
    Matrix A = randomMatrix(7, 10, 3);
    Matrix B = randomMatrix(10, 9, 4);
    expectMatricesNear(multiplyBlocked(A, B, 4), referenceMultiply(A, B));
}

TEST(MultiplyBlockedTest, TailInRowsOnly) {
    Matrix A = randomMatrix(9, 8, 5);
    Matrix B = randomMatrix(8, 8, 6);
    expectMatricesNear(multiplyBlocked(A, B, 4), referenceMultiply(A, B));
}

TEST(MultiplyBlockedTest, TailInColsOnly) {
    Matrix A = randomMatrix(8, 8, 7);
    Matrix B = randomMatrix(8, 11, 8);
    expectMatricesNear(multiplyBlocked(A, B, 4), referenceMultiply(A, B));
}

TEST(MultiplyBlockedTest, TailInInnerDimensionOnly) {
    Matrix A = randomMatrix(8, 13, 9);
    Matrix B = randomMatrix(13, 8, 10);
    expectMatricesNear(multiplyBlocked(A, B, 4), referenceMultiply(A, B));
}

TEST(MultiplyBlockedTest, BlockSizeOne) {
    Matrix A = randomMatrix(5, 6, 11);
    Matrix B = randomMatrix(6, 4, 12);
    expectMatricesNear(multiplyBlocked(A, B, 1), referenceMultiply(A, B));
}

TEST(MultiplyBlockedTest, BlockSizeLargerThanEveryDimension) {
    Matrix A = randomMatrix(5, 6, 13);
    Matrix B = randomMatrix(6, 4, 14);
    expectMatricesNear(multiplyBlocked(A, B, 64), referenceMultiply(A, B));
}

TEST(MultiplyBlockedTest, BlockSizeEqualToDimension) {
    Matrix A = randomMatrix(6, 6, 15);
    Matrix B = randomMatrix(6, 6, 16);
    expectMatricesNear(multiplyBlocked(A, B, 6), referenceMultiply(A, B));
}

TEST(MultiplyBlockedTest, MatchesNaiveOnRandomInput) {
    Matrix A = randomMatrix(17, 23, 17);
    Matrix B = randomMatrix(23, 19, 18);
    expectMatricesNear(multiplyBlocked(A, B, 5), multiplyNaive(A, B));
}

TEST(MultiplyBlockedTest, EmptyInnerDimensionGivesZeros) {
    Matrix A(3, 0), B(0, 2);
    Matrix C = multiplyBlocked(A, B, 4);
    EXPECT_EQ(C.rows, 3u);
    EXPECT_EQ(C.cols, 2u);
    for (double x : C.data) EXPECT_EQ(x, 0.0);
}

TEST(MultiplyBlockedTest, EmptyResult) {
    Matrix A(0, 4), B(4, 0);
    Matrix C = multiplyBlocked(A, B, 2);
    EXPECT_EQ(C.rows, 0u);
    EXPECT_EQ(C.cols, 0u);
    EXPECT_TRUE(C.data.empty());
}

TEST(MultiplyBlockedTest, DoesNotModifyInputs) {
    Matrix A = randomMatrix(6, 7, 19);
    Matrix B = randomMatrix(7, 5, 20);
    auto aCopy = A.data, bCopy = B.data;
    multiplyBlocked(A, B, 3);
    EXPECT_EQ(A.data, aCopy);
    EXPECT_EQ(B.data, bCopy);
}

TEST(MultiplyBlockedTest, InnerDimensionMismatchThrows) {
    Matrix A(2, 3), B(4, 2);
    EXPECT_THROW(multiplyBlocked(A, B, 2), std::invalid_argument);
}

TEST(MultiplyBlockedTest, ZeroBlockSizeThrows) {
    Matrix A(4, 4), B(4, 4);
    EXPECT_THROW(multiplyBlocked(A, B, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// multiplyBlock
// ---------------------------------------------------------------------------

TEST(MultiplyBlockTest, FullRangeMatchesReference) {
    Matrix A = randomMatrix(6, 7, 21);
    Matrix B = randomMatrix(7, 5, 22);
    Matrix C(6, 5);
    multiplyBlock(A, B, C, 0, 6, 0, 5, 3);
    expectMatricesNear(C, referenceMultiply(A, B));
}

TEST(MultiplyBlockTest, WritesOnlyNamedTile) {
    Matrix A = randomMatrix(8, 8, 23);
    Matrix B = randomMatrix(8, 8, 24);
    const double sentinel = -12345.0;
    Matrix C = filled(8, 8, sentinel);

    multiplyBlock(A, B, C, 2, 5, 3, 7, 4);   // rows [2,5), cols [3,7)

    // multiplyBlock accumulates, so the tile holds sentinel + product; everything
    // outside the tile must be untouched.
    Matrix expected = referenceMultiply(A, B);
    for (std::size_t r = 0; r < 8; r++)
        for (std::size_t c = 0; c < 8; c++) {
            bool inTile = (r >= 2 && r < 5) && (c >= 3 && c < 7);
            if (inTile)
                EXPECT_NEAR(C.at(r, c), sentinel + expected.at(r, c), kTol)
                    << "(" << r << ", " << c << ")";
            else
                EXPECT_EQ(C.at(r, c), sentinel) << "clobbered (" << r << ", " << c << ")";
        }
}

TEST(MultiplyBlockTest, AccumulatesIntoExistingTile) {
    // Two calls on the same tile add up to twice the product.
    Matrix A = randomMatrix(6, 7, 35);
    Matrix B = randomMatrix(7, 5, 36);
    Matrix C(6, 5);
    multiplyBlock(A, B, C, 0, 6, 0, 5, 3);
    multiplyBlock(A, B, C, 0, 6, 0, 5, 3);

    Matrix expected = referenceMultiply(A, B);
    for (double& x : expected.data) x *= 2;
    expectMatricesNear(C, expected);
}

TEST(MultiplyBlockTest, TileIsCompleteAfterReturn) {
    // The whole K dimension must be accumulated inside a single call, even when
    // blockSize does not divide K.
    Matrix A = randomMatrix(4, 11, 25);
    Matrix B = randomMatrix(11, 4, 26);
    Matrix C(4, 4);
    multiplyBlock(A, B, C, 0, 4, 0, 4, 3);
    expectMatricesNear(C, referenceMultiply(A, B));
}

TEST(MultiplyBlockTest, SingleElementTile) {
    Matrix A = randomMatrix(5, 6, 27);
    Matrix B = randomMatrix(6, 5, 28);
    Matrix C(5, 5);
    multiplyBlock(A, B, C, 3, 4, 1, 2, 2);
    Matrix expected = referenceMultiply(A, B);
    EXPECT_NEAR(C.at(3, 1), expected.at(3, 1), kTol);
    // Everything else stays zero.
    for (std::size_t r = 0; r < 5; r++)
        for (std::size_t c = 0; c < 5; c++)
            if (!(r == 3 && c == 1)) { EXPECT_EQ(C.at(r, c), 0.0); }
}

TEST(MultiplyBlockTest, DisjointTilesAssembleFullProduct) {
    // Emulates what the parallel version does, but serially: cover C with tiles and
    // check the union equals the product with nothing missing or double-counted.
    Matrix A = randomMatrix(7, 9, 29);
    Matrix B = randomMatrix(9, 10, 30);
    Matrix C(7, 10);
    const std::size_t bs = 3;
    for (std::size_t r0 = 0; r0 < 7; r0 += bs)
        for (std::size_t c0 = 0; c0 < 10; c0 += bs)
            multiplyBlock(A, B, C, r0, std::min(r0 + bs, std::size_t(7)),
                          c0, std::min(c0 + bs, std::size_t(10)), bs);
    expectMatricesNear(C, referenceMultiply(A, B));
}

TEST(MultiplyBlockTest, EmptyTileIsNoOp) {
    Matrix A = randomMatrix(4, 4, 31);
    Matrix B = randomMatrix(4, 4, 32);
    Matrix C = filled(4, 4, 1.0);
    multiplyBlock(A, B, C, 2, 2, 0, 4, 2);   // rowStart == rowEnd
    multiplyBlock(A, B, C, 0, 4, 3, 3, 2);   // colStart == colEnd
    for (double x : C.data) EXPECT_EQ(x, 1.0);
}

TEST(MultiplyBlockTest, BlockSizeLargerThanK) {
    Matrix A = randomMatrix(3, 4, 33);
    Matrix B = randomMatrix(4, 3, 34);
    Matrix C(3, 3);
    multiplyBlock(A, B, C, 0, 3, 0, 3, 100);
    expectMatricesNear(C, referenceMultiply(A, B));
}

// ---------------------------------------------------------------------------
// multiplyBlockedParallel
// ---------------------------------------------------------------------------

TEST(MultiplyBlockedParallelTest, OutputHasMxNShape) {
    Threadpool pool(2);
    Matrix A(3, 5), B(5, 2);
    Matrix C = multiplyBlockedParallel(A, B, 2, pool);
    EXPECT_EQ(C.rows, 3u);
    EXPECT_EQ(C.cols, 2u);
}

TEST(MultiplyBlockedParallelTest, KnownTwoByTwo) {
    Threadpool pool(2);
    Matrix A = fromList(2, 2, {1, 2, 3, 4});
    Matrix B = fromList(2, 2, {5, 6, 7, 8});
    Matrix expected = fromList(2, 2, {19, 22, 43, 50});
    expectMatricesNear(multiplyBlockedParallel(A, B, 1, pool), expected, 0.0);
}

TEST(MultiplyBlockedParallelTest, ResultIsCompleteWhenReturned) {
    // If the function returns before all tiles finish, C will still contain zeros.
    Threadpool pool(4);
    Matrix A = filled(32, 32, 1.0);
    Matrix B = filled(32, 32, 1.0);
    Matrix C = multiplyBlockedParallel(A, B, 4, pool);
    for (double x : C.data) EXPECT_DOUBLE_EQ(x, 32.0);
}

TEST(MultiplyBlockedParallelTest, MatchesReferenceMultiplesOfBlockSize) {
    Threadpool pool(4);
    Matrix A = randomMatrix(16, 24, 41);
    Matrix B = randomMatrix(24, 8, 42);
    expectMatricesNear(multiplyBlockedParallel(A, B, 4, pool), referenceMultiply(A, B));
}

TEST(MultiplyBlockedParallelTest, MatchesReferenceWithTailTiles) {
    Threadpool pool(4);
    Matrix A = randomMatrix(13, 17, 43);
    Matrix B = randomMatrix(17, 11, 44);
    expectMatricesNear(multiplyBlockedParallel(A, B, 4, pool), referenceMultiply(A, B));
}

TEST(MultiplyBlockedParallelTest, MatchesNaiveAndBlocked) {
    Threadpool pool(4);
    Matrix A = randomMatrix(21, 19, 45);
    Matrix B = randomMatrix(19, 23, 46);
    Matrix parallel = multiplyBlockedParallel(A, B, 5, pool);
    expectMatricesNear(parallel, multiplyNaive(A, B));
    expectMatricesNear(parallel, multiplyBlocked(A, B, 5));
}

TEST(MultiplyBlockedParallelTest, SingleThreadPool) {
    Threadpool pool(1);
    Matrix A = randomMatrix(10, 12, 47);
    Matrix B = randomMatrix(12, 9, 48);
    expectMatricesNear(multiplyBlockedParallel(A, B, 3, pool), referenceMultiply(A, B));
}

TEST(MultiplyBlockedParallelTest, MoreThreadsThanTiles) {
    Threadpool pool(16);
    Matrix A = randomMatrix(4, 4, 49);
    Matrix B = randomMatrix(4, 4, 50);
    expectMatricesNear(multiplyBlockedParallel(A, B, 4, pool), referenceMultiply(A, B));   // one tile
}

TEST(MultiplyBlockedParallelTest, ManyMoreTilesThanThreads) {
    Threadpool pool(2);
    Matrix A = randomMatrix(32, 16, 51);
    Matrix B = randomMatrix(16, 32, 52);
    expectMatricesNear(multiplyBlockedParallel(A, B, 1, pool), referenceMultiply(A, B));   // 1024 tiles
}

TEST(MultiplyBlockedParallelTest, BlockSizeLargerThanEveryDimension) {
    Threadpool pool(4);
    Matrix A = randomMatrix(5, 6, 53);
    Matrix B = randomMatrix(6, 4, 54);
    expectMatricesNear(multiplyBlockedParallel(A, B, 64, pool), referenceMultiply(A, B));
}

TEST(MultiplyBlockedParallelTest, EmptyInnerDimensionGivesZeros) {
    Threadpool pool(2);
    Matrix A(3, 0), B(0, 2);
    Matrix C = multiplyBlockedParallel(A, B, 2, pool);
    EXPECT_EQ(C.rows, 3u);
    EXPECT_EQ(C.cols, 2u);
    for (double x : C.data) EXPECT_EQ(x, 0.0);
}

TEST(MultiplyBlockedParallelTest, EmptyResult) {
    Threadpool pool(2);
    Matrix A(0, 4), B(4, 0);
    Matrix C = multiplyBlockedParallel(A, B, 2, pool);
    EXPECT_EQ(C.rows, 0u);
    EXPECT_EQ(C.cols, 0u);
    EXPECT_TRUE(C.data.empty());
}

TEST(MultiplyBlockedParallelTest, PoolIsReusableAcrossCalls) {
    Threadpool pool(4);
    for (int round = 0; round < 5; round++) {
        Matrix A = randomMatrix(12, 10, 60 + round);
        Matrix B = randomMatrix(10, 14, 70 + round);
        expectMatricesNear(multiplyBlockedParallel(A, B, 4, pool), referenceMultiply(A, B));
    }
}

TEST(MultiplyBlockedParallelTest, PoolStillUsableAfterMultiply) {
    // The multiply must not leave the pool in a stopped or wedged state.
    Threadpool pool(4);
    Matrix A = randomMatrix(8, 8, 80);
    Matrix B = randomMatrix(8, 8, 81);
    multiplyBlockedParallel(A, B, 4, pool);

    std::atomic<int> counter{0};
    for (int i = 0; i < 100; i++) pool.enqueue([&counter] { counter++; });
    pool.waitAll();
    EXPECT_EQ(counter.load(), 100);
}

TEST(MultiplyBlockedParallelTest, RepeatedRunsAreDeterministic) {
    // Each tile accumulates in a fixed order regardless of which thread runs it,
    // so the result should be bit-identical run to run.
    Threadpool pool(8);
    Matrix A = randomMatrix(20, 20, 90);
    Matrix B = randomMatrix(20, 20, 91);
    Matrix first = multiplyBlockedParallel(A, B, 3, pool);
    for (int i = 0; i < 10; i++) {
        Matrix again = multiplyBlockedParallel(A, B, 3, pool);
        EXPECT_EQ(again.data, first.data) << "run " << i;
    }
}

TEST(MultiplyBlockedParallelTest, DoesNotModifyInputs) {
    Threadpool pool(4);
    Matrix A = randomMatrix(9, 9, 92);
    Matrix B = randomMatrix(9, 9, 93);
    auto aCopy = A.data, bCopy = B.data;
    multiplyBlockedParallel(A, B, 4, pool);
    EXPECT_EQ(A.data, aCopy);
    EXPECT_EQ(B.data, bCopy);
}

TEST(MultiplyBlockedParallelTest, InnerDimensionMismatchThrows) {
    Threadpool pool(2);
    Matrix A(2, 3), B(4, 2);
    EXPECT_THROW(multiplyBlockedParallel(A, B, 2, pool), std::invalid_argument);
}

TEST(MultiplyBlockedParallelTest, ZeroBlockSizeThrows) {
    Threadpool pool(2);
    Matrix A(4, 4), B(4, 4);
    EXPECT_THROW(multiplyBlockedParallel(A, B, 0, pool), std::invalid_argument);
}

TEST(MultiplyBlockedParallelTest, PoolStillUsableAfterThrow) {
    // Validation must happen on the calling thread before anything is enqueued,
    // so a rejected call leaves the pool in a clean state.
    Threadpool pool(2);
    Matrix bad(2, 3), B(4, 2);
    EXPECT_THROW(multiplyBlockedParallel(bad, B, 2, pool), std::invalid_argument);

    Matrix A = randomMatrix(6, 4, 94);
    expectMatricesNear(multiplyBlockedParallel(A, B, 2, pool), referenceMultiply(A, B));
}
