#include "matmul.hpp"

#include <algorithm>
#include <random>

Matrix multiplyNaive(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    Matrix C(A.rows, B.cols);

    // A is i rows by k columns,
    // B is k rows by j columns
    // C is i rows by j columns

    for (size_t i = 0; i < A.rows; i++) {
        for (size_t j = 0; j < B.cols; j++) {
            double sum = 0;
            for (size_t k = 0; k < A.cols; k++) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }

    return C;
}

Matrix multiplyBlocked(const Matrix& A, const Matrix& B, std::size_t blockSize) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    if (blockSize == 0) {
        throw std::invalid_argument("Block size must be greater than zero");
    }

    Matrix C(A.rows, B.cols);

    size_t numBlocksHorizontal = C.cols / blockSize;
    if (C.cols % blockSize != 0) {
        numBlocksHorizontal += 1;
    }

    size_t numBlocksVertical = C.rows / blockSize;
    if (C.rows % blockSize != 0) {
        numBlocksVertical += 1;
    }

    for (size_t i = 0; i < numBlocksVertical; i++) {
        for (size_t j = 0; j < numBlocksHorizontal; j++) {
            size_t rowStart = i * blockSize;
            size_t colStart = j * blockSize;
            size_t rowEnd   = std::min(rowStart + blockSize, C.rows);
            size_t colEnd   = std::min(colStart + blockSize, C.cols);

            multiplyBlock(A, B, C, rowStart, rowEnd, colStart, colEnd, blockSize);
        }
    }

    return C;
}

// rowStart, rowEnd, colStart, colEnd are inclusive 
// and denote the position of the output tile
void multiplyBlock(const Matrix& A, const Matrix& B, Matrix& C,
                   std::size_t rowStart, std::size_t rowEnd,
                   std::size_t colStart, std::size_t colEnd,
                   std::size_t blockSize) {

    size_t numBlocksInner = A.cols / blockSize;
    if (A.cols % blockSize != 0) {
        numBlocksInner += 1;
    }

    size_t ACols = A.cols; // for hoisting, so GCC doesn't have to reload A.cols every iteration 
    size_t BCols = B.cols; // for hoisting, so GCC doesn't have to reload B.cols every iteration

    const double* Bptr = B.data.data();

    for (size_t b = 0; b < numBlocksInner; b++) {

        for (size_t i = rowStart; i < rowEnd; i++) {
            const double* Aiptr = A.data.data() + i * ACols; // Aik hoisted

            for (size_t k = 0; k < blockSize; k++) {
                size_t AColBRow = b * blockSize + k;
                if (AColBRow >= A.cols) {
                        break;
                }
                double Aik = *(Aiptr + AColBRow); // Aik hoisted: stores to C can't change this local, so GCC drops one overlap test per k iteration before entering j loop vs. leaving it in the j loop
                const double* Bkptr = Bptr + AColBRow * BCols;
                
                for (size_t j = colStart; j < colEnd; j++) {
                    C.at(i, j) += Aik * *(Bkptr + j);

                    // C.at(i, j) += Aik * B.at(AColBRow, j);
                    // C.at(i, j) += A.at(i, AColBRow) * B.at(AColBRow, j); // no Aik hoist
                }
            }
        }
    }
}

Matrix multiplyBlockedParallel(const Matrix& A, const Matrix& B, std::size_t blockSize, Threadpool& pool) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    if (blockSize == 0) {
        throw std::invalid_argument("Block size must be greater than zero");
    }

    Matrix C(A.rows, B.cols);

    size_t numBlocksHorizontal = C.cols / blockSize;
    if (C.cols % blockSize != 0) {
        numBlocksHorizontal += 1;
    }

    size_t numBlocksVertical = C.rows / blockSize;
    if (C.rows % blockSize != 0) {
        numBlocksVertical += 1;
    }

    for (size_t i = 0; i < numBlocksVertical; i++) {
        for (size_t j = 0; j < numBlocksHorizontal; j++) {
            size_t rowStart = i * blockSize;
            size_t colStart = j * blockSize;
            size_t rowEnd   = std::min(rowStart + blockSize, C.rows);
            size_t colEnd   = std::min(colStart + blockSize, C.cols);
            pool.enqueue([&A, &B, &C, rowStart, rowEnd, colStart, colEnd, blockSize] {
                multiplyBlock(A, B, C, rowStart, rowEnd, colStart, colEnd, blockSize);
            });
        }
    }

    pool.waitAll();
    return C;
}

Matrix randomMatrix(std::size_t rows, std::size_t cols, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1000.0, 1000.0);

    Matrix m(rows, cols);

    for (size_t i = 0; i < rows * cols; i++) {
        m.data[i] = dis(gen);
    }

    return m;
}

bool matricesEqual(const Matrix& a, const Matrix& b, double tolerance) {
    if (a.rows != b.rows || a.cols != b.cols) {
        return false;
    }

    for (size_t i = 0; i < a.data.size(); i++) {
        if (std::fabs(a.data[i] - b.data[i]) > tolerance) {
            return false;
        }
    }

    return true;
}
