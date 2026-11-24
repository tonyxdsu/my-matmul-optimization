#include "matmul.hpp"

#include <algorithm>
#include <immintrin.h>
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

const size_t regVBlockSize = 6;
const size_t regHBlockSize = 4;

// rowStart, rowEnd, colStart, colEnd are inclusive 
// and denote the position of the output tile
void multiplyBlock(const Matrix& A, const Matrix& B, Matrix& C,
                   std::size_t rowStart, std::size_t rowEnd,
                   std::size_t colStart, std::size_t colEnd,
                   std::size_t blockSize) {

    size_t numBlocksHorizontal = A.cols / blockSize;
    if (A.cols % blockSize != 0) {
        numBlocksHorizontal += 1;
    }

    size_t ACols = A.cols; 
    size_t BCols = B.cols;

    const double* APtr = A.data.data();
    const double* BPtr = B.data.data();
    double* CPtr = C.data.data();

    for (size_t b = 0; b < numBlocksHorizontal; b++) {
        size_t kStart = b * blockSize;
        size_t kEnd = std::min(ACols, kStart + blockSize);

        for (size_t i = rowStart; i < rowEnd; i += regVBlockSize) {
            const double* AiPtr = APtr + i * ACols;
            double* CiPtr = CPtr + i * BCols;
            
            for (size_t j = colStart; j < colEnd; j += 8*regHBlockSize) {
                
                // load regVBlockSize * regHBlockSize block of C accumulators
                __m512d accCArr[regVBlockSize * regHBlockSize];
                #pragma GCC unroll 8
                for (size_t rV = 0; rV < regVBlockSize; rV++) {

                    #pragma GCC unroll 8
                    for (size_t rH = 0; rH < regHBlockSize; rH++) {
                        accCArr[rV*regHBlockSize + rH] = _mm512_loadu_pd(CiPtr + rV*BCols + j + rH*8);
                    }
                }

                for (size_t k = kStart; k < kEnd; k++) {

                    // load regVBlockSize A values at index k
                    __m512d AVec[regVBlockSize];
                    #pragma GCC unroll 8
                    for (size_t rV = 0; rV < regVBlockSize; rV++) {
                        AVec[rV] = _mm512_set1_pd((AiPtr + rV*ACols)[k]);
                    }

                    const double* BkPtr = BPtr + k * BCols;

                    // FMA the block of C accumulators
                    #pragma GCC unroll 8 
                    for (size_t rH = 0; rH < regHBlockSize; rH++) {
                        __m512d BVec = _mm512_loadu_pd(BkPtr + j + rH*8);
                        
                        #pragma GCC unroll 8
                        for (size_t rV = 0; rV < regVBlockSize; rV++) {
                            accCArr[rV*regHBlockSize + rH] = _mm512_fmadd_pd(AVec[rV], BVec, accCArr[rV*regHBlockSize + rH]);
                        }
                    }
                }

                // store C
                #pragma GCC unroll 8
                for (size_t rV = 0; rV < regVBlockSize; rV++) {

                    #pragma GCC unroll 8
                    for (size_t rH = 0; rH < regHBlockSize; rH++) {
                        _mm512_storeu_pd(CiPtr + rV*BCols + j + rH*8, accCArr[rV*regHBlockSize + rH]);
                    }
                }
            }

        } 
    }
}

// // rowStart, rowEnd, colStart, colEnd are inclusive 
// // and denote the position of the output tile
// void multiplyBlock(const Matrix& A, const Matrix& B, Matrix& C,
//                    std::size_t rowStart, std::size_t rowEnd,
//                    std::size_t colStart, std::size_t colEnd,
//                    std::size_t blockSize) {

//     size_t numBlocksInner = A.cols / blockSize;
//     if (A.cols % blockSize != 0) {
//         numBlocksInner += 1;
//     }

//     // size_t numRegBlocksVertical = (rowEnd - rowStart) / regVBlockSize;
//     // if ((rowEnd - rowStart) % regVBlockSize != 0) {
//     //     numRegBlocksVertical += 1;
//     // }

//     // size_t numRegBlocksHorizontal = (colEnd - colStart) / regHBlockSize;
//     // if ((colEnd - colStart) % regHBlockSize != 0) {
//     //     numRegBlocksHorizontal += 1;
//     // }

//     size_t ACols = A.cols; 
//     size_t BCols = B.cols;

//     const double* APtr = A.data.data();
//     const double* BPtr = B.data.data();
//     double* CPtr = C.data.data();

//     for (size_t b = 0; b < numBlocksInner; b++) {
//         size_t kStart = b * blockSize;
//         size_t kEnd = std::min(ACols, kStart + blockSize);

//         for (size_t i = rowStart; i < rowEnd; i++) {
//             const double* AiPtr = APtr + i * ACols;
//             double* CiPtr = CPtr + i * BCols;

//             for (size_t k = kStart; k < kEnd; k++) {
//                 double Aik = AiPtr[k];
//                 const double* BkPtr = BPtr + k * BCols;
                
//                 for (size_t j = colStart; j < colEnd; j++) {
//                     CiPtr[j] += Aik * BkPtr[j];
//                 }
//             }
//         }
//     }
// }

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
