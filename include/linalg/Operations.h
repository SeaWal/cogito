#pragma once

#include "Matrix.h"
#include <stdexcept>

namespace linalg
{
    Matrix mat_add(const linalg::Matrix &lhs, const linalg::Matrix &rhs);
    Matrix mat_subtract(const linalg::Matrix &lhs, const linalg::Matrix &rhs);
    Matrix mat_flatten(const linalg::Matrix &mat);
    Matrix mat_transpose(const linalg::Matrix &mat);
    Matrix mat_multiply(const linalg::Matrix &lhs, const linalg::Matrix &rhs);
    Matrix scalar_multiply(const linalg::Matrix &mat, double scalar);
    Matrix scalar_add(const linalg::Matrix &mat, double scalar);
    Matrix hadamard_product(const linalg::Matrix &lhs, const linalg::Matrix &rhs);

    double trace(const linalg::Matrix &mat);
    double dot(const std::vector<double> &vec1, const std::vector<double> &vec2);
    bool is_square(const Matrix &mat);

    std::pair<Matrix, Matrix> lu_decomp(const Matrix &mat);

    Matrix mat_vec_multiply(const Matrix &mat, const std::vector<double> &vec);

    Matrix max(const Matrix& mat, const double x);
    Matrix rowsum(const Matrix& mat);
    Matrix colsum(const Matrix& mat);

    // QR decomp
    // Cholesky decomp
    // double det();
    // rank
    // linear independence
    // solver

    // Matrix-vector multiplication
    // Matrix inverse/pinverse
    // Eigenvalue/Eigenvectors
    // Covariance
    // SVD
    // PCA
    // ref/rref

}
