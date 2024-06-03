#pragma once

#include "Matrix.h"
#include <stdexcept>

namespace linalg {
    Matrix mat_add(const linalg::Matrix& lhs, const linalg::Matrix& rhs);
    Matrix mat_subtract(const linalg::Matrix& lhs, const linalg::Matrix& rhs);
    Matrix mat_flatten(const linalg::Matrix& mat, bool as_rowvec = true);
    Matrix mat_transpose(const linalg::Matrix& mat);
    // Matrix mat_multiply(const linalg::Matrix& lhs, const linalg::Matrix& rhs);
    Matrix scalar_multiply(const linalg::Matrix& mat, double scalar);
    Matrix scalar_add(const linalg::Matrix& mat, double scalar);
    Matrix hadamard_product(const linalg::Matrix& lhs, const linalg::Matrix& rhs);

    // double trace();
    // double det();

    // double dot(std::vector<double> vec1, std::vector<double> vec2);

    // Matrix-vector multiplication
    // Matrix inverse/pinverse
    // Eigenvalue/Eigenvectors
    // Covariance
    // SVD
    // PCA


}

