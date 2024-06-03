#include "linalg/Operations.h"
#include "linalg/Matrix.h"

linalg::Matrix linalg::mat_add(const linalg::Matrix& lhs, const linalg::Matrix& rhs)
{
    if(lhs.rows() != rhs.rows() && lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    linalg::Matrix result(lhs.rows(), lhs.cols());
    for(std::size_t i = 0; i < lhs.rows(); i++) {
        for(std::size_t j = 0; j < lhs.cols(); j++) {
            result(i, j) = lhs(i, j) + rhs(i, j);
        }
    }

    return result;
}


linalg::Matrix linalg::mat_subtract(const linalg::Matrix& lhs, const linalg::Matrix& rhs)
{
    if(lhs.rows() != rhs.rows() && lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    linalg::Matrix result(lhs.rows(), lhs.cols());
    for(std::size_t i = 0; i < lhs.rows(); i++) {
        for(std::size_t j = 0; j < lhs.cols(); j++) {
            result(i, j) = lhs(i, j) - rhs(i, j);
        }
    }

    return result;
}

// TODO: use Matrix copy constructor instead
linalg::Matrix linalg::mat_flatten(const linalg::Matrix& mat, bool as_rowvec)
{
    linalg::Matrix flattened(1, mat.rows() * mat.cols());
    for(std::size_t i = 0; i < mat.rows(); i++) {
        for(std::size_t j = 0; j < mat.cols(); j++) {
            flattened(0, i*mat.cols() + j) = mat(i, j);
        }
    }

    return flattened;
}

// TODO: use Matrix copy constructor instead
linalg::Matrix linalg::mat_transpose(const linalg::Matrix& mat)
{
    linalg::Matrix transposed(mat.cols(), mat.rows());
    for(std::size_t i = 0; i < mat.rows(); i++) {
        for(std::size_t j = 0; j < mat.cols(); j++) {
            transposed(j, i) = mat(i, j);
        }
    }

    return transposed;
}