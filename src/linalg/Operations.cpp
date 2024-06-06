#include <numeric>
#include <vector>

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
linalg::Matrix linalg::mat_flatten(const linalg::Matrix& mat)
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

linalg::Matrix linalg::scalar_multiply(const linalg::Matrix& mat, double scalar)
{
    linalg::Matrix result(mat.rows(), mat.cols());
    for(std::size_t i = 0; i < mat.rows(); i++) {
        for(std::size_t j = 0; j < mat.cols(); j++) {
            result(i, j) = mat(i, j) * scalar;
        }
    }

    return result;
}

linalg::Matrix linalg::scalar_add(const linalg::Matrix& mat, double scalar)
{
    linalg::Matrix result(mat.rows(), mat.cols());
    for(std::size_t i = 0; i < mat.rows(); i++) {
        for(std::size_t j = 0; j < mat.cols(); j++) {
                        result(i, j) = mat(i, j) + scalar;

        }
    }

    return result;
}

linalg::Matrix linalg::hadamard_product(const linalg::Matrix& lhs, const linalg::Matrix& rhs)
{
    if(lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("For Hadamard product, matrix dimensions must match");
    }

    linalg::Matrix result(lhs.rows(), lhs.cols());
    for(std::size_t i = 0; i < lhs.rows(); i++) {
        for(std::size_t j = 0; j < lhs.cols(); j++) {
            result(i, j) = lhs(i, j) * rhs(i, j);
        }
    }

    return result;
}

double linalg::trace(const linalg::Matrix& mat)
{
    if(mat.rows() != mat.cols()) {
        throw std::invalid_argument("Can't calculate the trace of non-square Matrix");
    }
    double result = 0.0f;
    for(std::size_t i = 0; i < mat.rows(); i++) {
        result += mat(i, i);
    }

    return result;
}


double linalg::dot(const std::vector<double>& vec1, const std::vector<double>& vec2)
{
    if(vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be same length");
    }

    return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
}
