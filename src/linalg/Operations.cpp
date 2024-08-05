#include <algorithm>
#include <numeric>
#include <vector>

#include "linalg/Operations.h"
#include "linalg/Matrix.h"

#include "common/Checks.h"

linalg::Matrix linalg::mat_add(const linalg::Matrix &lhs, const linalg::Matrix &rhs)
{
    check_full_dims(lhs, rhs);
    linalg::Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.rows(); i++)
    {
        for (std::size_t j = 0; j < lhs.cols(); j++)
        {
            result(i, j) = lhs(i, j) + rhs(i, j);
        }
    }

    return result;
}

linalg::Matrix linalg::mat_subtract(const linalg::Matrix &lhs, const linalg::Matrix &rhs)
{
    check_full_dims(lhs, rhs);
    linalg::Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.rows(); i++)
    {
        for (std::size_t j = 0; j < lhs.cols(); j++)
        {
            result(i, j) = lhs(i, j) - rhs(i, j);
        }
    }

    return result;
}

// TODO: use Matrix copy constructor instead
linalg::Matrix linalg::mat_flatten(const linalg::Matrix &mat)
{
    linalg::Matrix flattened(1, mat.rows() * mat.cols());
    for (std::size_t i = 0; i < mat.rows(); i++)
    {
        for (std::size_t j = 0; j < mat.cols(); j++)
        {
            flattened(0, i * mat.cols() + j) = mat(i, j);
        }
    }

    return flattened;
}

// TODO: use Matrix copy constructor instead
linalg::Matrix linalg::mat_transpose(const linalg::Matrix &mat)
{
    linalg::Matrix transposed(mat.cols(), mat.rows());
    for (std::size_t i = 0; i < mat.rows(); i++)
    {
        for (std::size_t j = 0; j < mat.cols(); j++)
        {
            transposed(j, i) = mat(i, j);
        }
    }

    return transposed;
}

linalg::Matrix linalg::scalar_multiply(const linalg::Matrix &mat, double scalar)
{
    linalg::Matrix result(mat.rows(), mat.cols());
    for (std::size_t i = 0; i < mat.rows(); i++)
    {
        for (std::size_t j = 0; j < mat.cols(); j++)
        {
            result(i, j) = mat(i, j) * scalar;
        }
    }

    return result;
}

linalg::Matrix linalg::scalar_add(const linalg::Matrix &mat, double scalar)
{
    linalg::Matrix result(mat.rows(), mat.cols());
    for (std::size_t i = 0; i < mat.rows(); i++)
    {
        for (std::size_t j = 0; j < mat.cols(); j++)
        {
            result(i, j) = mat(i, j) + scalar;
        }
    }

    return result;
}

linalg::Matrix linalg::hadamard_product(const linalg::Matrix &lhs, const linalg::Matrix &rhs)
{
    check_full_dims(lhs, rhs);
    linalg::Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.rows(); i++)
    {
        for (std::size_t j = 0; j < lhs.cols(); j++)
        {
            result(i, j) = lhs(i, j) * rhs(i, j);
        }
    }

    return result;
}

double linalg::trace(const linalg::Matrix &mat)
{
    if (mat.rows() != mat.cols())
    {
        throw std::invalid_argument("Can't calculate the trace of non-square Matrix");
    }
    double result = 0.0f;
    for (std::size_t i = 0; i < mat.rows(); i++)
    {
        result += mat(i, i);
    }

    return result;
}

double linalg::dot(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
    if (vec1.size() != vec2.size())
    {
        throw std::invalid_argument("Vectors must be same length");
    }

    return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
}

linalg::Matrix linalg::mat_multiply(const linalg::Matrix &lhs, const linalg::Matrix &rhs)
{
    check_inner_dims(lhs, rhs);
    std::size_t outer_row = lhs.rows();
    std::size_t inner_dim = lhs.cols();
    std::size_t outer_col = rhs.cols();

    linalg::Matrix result(outer_row, outer_col);
    for (std::size_t row = 0; row < outer_row; row++)
    {
        for (std::size_t col = 0; col < outer_col; col++)
        {
            double sum = 0.0;
            for (std::size_t el = 0; el < inner_dim; el++)
            {
                sum += lhs(row, el) * rhs(el, col);
            }
            result(row, col) = sum;
        }
    }

    return result;
}

bool linalg::is_square(const linalg::Matrix &mat)
{
    return mat.rows() == mat.cols();
}

// using Crout-Doolittle https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
std::pair<linalg::Matrix, linalg::Matrix> linalg::lu_decomp(const linalg::Matrix &mat)
{
    if (!linalg::is_square(mat))
    {
        throw std::runtime_error("Matrix must be square for LU decomposition");
    }

    std::size_t n = mat.rows();

    linalg::Matrix L = linalg::Matrix::identity(n, n);
    linalg::Matrix U = linalg::Matrix::zeros(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        // Upper triangular
        for (std::size_t k = i; k < n; k++)
        {
            // could replace this loop with dot-product?
            double sum = 0.0;
            for (std::size_t j = 0; j < i; j++)
            {
                sum += L(i, j) * U(j, k);
            }
            U(i, k) = mat(i, k) - sum;
        }

        // Lower triangular
        for (std::size_t k = i + 1; k < n; k++)
        {
            double sum = 0.0;
            for (std::size_t j = 0; j < i; j++)
            {
                sum += L(k, j) * U(j, i);
            }
            L(k, i) = (mat(k, i) - sum) / U(i, i);
        }
    }

    return std::make_pair(L, U);
}


linalg::Matrix linalg::mat_vec_multiply(const linalg::Matrix &mat, const std::vector<double> &vec)
{
    if (mat.cols() != vec.size()) {
        throw std::invalid_argument("Matrix columns and vector size must match for multiplication.");
    }

    std::size_t rows = mat.rows();
    std::size_t cols = mat.cols();
    std::vector<double> result_data(rows, 0.0);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result_data[i] += mat(i, j) * vec[j];
        }
    }

    return Matrix(rows, 1, result_data);
}

// element-wise max 
linalg::Matrix linalg::max(const linalg::Matrix& mat, const double x)
{
    linalg::Matrix res(mat.rows(), mat.cols());
    for(std::size_t i = 0; i < mat.rows(); i++) {
        for(std::size_t j = 0; j < mat.cols(); j++) {
            res(i, j) = std::max(mat(i, j), x);
        }
    }

    return res;
}