#include "linalg/Matrix.h"
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <random>

linalg::Matrix::Matrix(std::size_t n_rows, std::size_t n_cols)
    : m_nRows(n_rows), m_nCols(n_cols), m_nElements(n_rows * n_cols), m_MatrixData(n_rows * n_cols)
{
    for(std::size_t i = 0; i < m_nElements; i++) {
        m_MatrixData[i] = 0.0;
    }
}

linalg::Matrix::~Matrix() {}


linalg::Matrix linalg::Matrix::identity(std::size_t n_rows, std::size_t n_cols)
{
    linalg::Matrix mat(n_rows, n_cols);
    // allow non-square identity in major diagonal
    std::size_t min_dim = std::min(n_rows, n_cols);
    for (std::size_t i = 0; i < min_dim; ++i) {
        mat(i, i) = 1.0;
    }
    return mat;
}

linalg::Matrix linalg::Matrix::zeros(std::size_t n_rows, std::size_t n_cols)
{
    return linalg::Matrix(n_rows, n_cols);
}

linalg::Matrix linalg::Matrix::random(std::size_t n_rows, std::size_t n_cols)
{
    linalg::Matrix mat(n_rows, n_cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (std::size_t i = 0; i < n_rows; i++) {
        for(std::size_t j = 0; j < n_cols; j++) {
            mat(i, j) = dis(gen);
        } 
    }
    return mat;
}


void linalg::Matrix::Print()
{
    for(std::size_t i = 0; i < m_nRows; i++) {
        for(std::size_t j = 0; j < m_nCols; j++) {
            std::cout << m_MatrixData[(i * m_nCols) + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "" << std::endl;
}


double& linalg::Matrix::operator()(std::size_t row, std::size_t col)
{
    if(row > m_nRows || col > m_nCols) {
        throw std::out_of_range("Index is out of bounds.");
    }
    return m_MatrixData[row*m_nCols + col];
}

const double& linalg::Matrix::operator()(std::size_t row, std::size_t col) const
{
    if(row > m_nRows || col > m_nCols) {
        throw std::out_of_range("Index is out of bounds.");
    }
    return m_MatrixData[row*m_nCols + col];
}

