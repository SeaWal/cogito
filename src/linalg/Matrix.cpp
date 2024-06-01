#include "linalg/Matrix.h"
#include <cstddef>
#include <iostream>
#include <stdexcept>

linalg::Matrix::Matrix(std::size_t n_rows, std::size_t n_cols)
    : m_nRows(n_rows), m_nCols(n_cols), m_nElements(n_rows * n_cols), m_MatrixData(n_rows * n_cols)
{
    for(std::size_t i = 0; i < m_nElements; i++) {
        m_MatrixData[i] = 0.0;
    }
}

linalg::Matrix::~Matrix() {}

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
        throw std::invalid_argument("Index is out of bounds.");
    }
    return m_MatrixData[row*m_nCols + col];
}
