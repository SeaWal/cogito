#pragma once

#include <cstddef>
#include <vector>
namespace linalg {

    class Matrix
    {
    public:
        Matrix(std::size_t n_rows, std::size_t n_cols);
        ~Matrix();

        static Matrix identity(std::size_t n_rows, std::size_t n_cols);
        static Matrix zeros(std::size_t n_rows, std::size_t n_cols);
        static Matrix random(std::size_t n_rows, std::size_t n_cols);

    public:
        std::size_t rows() const { return m_nRows; }
        std::size_t cols() const { return m_nCols; }

        void Print();

        double& operator()(std::size_t row, std::size_t col);
        const double& operator()(std::size_t row, std::size_t col) const;

    private:
        std::size_t m_nRows, m_nCols, m_nElements;
        std::vector<double> m_MatrixData;
    };
}
