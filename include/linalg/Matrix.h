#pragma once

#include <cstddef>
#include <vector>
namespace linalg
{

    class Matrix
    {
    public:
        Matrix() : m_nRows(0), m_nCols(0), m_MatrixData() {};
        Matrix(std::size_t n_rows, std::size_t n_cols);
        Matrix(const std::vector<std::vector<double>> &data);
        Matrix(std::size_t n_rows, std::size_t n_cols, const std::vector<double> &data);

        ~Matrix() = default;

        static Matrix identity(std::size_t n_rows, std::size_t n_cols);
        static Matrix zeros(std::size_t n_rows, std::size_t n_cols);
        static Matrix random(std::size_t n_rows, std::size_t n_cols);

    public:
        std::size_t rows() const { return m_nRows; }
        std::size_t cols() const { return m_nCols; }
        std::pair<std::size_t, std::size_t> dims() const { 
            return std::make_pair(m_nRows, m_nCols); 
        }

        std::vector<double> get_row(std::size_t row);
        std::vector<double> get_col(std::size_t col);

        void Print();
        bool isSquare();

        double &operator()(std::size_t row, std::size_t col);
        const double &operator()(std::size_t row, std::size_t col) const;

        Matrix operator+(const Matrix &other) const;
        Matrix operator+(const double scalar) const;

        Matrix operator-(const Matrix &other) const;
        Matrix operator-(const double scalar) const;

        Matrix operator*(const Matrix &other) const;
        Matrix operator*(const double scalar) const;

        bool operator==(const Matrix& other) const;

    private:
        std::size_t m_nRows, m_nCols, m_nElements;
        std::vector<double> m_MatrixData;
    };
}
