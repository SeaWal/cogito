#include <gtest/gtest.h>
#include "linalg/Matrix.h"
#include "linalg/Operations.h"
#include "common/Exception.h"

TEST(LinAlgArithmeticTest, MatrixAddition)
{
    linalg::Matrix A = linalg::Matrix::identity(3, 3);
    linalg::Matrix B = linalg::Matrix::identity(3, 3);

    linalg::Matrix C = linalg::mat_add(A, B);
    for(std::size_t i = 0; i < 3; i++) {
        for(std::size_t j = 0; j < 3; j++) {
            if(i == j)
                EXPECT_EQ(C(i, j), 2.0);
            else
                EXPECT_EQ(C(i, j), 0.0);
        }
    }
}

TEST(LinAlgArithmeticTest, MatrixAdditionThrowsOnDimensionMismatch)
{
    linalg::Matrix A = linalg::Matrix::random(2, 3);
    linalg::Matrix B = linalg::Matrix::random(4, 6);

    EXPECT_THROW(linalg::mat_add(A, B), DimensionMismatchException);
}

TEST(LinAlgArithmeticTest, MatrixSubtraction)
{
    linalg::Matrix A = linalg::Matrix::identity(3, 3);
    linalg::Matrix B = linalg::Matrix::identity(3, 3);

    linalg::Matrix C = linalg::mat_subtract(A, B);
    for(std::size_t i = 0; i < 3; i++) {
        for(std::size_t j = 0; j < 3; j++) {
                EXPECT_EQ(C(i, j), 0.0);
        }
    }
}

TEST(LinAlgArithmeticTest, MatrixSubtractionThrowsOnDimensionMismatch)
{
    linalg::Matrix A = linalg::Matrix::random(2, 3);
    linalg::Matrix B = linalg::Matrix::random(4, 6);

    EXPECT_THROW(linalg::mat_subtract(A, B), DimensionMismatchException);
}

TEST(LinAlgArithmeticTest, MatrixMultiplication) 
{
    linalg::Matrix A({{2, 3}, {4, 5}});
    linalg::Matrix B({{1, 4, 2}, {3, 6, 4}});

    linalg::Matrix C = linalg::mat_multiply(A, B);

    EXPECT_EQ(C(0, 0), 11);
    EXPECT_EQ(C(0, 1), 26);
    EXPECT_EQ(C(0, 2), 16);
    EXPECT_EQ(C(1, 0), 19);
    EXPECT_EQ(C(1, 1), 46);
    EXPECT_EQ(C(1, 2), 28); 
}

TEST(LinAlgArithmeticTest, MatrixMultiplicationThrowsOnInnerDimMismatch) 
{
    linalg::Matrix A = linalg::Matrix::random(2, 3);
    linalg::Matrix B = linalg::Matrix::random(4, 6);

    EXPECT_THROW(linalg::mat_multiply(A, B), DimensionMismatchException);
}


TEST(LinAlgArithmeticTest, MatrixScalarAddition) 
{
    linalg::Matrix A({{1, 2}, {3, 4}});
    A = linalg::scalar_add(A, 3.0);

    EXPECT_EQ(A(0, 0), 4.0);
    EXPECT_EQ(A(0, 1), 5.0);
    EXPECT_EQ(A(1, 0), 6.0);
    EXPECT_EQ(A(1, 1), 7.0);
}

TEST(LinAlgArithmeticTest, MatrixScalarMultiplication) 
{
    linalg::Matrix A({{1, 2}, {3, 4}});
    A = linalg::scalar_multiply(A, 3.0);

    EXPECT_EQ(A(0, 0), 3.0);
    EXPECT_EQ(A(0, 1), 6.0);
    EXPECT_EQ(A(1, 0), 9.0);
    EXPECT_EQ(A(1, 1), 12.0);
}

TEST(LinAlgArithmeticTest, HadamardProduct) 
{
    linalg::Matrix A({{1, 2}, {3, 4}});
    linalg::Matrix B({{2, 3}, {2, 5}});

    linalg::Matrix C = linalg::hadamard_product(A, B);
    EXPECT_EQ(C(0, 0), 2.0);
    EXPECT_EQ(C(0, 1), 6.0);
    EXPECT_EQ(C(1, 0), 6.0);
    EXPECT_EQ(C(1, 1), 20.0);
}

TEST(LinAlgArithmeticTest, HadamardProductThrowsOnDimensionMismatch) 
{
    linalg::Matrix A({{1, 2}, {3, 4}});
    linalg::Matrix B({{2, 3, 4}, {2, 5, 4}});

    EXPECT_THROW(linalg::hadamard_product(A, B), DimensionMismatchException);
}

TEST(LinAlgArithmeticTest, VectorDotProduct)
{
    std::vector<double> vec1 = {1.0, 2.0, 3.0};
    std::vector<double> vec2 = {3.0, 4.0, 5.0};

    EXPECT_EQ(linalg::dot(vec1, vec2), 26.0);
}

TEST(LinAlgArithmeticTest, VectorDotProductThrowsOnUnequalSize)
{
    std::vector<double> vec1 = {1.0, 2.0, 3.0};
    std::vector<double> vec2 = {3.0, 4.0, 5.0, 7.0};

    EXPECT_THROW(linalg::dot(vec1, vec2), std::invalid_argument);
}

TEST(LinAlgArithmeticTest, MatrixLUDecomposition)
{
    linalg::Matrix mat({ {4.0, 3.0, 2.0}, {2.0, 1.0, 1.0}, {6.0, 7.0, 9.0} });
    auto [L, U] = linalg::lu_decomp(mat);

    linalg::Matrix expected_L({ {1.0, 0.0, 0.0}, {0.5, 1.0, 0.0}, {1.5, -5.0, 1.0} });
    linalg::Matrix expected_U({ {4.0, 3.0, 2.0}, {0.0, -0.5, 0.0}, {0.0, 0.0, 6.0} });

    for(std::size_t i = 0; i < 3; i++) {
        for(std::size_t j = 0; j < 3; j++) {
            EXPECT_EQ(L(i, j), expected_L(i, j));
            EXPECT_EQ(U(i, j), expected_U(i, j));
        }
    }
}

TEST(LinAlgArithmeticTest, MatrixLUDecompThrowsOnNonSquareMatrix)
{
    linalg::Matrix mat({ {4.0, 3.0, 2.0}, {2.0, 1.0, 1.0} });
    EXPECT_THROW(linalg::lu_decomp(mat), std::runtime_error);
}

TEST(LinAlgArithmeticTest, MatrixVectorMultiplication)
{
    linalg::Matrix mat({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} });
    std::vector<double> vec = {1.0, 2.0, 3.0};

    linalg::Matrix result = linalg::mat_vec_multiply(mat, vec);

    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 1);

    EXPECT_DOUBLE_EQ(result(0, 0), 14.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 32.0);
}