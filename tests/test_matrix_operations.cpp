#include <gtest/gtest.h>
#include "linalg/Matrix.h"
#include "linalg/Operations.h"

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

    EXPECT_THROW(linalg::mat_add(A, B), std::invalid_argument);
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

    EXPECT_THROW(linalg::mat_subtract(A, B), std::invalid_argument);
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

    EXPECT_THROW(linalg::mat_multiply(A, B), std::invalid_argument);
}

