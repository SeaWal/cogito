#include <gtest/gtest.h>
#include "linalg/Matrix.h"
#include "linalg/Operations.h"

using Matrix = linalg::Matrix;

TEST (MatrixUtilTest, MatrixFlatten)
{
    Matrix A({{1, 2, 3}, {4, 5, 6}});
    A = linalg::mat_flatten(A);
    
    EXPECT_EQ(A.rows(), 1);
    EXPECT_EQ(A.cols(), 6);
}

TEST (MatrixUtilTest, MatrixIsSquare)
{
    Matrix A({{1, 2, 3}, {4, 5, 6}});
    Matrix B({{1, 2}, {4, 5}});

    EXPECT_EQ(linalg::is_square(A), false);
    EXPECT_EQ(linalg::is_square(B), true);
}

TEST (MatrixUtilTest, MatrixTrace)
{
    Matrix A({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    double tr = linalg::trace(A);

    EXPECT_EQ(tr, 15);
}

TEST (MatrixUtilTest, MatrixTraceThrowsIfNotSquare)
{
    Matrix A({{1, 2, 3}, {4, 5, 6}});

    EXPECT_THROW(linalg::trace(A), std::invalid_argument);
}