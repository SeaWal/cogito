#include <gtest/gtest.h>
#include "linalg/Matrix.h"

using linalg::Matrix;

TEST(MatrixEqualityTest, IdenticalMatrices) 
{
    Matrix mat1(3, 3);
    Matrix mat2(3, 3);

    for(std::size_t i = 0; i < 3; i++) {
        for(std::size_t j = 0; j < 3; j++) {
            mat1(i, j) = i + j;
            mat2(i, j) = i + j;
        }
    }

    EXPECT_TRUE(mat1 == mat2);
}

TEST(MatrixEqualityTest, DifferentDimensions) 
{
    Matrix mat1(3, 3);
    Matrix mat2(2, 2);

    EXPECT_FALSE(mat1 == mat2);
}

TEST(MatrixEqualityTest, SameDimensionsUnequalValues) 
{
    Matrix mat1(3, 3);
    Matrix mat2(3, 3);

    for(std::size_t i = 0; i < 3; i++) {
        for(std::size_t j = 0; j < 3; j++) {
            mat1(i, j) = i + j;
            mat2(i, j) = i * j;
        }
    }

    EXPECT_FALSE(mat1 == mat2);
}

TEST(MatrixEqualityTest, FloatingPointPrecision) {
    Matrix mat1(3, 3);
    Matrix mat2(3, 3);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            mat1(i, j) = i + j;
            mat2(i, j) = i + j + 1e-10;
        }
    }

    EXPECT_TRUE(mat1 == mat2);
}