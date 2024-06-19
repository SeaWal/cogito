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
