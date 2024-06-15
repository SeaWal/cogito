#include <gtest/gtest.h>
#include "linalg/Matrix.h"


TEST(MatrixConstructorTest, DefaultConstructor)
{
    linalg::Matrix mat(3, 3);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 3);

    for(std::size_t i = 0; i < mat.rows(); i++) {
        for(std::size_t j = 0; j < mat.cols(); j++) {
            EXPECT_EQ(mat(i, j), 0.0f);
        }
    }
}


TEST(MatrixConstructorTest, Vector2DConstructor)
{
    std::vector<std::vector<double>> vec2d = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    linalg::Matrix mat(vec2d);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(mat(i, j), vec2d[i][j]);
        }
    }
}

TEST(MatrixConstructorTest, Vector1DConstructor)
{
    std::vector<double> vec1d = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    linalg::Matrix mat(2, 3, vec1d);
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 3);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(mat(i, j), vec1d[i*3 + j]);
        }
    }
}

TEST(MatrixConstructorTest, IdentityConstructor)
{
    linalg::Matrix id = linalg::Matrix::identity(3, 3);   
    EXPECT_EQ(id.rows(), 3);
    EXPECT_EQ(id.cols(), 3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if(i == j)
                EXPECT_EQ(id(i, j), 1.0f);
            else
                EXPECT_EQ(id(i, j), 0.0f);
        }
    }
}

TEST(MatrixConstructorTest, ZerosConstructor)
{
    linalg::Matrix zero = linalg::Matrix::zeros(3, 3);
    EXPECT_EQ(zero.rows(), 3);
    EXPECT_EQ(zero.cols(), 3);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(zero(i, j), 0.0f);
        }
    }
}

// TODO: add seed to Random constructor for repeatability
TEST(MatrixConstructorTest, RandomConstructor)
{
    linalg::Matrix rand = linalg::Matrix::random(3, 3);
    EXPECT_EQ(rand.rows(), 3);
    EXPECT_EQ(rand.cols(), 3);
    bool all_elements_are_zero = true;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (rand(i, j) != 0.0) {
                all_elements_are_zero = false;
                break;
            }
        }
        if (!all_elements_are_zero) break;
    }
    EXPECT_FALSE(all_elements_are_zero); // Random should not all be zeros
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
