#include "linalg/Matrix.h"
#include "linalg/Operations.h"
#include <iostream>

int main()
{
    linalg::Matrix mat1 = linalg::Matrix(3, 3);
    mat1(0, 0) = 1;
    mat1(0, 1) = 2;
    mat1(0, 2) = 3;
    mat1(1, 0) = 4;
    mat1(1, 1) = 5;
    mat1(1, 2) = 6;
    mat1(2, 0) = 7;
    mat1(2, 1) = 8;
    mat1(2, 2) = 9;

    linalg::Matrix mat2 = linalg::Matrix(3, 3);
    mat2(0, 0) = 1;
    mat2(0, 1) = 2;
    mat2(0, 2) = 3;
    mat2(1, 0) = 4;
    mat2(1, 1) = 5;
    mat2(1, 2) = 6;
    mat2(2, 0) = 7;
    mat2(2, 1) = 8;
    mat2(2, 2) = 9;

    linalg::Matrix mat3 = linalg::mat_add(mat1, mat2);
    mat3.Print();
}
