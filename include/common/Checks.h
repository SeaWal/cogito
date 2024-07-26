#pragma once

#include <sstream>

#include "Exception.h"
#include "linalg/Matrix.h"

using linalg::Matrix;

void check_full_dims(const Matrix& A, const Matrix& B)
{
    if(A.rows() != B.rows() && A.cols() != B.cols()) {
        std::ostringstream oss;

        oss << "Matrix dimensions do not match for addition: "
            << "(" << A.rows() << "," << A.cols() << ") and"
            << "(" << B.rows() << "," << B.cols() << ")" << std::endl;

        throw DimensionMismatchException(oss.str());
    }
}

void check_inner_dims(const Matrix& A, const Matrix& B)
{
    if(A.cols() != B.rows()) {
        std::ostringstream oss;

        oss << "Matrix dimensions do not match for multiplication: "
            << "(" << A.rows() << "," << A.cols() << ") and "
            << "(" << B.rows() << "," << B.cols() << ")" << std::endl;

        throw DimensionMismatchException(oss.str());
    }
}