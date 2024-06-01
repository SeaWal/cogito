#pragma once

#include "Matrix.h"
#include <stdexcept>

linalg::Matrix mat_add(const linalg::Matrix& lhs, const linalg::Matrix& rhs)
{
    if(lhs.rows() != rhs.rows() && lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    return linalg::Matrix(1, 1);
}
