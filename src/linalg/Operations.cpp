#include "linalg/Operations.h"
#include "linalg/Matrix.h"

linalg::Matrix linalg::mat_add(const linalg::Matrix& lhs, const linalg::Matrix& rhs)
{
    if(lhs.rows() != rhs.rows() && lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    linalg::Matrix result(lhs.rows(), lhs.cols());
    for(std::size_t i = 0; i < lhs.rows(); i++) {
        for(std::size_t j = 0; j < lhs.cols(); j++) {
            result(i, j) = lhs(i, j) + rhs(i, j);
        }
    }

    return result;
}
