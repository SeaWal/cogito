#include "linalg/Matrix.h"
#include "linalg/Operations.h"
#include "nn/Relu.h"

using linalg::Matrix;

Matrix ReLU::forward(const Matrix &input)
{
    m_Input = input;
    return linalg::max(input, 0.0);
}

Matrix ReLU::backward(const Matrix &output_grad)
{
    linalg::Matrix grad_input = output_grad;
    for (std::size_t i = 0; i < m_Input.rows(); ++i) {
        for (std::size_t j = 0; j < m_Input.cols(); ++j) {
            grad_input(i, j) = (m_Input(i, j) > 0) ? output_grad(i, j) : 0.0;
        }
    }
    return grad_input;
}
