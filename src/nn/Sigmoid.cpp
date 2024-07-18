#include <cmath>

#include "linalg/Matrix.h"
#include "nn/Sigmoid.h"

linalg::Matrix Sigmoid::forward(const linalg::Matrix &input)
{
    m_Input = input;
    linalg::Matrix output = input;
    for (std::size_t i = 0; i < input.rows(); ++i) {
        for (std::size_t j = 0; j < input.cols(); ++j) {
            output(i, j) = 1.0 / (1.0 + std::exp(-input(i, j)));
        }
    }
    return output;
}

linalg::Matrix Sigmoid::backward(const linalg::Matrix &grad_output)
{
    linalg::Matrix grad_input = grad_output;
    linalg::Matrix sigmoid_output = forward(m_Input);
    for (std::size_t i = 0; i < m_Input.rows(); ++i) {
        for (std::size_t j = 0; j < m_Input.cols(); ++j) {
            grad_input(i, j) = grad_output(i, j) * sigmoid_output(i, j) * (1.0 - sigmoid_output(i, j));
        }
    }
    return grad_input;
}