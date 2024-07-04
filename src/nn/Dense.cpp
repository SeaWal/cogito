#include "nn/Dense.h"
#include "linalg/Matrix.h"
#include "linalg/Operations.h"

using linalg::Matrix;

Dense::Dense(std::size_t input_dim, std::size_t output_dim, std::optional<std::string> name = std::nullopt)
    : Layer(name)
{
    m_InputDim = input_dim;
    m_OutputDim = output_dim;
    m_Weights = Matrix::random(input_dim, output_dim);
    m_Biases = Matrix::random(output_dim, 1);
}

Matrix Dense::forward(const Matrix &input)
{
    m_Input = input;
    return (m_Weights * input) + m_Biases;
}

Matrix Dense::backward(const Matrix &output_grad)
{
    Matrix grad_input = linalg::mat_transpose(m_Weights) * output_grad;
    m_WeightsGrad = output_grad * linalg::mat_transpose(m_Input);
    m_BiasesGrad = output_grad; // this is not necessarily correct
    return grad_input;
}

void Dense::update_parameters(double learning_rate)
{
    m_Weights = m_Weights - (m_WeightsGrad * learning_rate);
    m_Biases = m_Biases - (m_Biases * learning_rate);
}
