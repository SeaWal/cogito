#include <iostream>

#include "nn/Dense.h"
#include "linalg/Matrix.h"
#include "linalg/Operations.h"

using linalg::Matrix;

namespace {
    Matrix broadcast_biases(const Matrix& biases, std::size_t ns)
    {
        Matrix bcast(ns, biases.cols());
        for(std::size_t i = 0; i < ns; i++) {
            for(std::size_t j = 0; j < biases.cols(); j++) {
                bcast(i, j) = biases(0, j);
            }
        }

        return bcast;
    }
}

Dense::Dense(std::size_t n_neurons, std::size_t n_inputs, std::optional<std::string> name = std::nullopt)
    : Layer(name)
{
    m_InputDim = n_inputs;
    m_OutputDim = n_neurons;
    m_Weights = Matrix::random(n_inputs, n_neurons) - 0.5;
    m_Biases = Matrix::random(1, n_neurons) - 0.5;
}

Matrix Dense::forward(const Matrix &input)
{
    m_Input = input;
    return (input * m_Weights) + broadcast_biases(m_Biases, input.rows());
}

Matrix Dense::backward(const Matrix &output_grad)
{
    Matrix WT = linalg::mat_transpose(m_Weights);
    Matrix IT = linalg::mat_transpose(m_Input);

    m_WeightsGrad = IT * output_grad;
    m_BiasesGrad = linalg::colsum(output_grad);
    return output_grad * WT;
}

void Dense::update_parameters(double learning_rate)
{
    m_Weights = m_Weights - (m_WeightsGrad * learning_rate);
    m_Biases = m_Biases - (m_BiasesGrad * learning_rate);
}
