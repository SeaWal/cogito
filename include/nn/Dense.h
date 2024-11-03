#pragma once

#include <iostream>
#include <optional>
#include <string>

#include "nn/Layer.h"
#include "linalg/Matrix.h"

using linalg::Matrix;

class Dense : public Layer
{
public:
    Dense(std::size_t n_neurons, std::size_t n_inputs, std::optional<std::string> name);
    ~Dense() = default;
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& output_grad) override;
    void update_parameters(double learning_rate) override;
    void Print() override
    {
        std::cout << "weights = [\n";
        m_Weights.Print();
        std::cout << "]\nbiases = [\n";
        m_Biases.Print();
        std::cout << "\n" << std::endl;
    }

private:
    std::size_t m_InputDim;
    std::size_t m_OutputDim;
    Matrix m_Input;
    Matrix m_Weights;
    Matrix m_Biases;
    Matrix m_WeightsGrad;
    Matrix m_BiasesGrad;
};