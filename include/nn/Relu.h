#pragma once

#include "linalg/Matrix.h"
#include "nn/Layer.h"

using linalg::Matrix;

class ReLU : public Layer
{
public:
    ReLU();
    explicit ReLU(std::optional<std::string> name) : Layer(name) {};
    ~ReLU() override = default;
    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &output_grad) override;
    void update_parameters([[maybe_unused]] const double learning_rate) override {};

private:
    Matrix m_Input;
};