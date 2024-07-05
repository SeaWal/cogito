#pragma once

#include "linalg/Matrix.h"
#include "nn/Layer.h"

using linalg::Matrix;

class ReLU : public Layer
{
    ReLU() = default;
    ~ReLU() = default;
    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &output_grad) override;
    void update_parameters(const double learning_rate) override;
};