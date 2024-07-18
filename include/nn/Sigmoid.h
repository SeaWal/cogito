#pragma once

#include "linalg/Matrix.h"
#include "nn/Layer.h"

class Sigmoid : public Layer
{
public:
    Sigmoid(std::optional<std::string> name = std::nullopt) : Layer(name) {}
    ~Sigmoid() = default;

    linalg::Matrix forward(const linalg::Matrix &input) override;

    linalg::Matrix backward(const linalg::Matrix &grad_output) override;

    void update_parameters([[maybe_unused]] const double learning_rate) override {};

private:
    linalg::Matrix m_Input;
};
