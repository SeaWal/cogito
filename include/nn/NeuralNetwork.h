#pragma once

#include <memory>
#include <vector>

#include "Layer.h"
#include "linalg/Matrix.h"

class NeuralNetwork
{
public:
    NeuralNetwork() = default;
    void add_layer(std::unique_ptr<Layer> layer);
    linalg::Matrix forward(const linalg::Matrix &input);
    void backward(const linalg::Matrix &loss_grad);
    void update_parameters(const double learning_rate);
    void train(const linalg::Matrix &input, const linalg::Matrix &target,
               const std::size_t epochs, const double learning_rate);

private:
    std::vector<std::unique_ptr<Layer>> m_Layers;
};