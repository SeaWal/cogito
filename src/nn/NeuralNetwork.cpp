#include <vector>
#include <memory>

#include "nn/Layer.h"
#include "nn/NeuralNetwork.h"
#include "linalg/Matrix.h"

using linalg::Matrix;

void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer)
{
    m_Layers.push_back(std::move(layer));
}

Matrix NeuralNetwork::forward(const Matrix &input)
{
    Matrix output = input;
    for (auto &layer : m_Layers) {
        output = layer->forward(output);
    }

    return output;
}

void NeuralNetwork::backward(const Matrix &loss_grad)
{
    Matrix grad = loss_grad;
    for (auto it = m_Layers.rbegin(); it != m_Layers.rend(); it++) {
        grad = (*it)->backward(grad);
    }
}

void NeuralNetwork::update_parameters(const double learning_rate)
{
    for (auto &layer : m_Layers) {
        layer->update_parameters(learning_rate);
    }
}