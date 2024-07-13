#include <iostream>
#include <vector>
#include <memory>

#include "nn/Layer.h"
#include "nn/LossFunction.h"
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

template<typename LossFunc>
void NeuralNetwork::train(const linalg::Matrix &input, const linalg::Matrix &target,
        LossFunc loss_fn, const std::size_t epochs, const double learning_rate)
        requires LossFunctionConcept<LossFunc>
{
    for(std::size_t epoch = 0; epoch < epochs; epoch++) {
        linalg::Matrix predicted = forward(input);
        double loss = loss_fn.compute(predicted, target);
        linalg::Matrix loss_grad = loss_fn.gradient(predicted, target);
        backward(loss_grad);
        update_parameters(learning_rate);
        
        std::cout << "Epoch " << epoch << " - Loss = " << loss << "\n";
    }
    std::cout << std::endl;
}