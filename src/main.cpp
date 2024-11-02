#include "linalg/Matrix.h"
#include "linalg/Operations.h"
#include "nn/Dense.h"
#include "nn/Relu.h"
#include "nn/Sigmoid.h"
#include "nn/NeuralNetwork.h"
#include "nn/MeanSquaredError.h"
#include <iostream>

int main()
{
    NeuralNetwork network;
    network.add_layer(std::make_unique<Dense>(2, 2, "HiddenLayer"));
    network.add_layer(std::make_unique<Sigmoid>("Sigmoid_1"));
    network.add_layer(std::make_unique<Dense>(1, 2, "OutputLayer"));
    network.add_layer(std::make_unique<Sigmoid>("Sigmoid_2"));

    linalg::Matrix XOR_inputs({
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    });

    linalg::Matrix XOR_targets({
        {0},
        {1},
        {1},
        {0}
    });

    // Train the Neural Network
    MeanSquaredError mse;
    network.train(XOR_inputs, XOR_targets, mse, 1000, 0.1);

    linalg::Matrix predictions = network.forward(XOR_inputs);

    for (std::size_t i = 0; i < XOR_inputs.rows(); ++i)
    {
        std::cout << "Input: (" << XOR_inputs(i, 0) << ", " << XOR_inputs(i, 1) << ") ";
        std::cout << "Predicted Value: " << predictions(i, 0) << " | Actual Value: " 
            << XOR_targets(i, 0) << "\n";
    }
}
