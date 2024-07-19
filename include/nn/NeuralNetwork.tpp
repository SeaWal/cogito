#include <iostream>

template<LossFunctionConcept LossFunc>
void NeuralNetwork::train(const linalg::Matrix &input, const linalg::Matrix &target,
        LossFunc& loss_fn, const std::size_t epochs, const double learning_rate)
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