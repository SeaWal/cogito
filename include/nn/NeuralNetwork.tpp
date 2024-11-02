#include <iostream>

template<LossFunctionConcept LossFunc>
void NeuralNetwork::train(const linalg::Matrix &input, const linalg::Matrix &target,
        LossFunc& loss_fn, const std::size_t epochs, const double learning_rate)
{
    for(std::size_t epoch = 0; epoch < epochs; epoch++) {
        // std::cout << "Forward Pass" << std::endl;
        linalg::Matrix predicted = forward(input);
        // std::cout << "Compute Loss" << std::endl;
        double loss = loss_fn.compute(predicted, target);
        linalg::Matrix loss_grad = loss_fn.gradient(predicted, target);
        // std::cout << "Backward Pass" << std::endl;
        backward(loss_grad);
        // std::cout << "Updating" << std::endl;
        update_parameters(learning_rate);
        
        std::cout << "Epoch " << epoch << " - Loss = " << loss << "\n";
    }
    std::cout << std::endl;
}