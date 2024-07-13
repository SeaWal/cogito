#include "linalg/Matrix.h"
#include "nn/MeanSquaredError.h"

double compute(const linalg::Matrix &output, const linalg::Matrix &target)
{
    double loss = 0.0;
    for (std::size_t i = 0; i < output.rows(); ++i)
    {
        for (std::size_t j = 0; j < output.cols(); ++j)
        {
            double diff = output(i, j) - target(i, j);
            loss += diff * diff;
        }
    }
    return loss / output.rows();
}

linalg::Matrix gradient(const linalg::Matrix &output, const linalg::Matrix &target)
{
    linalg::Matrix grad(output.rows(), output.cols());
    for (std::size_t i = 0; i < output.rows(); ++i)
    {
        for (std::size_t j = 0; j < output.cols(); ++j)
        {
            grad(i, j) = 2 * (output(i, j) - target(i, j)) / output.rows();
        }
    }
    return grad;
}