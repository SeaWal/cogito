#include <optional>
#include <string>

#include "nn/Layer.h"
#include "linalg/Matrix.h"

using linalg::Matrix;

class Dense : public Layer
{
public:
    Dense(std::size_t input_dim, std::size_t output_dim, std::optional<std::string> name);
    ~Dense() = default;
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& output_grad) override;
    void update_parameters(double learning_rate) override;

private:
    std::size_t m_InputDim;
    std::size_t m_OutputDim;
    Matrix m_Input;
    Matrix m_Weights;
    Matrix m_Biases;
    Matrix m_WeightsGrad;
    Matrix m_BiasesGrad;
};