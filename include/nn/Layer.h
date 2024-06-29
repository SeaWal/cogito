#include <optional>
#include <string>
#include <vector>

#include "linalg/Matrix.h"

class Layer
{
public:
    Layer() = default;
    Layer(std::optional<std::string> name = std::nullopt) : m_Name(name) {}
    virtual ~Layer() = default;

    virtual linalg::Matrix forward(const linalg::Matrix& input) = 0;
    virtual linalg::Matrix backward(const linalg::Matrix& output_grad) = 0;
    virtual void update_parameters(double learning_rate) = 0;

    inline std::optional<std::string> name() const { return m_Name; }

protected:
    std::optional<std::string> m_Name;
};