#pragma once

#include <stdexcept>
#include <string>

class DimensionMismatchException : public std::runtime_error
{
public:
    explicit DimensionMismatchException(const std::string& message) : std::runtime_error(message) {}
};