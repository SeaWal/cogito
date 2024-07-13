#pragma once

#include "linalg/Matrix.h"

class LossFunction
{
public:
    virtual double compute(const linalg::Matrix &output, const linalg::Matrix &target) const = 0;
    virtual linalg::Matrix gradient(const linalg::Matrix &output, const linalg::Matrix &target) const = 0;
};

template <typename T>
concept LossFunctionConcept = requires(T loss_fn, const Matrix &predicted, const Matrix &target) {
    { loss_fn.compute(predicted, target) } -> std::convertible_to<double>;
    { loss_fn.gradient(predicted, target) } -> std::convertible_to<Matrix>;
};