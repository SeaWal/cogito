#pragma once

#include "nn/LossFunction.h"
#include "linalg/Matrix.h"

class MeanSquaredError : public LossFunction
{
public:
    double compute(const linalg::Matrix &output, const linalg::Matrix &target) const override;
    linalg::Matrix gradient(const linalg::Matrix &output, const linalg::Matrix &target) const override;
};