#pragma once

#include "linalg/Matrix.h"

class ILossFunction
{
public:
    virtual double compute(const linalg::Matrix &output, const linalg::Matrix &target) const = 0;
    virtual linalg::Matrix gradient(const linalg::Matrix &output, const linalg::Matrix &target) const = 0;
};