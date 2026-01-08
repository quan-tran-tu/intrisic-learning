#pragma once

#include "core/tensor.h"

void naive_gemv(const Tensor<float> &, const Tensor<float> &x, Tensor<float> &y);
void unroll_j_gemv(const Tensor<float> &W, const Tensor<float> &x, Tensor<float> &y);
void unroll_i_j_gemv(const Tensor<float> &W, const Tensor<float> &x, Tensor<float> &y);
void gemv(const Tensor<float> &W, const Tensor<float> &x, Tensor<float> &y);