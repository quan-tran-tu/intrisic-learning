#pragma once

#include "core/tensor.h"

void naive_gemv(const Tensor2D &, const Tensor2D &x, Tensor2D &y);
void unroll_j_gemv(const Tensor2D &W, const Tensor2D &x, Tensor2D &y);
void unroll_i_j_gemv(const Tensor2D &W, const Tensor2D &x, Tensor2D &y);
void gemv(const Tensor2D &W, const Tensor2D &x, Tensor2D &y);