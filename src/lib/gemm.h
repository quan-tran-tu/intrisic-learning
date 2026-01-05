#pragma once

#include "core/tensor.h"

void naive_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C);
void ikj_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C);
void ikj_broadcast_a_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C);
void register_blocking_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C);