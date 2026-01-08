#pragma once

#include "core/tensor.h"

void naive_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C);
void ikj_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C);
void ikj_broadcast_a_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C);
void register_blocking_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C);
void cache_tiling_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C);
void cache_tiling_packed_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C);
void packed_parallel_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C);