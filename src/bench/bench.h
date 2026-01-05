#pragma once

#include "core/tensor.h"

#include <iostream>
#include <vector>
#include <string>
#include <chrono>


struct BenchmarkResult {
    std::string name;
    std::string size;
    double avg_time_ms;
    double max_time_ms;
    double min_time_ms;
    size_t total_ops;
    size_t total_bytes;
    float mse;
};

struct Workload {
    static std::pair<size_t, size_t> gemm_f32(int M, int N, int K) {
        size_t ops = 2ULL * M * N * K;
        size_t bytes = 4ULL * (M * K + K * N + M * N);
        return {ops, bytes};
    }
};

float calculate_mse(const Tensor2D& ref, const Tensor2D& target);
void print_report(const std::vector<BenchmarkResult>& results);

template <typename TFunc, typename... Args>
BenchmarkResult run_benchmark(
    std::string name,
    int size,
    TFunc kernel_func,
    size_t expected_ops,
    size_t expected_bytes,
    const Tensor2D& ref_output,
    Tensor2D& test_output,
    Args&&... args
) {
    const int WARMUP = 2;
    const int ITERATIONS = 20;
    
    for(int i=0; i<WARMUP; ++i) {
        kernel_func(std::forward<Args>(args)..., test_output);
    }

    double total_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;

    for (int i = 0; i < ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        kernel_func(std::forward<Args>(args)..., test_output);
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    double avg_ms = total_ms / ITERATIONS;
    float mse = calculate_mse(ref_output, test_output);

    std::string s = std::to_string(size) + "^3";

    return {name, s, avg_ms, min_ms, max_ms, expected_ops, expected_bytes, mse};
}