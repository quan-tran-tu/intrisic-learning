#include "bench/bench.h"
#include "core/tensor.h"
#include "lib/gemm.h"

#include <iostream>
#include <vector>
#include <random>

void randomize(Tensor2D &t)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    int size = t.rows() * t.cols();
    for (int i = 0; i < size; ++i)
        t.data().get()[i] = dist(gen);
}

int main()
{
    std::vector<BenchmarkResult> results;

    std::cout << "Initializing Data..." << std::endl;

    const int dim = 256;
    int M = dim, N = dim, K = dim;
    Tensor2D A(M, K), B(K, N), C_ref(M, N), C_out(M, N);
    randomize(A);
    randomize(B);

    std::cout << "Running GEMM Reference..." << std::endl;
    naive_gemm(A, B, C_ref);

    auto gemm_load = Workload::gemm_f32(M, N, K);

    results.push_back(run_benchmark("GEMM naive", dim,
                                    naive_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    results.push_back(run_benchmark("GEMM ikj", dim,
                                    ikj_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    results.push_back(run_benchmark("GEMM ikj broadcast a", dim,
                                    ikj_broadcast_a_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    results.push_back(run_benchmark("GEMM register blocking", dim,
                                    register_blocking_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    print_report(results);

    return 0;
}