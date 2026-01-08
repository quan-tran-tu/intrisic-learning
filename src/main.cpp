#include "bench/bench.h"
#include "core/tensor.h"
#include "lib/gemm.h"
#include "lib/gemv.h"

#include <iostream>
#include <vector>
#include <random>
#include <xmmintrin.h>
#include <pmmintrin.h>

void randomize(Tensor<float> &t)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    int h = t.height();
    int w = t.width();
    for (int y = 0; y < h; ++y)
    {
        float *row_ptr = t.data() + (y * t.stride());

        for (int x = 0; x < w; ++x)
            row_ptr[x] = dist(gen);
    }
}

int main(int argc, char **argv)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    std::vector<BenchmarkResult> results;

    std::cout << "Initializing Data..." << std::endl;

    int dim = 1024;

    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--dim" && i + 1 < argc)
        {
            dim = std::stoi(argv[++i]);
        }
    }

    // GEMM
    int M = dim, N = dim, K = dim;
    Tensor<float> A(M, K), B(K, N), C_ref(M, N), C_out(M, N);
    randomize(A);
    randomize(B);

    std::cout << "Running GEMM Reference..." << std::endl;
    naive_gemm(A, B, C_ref);

    auto gemm_load = Workload::gemm_f32(M, N, K);

    // results.push_back(run_benchmark("GEMM naive", dim, "gemm",
    //                                 naive_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    results.push_back(run_benchmark("GEMM ikj", dim, "gemm",
                                    ikj_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    results.push_back(run_benchmark("GEMM ikj broadcast a", dim, "gemm",
                                    ikj_broadcast_a_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    results.push_back(run_benchmark("GEMM register blocking", dim, "gemm",
                                    register_blocking_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    results.push_back(run_benchmark("GEMM cache tiling", dim, "gemm",
                                    cache_tiling_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    results.push_back(run_benchmark("GEMM cache tiling packed", dim, "gemm",
                                    cache_tiling_packed_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    results.push_back(run_benchmark("GEMM packed parallel", dim, "gemm",
                                    packed_parallel_gemm, gemm_load.first, gemm_load.second, C_ref, C_out, A, B));

    // GEMV
    std::cout << "Running GEMV Reference..." << std::endl;

    Tensor<float> W(M, N);
    Tensor<float> x(N, 1), y_ref(M, 1), y_out(M, 1);

    randomize(W);
    randomize(x);

    naive_gemv(W, x, y_ref);

    auto gemv_load = Workload::gemv_f32(M, N);

    results.push_back(run_benchmark("GEMV naive", dim, "gemv",
                                    naive_gemv, gemv_load.first, gemv_load.second, y_ref, y_out, W, x));

    results.push_back(run_benchmark("GEMV unroll j", dim, "gemv",
                                    unroll_j_gemv, gemv_load.first, gemv_load.second, y_ref, y_out, W, x));

    results.push_back(run_benchmark("GEMV unroll i j", dim, "gemv",
                                    unroll_i_j_gemv, gemv_load.first, gemv_load.second, y_ref, y_out, W, x));

    results.push_back(run_benchmark("GEMV", dim, "gemv",
                                    gemv, gemv_load.first, gemv_load.second, y_ref, y_out, W, x));

    print_report(results);

    return 0;
}