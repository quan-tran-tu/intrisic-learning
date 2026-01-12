#include "bench/bench.h"
#include "core/tensor.h"
#include "lib/gemm.h"
#include "lib/gemv.h"

#include <random>
#include <string>

// GEMM
struct GemmParams
{
    int M, N, K;
};
struct GemmData
{
    Tensor<float> A, B, output;
    GemmData(int m, int n, int k) : A(m, k), B(k, n), output(m, n) {}
};

// GEMV
struct GemvParams
{
    int M, N;
};
struct GemvData
{
    Tensor<float> W, x, output;
    GemvData(int m, int n) : W(m, n), x(1, n), output(1, m) {} // note: x is (N x 1), output is (M x 1)
};

void randomize(Tensor<float> &t)
{
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    int size = t.height() * t.width();
    float *ptr = t.data();
    for (int i = 0; i < size; ++i)
        ptr[i] = dist(gen);
}

int main(int argc, char **argv)
{
    // _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    // _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    // ==========================================
    // GEMM BENCHMARK SUITE
    // ==========================================
    BenchmarkSuite<GemmParams, GemmData> gemm_suite("GEMM F32");

    gemm_suite.complexity_calc = [](const GemmParams &p)
    {
        size_t ops = 2ULL * p.M * p.N * p.K;
        size_t bytes = 4ULL * (p.M * p.K + p.K * p.N + p.M * p.N);
        return std::make_pair(ops, bytes);
    };

    gemm_suite.params_to_string = [](const GemmParams &p)
    {
        return std::to_string(p.M) + "x" + std::to_string(p.N) + "x" + std::to_string(p.K);
    };

    gemm_suite.set_provider([](const GemmParams &p)
                            {
        GemmData d(p.M, p.N, p.K);
        randomize(d.A);
        randomize(d.B);
        return d; });

    gemm_suite.set_reference([](const GemmData &d, Tensor<float> &out)
                             { naive_gemm(d.A, d.B, out); });

    gemm_suite.add_shape({512, 512, 512});
    gemm_suite.add_shape({1024, 1024, 1024});
    gemm_suite.add_shape({2048, 2048, 2048});

    gemm_suite.add_kernel("Packed Parallel", [](const GemmData &d, Tensor<float> &out)
                          { packed_parallel_gemm(d.A, d.B, out); });

    gemm_suite.run();

    // ==========================================
    // GEMV BENCHMARK SUITE
    // ==========================================
    BenchmarkSuite<GemvParams, GemvData> gemv_suite("GEMV F32");

    gemv_suite.complexity_calc = [](const GemvParams &p)
    {
        return std::make_pair(2ULL * p.M * p.N, 4ULL * (p.M * p.N + p.N + p.M));
    };

    gemv_suite.params_to_string = [](const GemvParams &p)
    {
        return "M=" + std::to_string(p.M) + ", N=" + std::to_string(p.N);
    };

    gemv_suite.set_provider([](const GemvParams &p)
                            {
        GemvData d(p.M, p.N);
        randomize(d.W);
        randomize(d.x);
        return d; });

    gemv_suite.set_reference([](const GemvData &d, Tensor<float> &out)
                             { naive_gemv(d.W, d.x, out); });

    gemv_suite.add_shape({1024, 1024});
    gemv_suite.add_shape({4096, 4096});
    gemv_suite.add_shape({10000, 1000});

    gemv_suite.add_kernel("Optimized", [](const GemvData &d, Tensor<float> &out)
                          { gemv(d.W, d.x, out); });

    gemv_suite.run();

    return 0;
}