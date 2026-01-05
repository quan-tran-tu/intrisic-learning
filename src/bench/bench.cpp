#include "bench/bench.h"

#include <iomanip>
#include <cmath>
#include <sstream>

float calculate_mse(const Tensor2D &ref, const Tensor2D &target)
{
    if (ref.rows() != target.rows() || ref.cols() != target.cols())
    {
        std::cerr << "dimension mismatch" << std::endl;
        return 9999.0f;
    }

    float mse = 0.0f;
    int size = ref.rows() * ref.cols();
    const float *r_ptr = ref.data().get();
    const float *t_ptr = target.data().get();

    for (int i = 0; i < size; ++i)
    {
        float diff = r_ptr[i] - t_ptr[i];
        mse += diff * diff;
    }
    return mse / size;
}

void print_report(const std::vector<BenchmarkResult> &results)
{
    std::cout << "\n"
              << std::string(110, '=') << "\n";
    std::cout << std::left
              << std::setw(25) << "Kernel"
              << std::setw(12) << "Size"
              << std::setw(12) << "Avg(ms)"
              << std::setw(18) << "Min/Max(ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(15) << "GB/s"
              << std::setw(12) << "MSE" << "\n";
    std::cout << std::string(110, '-') << "\n";

    for (const auto &res : results)
    {
        // GFLOPS = 10^9 ops / sec
        double gflops = (double)res.total_ops / (res.avg_time_ms * 1e6);
        // GB/s = 10^9 bytes / sec
        double gbs = (double)res.total_bytes / (res.avg_time_ms * 1e6);

        std::stringstream range_ss;
        range_ss << std::fixed << std::setprecision(2) << res.min_time_ms << "/" << res.max_time_ms;

        std::cout << std::left
                  << std::setw(25) << res.name
                  << std::setw(12) << res.size
                  << std::setw(12) << std::fixed << std::setprecision(3) << res.avg_time_ms
                  << std::setw(18) << range_ss.str()
                  << std::setw(15) << std::setprecision(2) << gflops
                  << std::setw(15) << std::setprecision(2) << gbs
                  << std::setw(12) << std::setprecision(5) << res.mse << "\n";
    }
    std::cout << std::string(110, '=') << "\n";
}