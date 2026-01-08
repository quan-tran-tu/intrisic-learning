#include "bench/bench.h"

#include <iomanip>
#include <cmath>
#include <sstream>

float calculate_mse(const Tensor<float> &ref, const Tensor<float> &target)
{
    if (ref.height() != target.height() || ref.width() != target.width())
    {
        std::cerr << "dimension mismatch" << std::endl;
        return 9999.0f;
    }

    double sum_sq_diff = 0.0;
    int h = ref.height();
    int w = ref.width();

    for (int y = 0; y < h; ++y)
    {
        const float *r_ptr = ref.data() + (y * ref.stride());
        const float *t_ptr = target.data() + (y * target.stride());

        for (int x = 0; x < w; ++x)
        {
            float diff = r_ptr[x] - t_ptr[x];
            sum_sq_diff += diff * diff;
        }
    }
    return static_cast<float>(sum_sq_diff / (h * w));
}

void print_report(const std::vector<BenchmarkResult> &results)
{
    std::cout << "\n"
              << std::string(105, '=') << "\n";
    std::cout << std::left
              << std::setw(25) << "Kernel"
              << std::setw(12) << "Size"
              << std::setw(12) << "Avg(ms)"
              << std::setw(18) << "Max/Min(ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(15) << "GB/s"
              << std::setw(12) << "MSE" << "\n";
    std::cout << std::string(105, '-') << "\n";

    std::string prev_type;
    bool first = true;

    for (const auto &res : results)
    {
        if (!first && res.type != prev_type)
        {
            std::cout << std::string(105, '-') << "\n";
        }
        first = false;
        prev_type = res.type;

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
    std::cout << std::string(105, '=') << "\n";
}