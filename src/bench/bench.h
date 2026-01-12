#pragma once

#include "core/tensor.h"
#include "bench_stats.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <iomanip>
#include <sstream>

struct BenchConfig
{
    int warmup_iters = 5;
    int run_iters = 20;
    bool verify_output = true;
};

float calculate_mse(const Tensor<float> &ref, const Tensor<float> &target);

template <typename P, typename Data>
class BenchmarkSuite
{
public:
    using KernelFunc = std::function<void(const Data &, Tensor<float> &)>;
    using DataProvider = std::function<Data(const P &)>;
    using RefFunc = std::function<void(const Data &, Tensor<float> &)>;

    struct KernelEntry
    {
        std::string name;
        KernelFunc func;
    };

    BenchmarkSuite(std::string name, BenchConfig config = {})
        : suite_name_(name), config_(config) {}

    void add_shape(P params)
    {
        shapes_.push_back(params);
    }

    void add_kernel(std::string name, KernelFunc func)
    {
        kernels_.push_back({name, func});
    }

    void set_provider(DataProvider dp) { provider_ = dp; }

    void set_reference(RefFunc rf) { reference_func_ = rf; }

    std::function<std::pair<size_t, size_t>(const P &)> complexity_calc;

    void run()
    {
        print_header();

        for (const auto &params : shapes_)
        {
            Data data = provider_(params);

            Tensor<float> ref_out = data.output.clone();
            if (config_.verify_output && reference_func_)
            {
                reference_func_(data, ref_out);
            }

            auto [ops, bytes] = complexity_calc(params);

            for (const auto &kernel : kernels_)
            {
                Tensor<float> test_out = data.output.clone();

                for (int i = 0; i < config_.warmup_iters; ++i)
                {
                    kernel.func(data, test_out);
                }

                std::vector<double> times;
                times.reserve(config_.run_iters);

                for (int i = 0; i < config_.run_iters; ++i)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    kernel.func(data, test_out);
                    auto end = std::chrono::high_resolution_clock::now();

                    double ms = std::chrono::duration<double, std::milli>(end - start).count();
                    times.push_back(ms);
                }

                float mse = -1.0f;
                if (config_.verify_output && reference_func_)
                {
                    mse = calculate_mse(ref_out, test_out);
                }

                BenchStats stats = compute_stats(times, ops, bytes, mse);
                print_row(kernel.name, params_to_string(params), stats);
            }
            std::cout << std::string(120, '-') << "\n";
        }
        std::cout << std::string(120, '=') << "\n\n";
    }

    std::function<std::string(const P &)> params_to_string;

private:
    std::string suite_name_;
    BenchConfig config_;
    std::vector<P> shapes_;
    std::vector<KernelEntry> kernels_;
    DataProvider provider_;
    RefFunc reference_func_;

    void print_header()
    {
        std::cout << "SUITE: " << suite_name_ << "\n";
        std::cout << std::string(120, '=') << "\n";
        std::cout << std::left
                  << std::setw(25) << "Kernel"
                  << std::setw(18) << "Shape"
                  << std::setw(12) << "Med(ms)"
                  << std::setw(12) << "Avg(ms)"
                  << std::setw(10) << "StdDev"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(12) << "GB/s"
                  << std::setw(10) << "MSE" << "\n";
        std::cout << std::string(120, '-') << "\n";
    }

    void print_row(const std::string &name, const std::string &shape, const BenchStats &s)
    {
        double gflops = (s.median_ms > 0) ? (double)s.ops / (s.median_ms * 1e6) : 0.0;
        double gbs = (s.median_ms > 0) ? (double)s.bytes / (s.median_ms * 1e6) : 0.0;

        std::cout << std::left
                  << std::setw(25) << name
                  << std::setw(18) << shape
                  << std::setw(12) << std::fixed << std::setprecision(3) << s.median_ms
                  << std::setw(12) << s.avg_ms
                  << std::setw(10) << std::setprecision(2) << s.std_dev_ms
                  << std::setw(12) << std::setprecision(2) << gflops
                  << std::setw(12) << std::setprecision(2) << gbs
                  << std::setw(10) << std::scientific << std::setprecision(2) << s.mse << "\n";
    }
};