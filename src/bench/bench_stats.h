#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>

struct BenchStats
{
    double avg_ms;
    double median_ms;
    double min_ms;
    double max_ms;
    double std_dev_ms;
    size_t ops;
    size_t bytes;
    float mse;
};

inline BenchStats compute_stats(std::vector<double> &times, size_t ops, size_t bytes, float mse)
{
    if (times.empty())
        return {};

    std::sort(times.begin(), times.end());

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double avg = sum / times.size();

    double min = times.front();
    double max = times.back();

    double median = 0.0;
    if (times.size() % 2 == 0)
        median = (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2.0;
    else
        median = times[times.size() / 2];

    double variance_sum = 0.0;
    for (double t : times)
        variance_sum += (t - avg) * (t - avg);
    double std_dev = std::sqrt(variance_sum / times.size());

    return {avg, median, min, max, std_dev, ops, bytes, mse};
}