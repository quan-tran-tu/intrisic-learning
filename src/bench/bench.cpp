#include "bench.h"
#include <iostream>

float calculate_mse(const Tensor<float> &ref, const Tensor<float> &target)
{
    if (ref.height() != target.height() || ref.width() != target.width())
        return 9999.0f;

    double sum_sq_diff = 0.0;
    size_t total = ref.height() * ref.width();
    const float *r = ref.data();
    const float *t = target.data();

    for (size_t i = 0; i < total; ++i)
    {
        float diff = r[i] - t[i];
        sum_sq_diff += diff * diff;
    }
    return static_cast<float>(sum_sq_diff / total);
}