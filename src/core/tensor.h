#pragma once

#include <memory>
#include <immintrin.h>
#include <cstring>
#include <new>

struct AlignedFree
{
    void operator()(void *p) const
    {
        if (p)
            _mm_free(p);
    }
};

template <typename T>
class Tensor
{
private:
    int height_;
    int width_;
    int channels_;
    int stride_;
    std::unique_ptr<T[], AlignedFree> data_;

public:
    int height() const noexcept { return height_; };
    int width() const noexcept { return width_; };
    int channels() const noexcept { return channels_; };
    int stride() const noexcept { return stride_; };
    T *data() const noexcept { return data_.get(); };

    Tensor(int h, int w, int c = 1)
        : height_(h), width_(w), channels_(c)
    {
        size_t bytes_per_row = width_ * channels_ * sizeof(T);
        size_t stride_in_bytes = ((bytes_per_row + 31) / 32) * 32;
        if (stride_in_bytes % 2048 == 0)
            stride_in_bytes += 64;
        stride_ = static_cast<int>(stride_in_bytes / sizeof(T));
        size_t total_size = static_cast<size_t>(height_) * stride_;
        T *raw_ptr = static_cast<T *>(_mm_malloc(total_size * sizeof(T), 32));
        if (!raw_ptr)
            throw std::bad_alloc();

        data_ = std::unique_ptr<T[], AlignedFree>(raw_ptr);
    }

    Tensor<T> clone() const
    {
        Tensor<T> copy(height_, width_, channels_);

        size_t total_bytes = static_cast<size_t>(height_) * stride_ * sizeof(T);

        std::memcpy(copy.data(), data_.get(), total_bytes);

        return copy;
    }

    // disable implicit copying to prevent expensive accidental copies
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    // allow transfer of ownership
    Tensor(Tensor &&) = default;
    Tensor &operator=(Tensor &&) = default;
};