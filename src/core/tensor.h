#pragma once

#include <memory>
#include <stdexcept>

class Tensor2D
{
private:
    int rows_;
    int cols_;
    std::shared_ptr<float[]> data_;

public:
    int rows() const noexcept { return rows_; };
    int cols() const noexcept { return cols_; };
    auto data() const noexcept { return data_; };

    Tensor2D(int r, int c)
        : rows_(r), cols_(c), data_(std::shared_ptr<float[]>(new float[r * c])) {};
};