#include "core/tensor.h"

#include <immintrin.h>

inline float hsum_avx(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);

    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    return _mm_cvtss_f32(sums);
}

void naive_gemv(const Tensor2D &W, const Tensor2D &x, Tensor2D &y)
{
    const int W_rows = W.rows();
    const int W_cols = W.cols();

    // shape check
    if (y.rows() != W_rows || x.rows() != W_cols || x.cols() != 1 || y.cols() != 1) 
    {
        throw std::invalid_argument("invalid shape");
    }

    const float *w_ptr = W.data().get();
    const float *x_ptr = x.data().get();
    float *y_ptr = y.data().get();

    for (int i = 0; i < W_rows; ++i) 
    {
        float sum = 0.0f;
        const float *w_row_ptr = &w_ptr[i * W_cols];
        for (int j = 0; j < W_cols; ++j) 
        {
            sum += w_row_ptr[j] * x_ptr[j];
        }
        y_ptr[i] = sum;
    }
}

void unroll_j_gemv(const Tensor2D &W, const Tensor2D &x, Tensor2D &y)
{
    const int W_rows = W.rows();
    const int W_cols = W.cols();

    // shape check
    if (y.rows() != W_rows || x.rows() != W_cols || x.cols() != 1 || y.cols() != 1) 
    {
        throw std::invalid_argument("invalid shape");
    }

    const float *w_ptr = W.data().get();
    const float *x_ptr = x.data().get();
    float *y_ptr = y.data().get();

    for (int i = 0; i < W_rows; ++i) 
    {
        __m256 vsum = _mm256_setzero_ps();
        const float *w_row_ptr = &w_ptr[i * W_cols];
        int j = 0;
        for (; j <= W_cols - 8; j += 8) 
        {
            __m256 vw = _mm256_loadu_ps(&w_row_ptr[j]);
            __m256 vx = _mm256_loadu_ps(&x_ptr[j]);
            vsum = _mm256_fmadd_ps(vw, vx, vsum);
        }
        
        float temp[8];
        _mm256_storeu_ps(temp, vsum);
        float row_sum = 0.0f;
        for (int i = 0; i < 8; ++i) 
        {
            row_sum += temp[i];
        }

        for (; j < W_cols; ++j) 
        {
            row_sum += w_row_ptr[j] * x_ptr[j];
        }
        y_ptr[i] = row_sum;
    }
}

void unroll_i_j_gemv(const Tensor2D &W, const Tensor2D &x, Tensor2D &y) 
{
    const int W_rows = W.rows();
    const int W_cols = W.cols();

    // shape check
    if (y.rows() != W_rows || x.rows() != W_cols || x.cols() != 1 || y.cols() != 1) 
    {
        throw std::invalid_argument("invalid shape");
    }

    const float *w_ptr = W.data().get();
    const float *x_ptr = x.data().get();
    float *y_ptr = y.data().get();

    int i = 0;
    for (; i <= W_rows - 4; i += 4) 
    {
        __m256 vsum0 = _mm256_setzero_ps();
        __m256 vsum1 = _mm256_setzero_ps();
        __m256 vsum2 = _mm256_setzero_ps();
        __m256 vsum3 = _mm256_setzero_ps();

        const float *w_row_ptr_0 = &w_ptr[(i + 0) * W_cols];
        const float *w_row_ptr_1 = &w_ptr[(i + 1) * W_cols];
        const float *w_row_ptr_2 = &w_ptr[(i + 2) * W_cols];
        const float *w_row_ptr_3 = &w_ptr[(i + 3) * W_cols];

        int j = 0;
        for (; j <= W_cols - 8; j += 8) 
        {
            __m256 vx = _mm256_loadu_ps(&x_ptr[j]);

            __m256 vw0 = _mm256_loadu_ps(&w_row_ptr_0[j]);
            __m256 vw1 = _mm256_loadu_ps(&w_row_ptr_1[j]);
            __m256 vw2 = _mm256_loadu_ps(&w_row_ptr_2[j]);
            __m256 vw3 = _mm256_loadu_ps(&w_row_ptr_3[j]);

            vsum0 = _mm256_fmadd_ps(vw0, vx, vsum0);
            vsum1 = _mm256_fmadd_ps(vw1, vx, vsum1);
            vsum2 = _mm256_fmadd_ps(vw2, vx, vsum2);
            vsum3 = _mm256_fmadd_ps(vw3, vx, vsum3);
        }

        float y0 = hsum_avx(vsum0);
        float y1 = hsum_avx(vsum1);
        float y2 = hsum_avx(vsum2);
        float y3 = hsum_avx(vsum3);

        for (; j < W_cols; ++j) 
        {
            y0 += w_row_ptr_0[j] * x_ptr[j];
            y1 += w_row_ptr_1[j] * x_ptr[j];
            y2 += w_row_ptr_2[j] * x_ptr[j];
            y3 += w_row_ptr_3[j] * x_ptr[j];
        }

        y_ptr[i] = y0;
        y_ptr[i + 1] = y1;
        y_ptr[i + 2] = y2;
        y_ptr[i + 3] = y3;
    }

    for (; i < W_rows; ++i) 
    {
        __m256 vsum = _mm256_setzero_ps();
        const float *w_row_ptr = &w_ptr[i * W_cols];
        int j = 0;
        for (; j <= W_cols - 8; j += 8) 
        {
            __m256 vw = _mm256_loadu_ps(&w_row_ptr[j]);
            __m256 vx = _mm256_loadu_ps(&x_ptr[j]);
            vsum = _mm256_fmadd_ps(vw, vx, vsum);
        }
        
        float row_sum = hsum_avx(vsum);

        for (; j < W_cols; ++j) 
        {
            row_sum += w_row_ptr[j] * x_ptr[j];
        }
        y_ptr[i] = row_sum;
    }
}