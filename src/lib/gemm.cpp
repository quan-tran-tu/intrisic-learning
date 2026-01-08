#include "core/tensor.h"
#include "lib/gemm.h"

#include <immintrin.h>
#include <stdexcept>

const int MC = 256, NC = 64, KC = 256;

void naive_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
{
    int A_height = A.height();
    int A_width = A.width();
    int B_width = B.width();

    int A_stride = A.stride();
    int B_stride = B.stride();
    int C_stride = C.stride();

    const float *a_ptr = A.data();
    const float *b_ptr = B.data();
    float *c_ptr = C.data();

    // shape check
    if (C.height() != A_height || C.width() != B_width)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_height * C_stride), 0.0f);

    for (int i = 0; i < A_height; ++i)
    {
        const float *a_row_ptr = &a_ptr[i * A_stride];
        float *c_row_ptr = &c_ptr[i * C_stride];
        for (int j = 0; j < B_width; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < A_width; ++k)
            {
                sum += a_row_ptr[k] * b_ptr[k * B_stride + j];
            }
            c_row_ptr[j] = sum;
        }
    }
}

void ikj_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
{
    int A_height = A.height();
    int A_width = A.width();
    int B_width = B.width();

    int A_stride = A.stride();
    int B_stride = B.stride();
    int C_stride = C.stride();

    const float *a_ptr = A.data();
    const float *b_ptr = B.data();
    float *c_ptr = C.data();

    // shape check
    if (C.height() != A_height || C.width() != B_width)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_height * C_stride), 0.0f);

    for (int i = 0; i < A_height; ++i)
    {
        for (int k = 0; k < A_width; ++k)
        {
            float a = a_ptr[i * A_stride + k];
            const float *b_row_ptr = &b_ptr[k * B_stride];
            float *c_row_ptr = &c_ptr[i * C_stride];
            for (int j = 0; j < B_width; ++j)
            {
                c_row_ptr[j] += a * b_row_ptr[j];
            }
        }
    }
}

void ikj_broadcast_a_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
{
    int A_height = A.height();
    int A_width = A.width();
    int B_width = B.width();

    int A_stride = A.stride();
    int B_stride = B.stride();
    int C_stride = C.stride();

    const float *a_ptr = A.data();
    const float *b_ptr = B.data();
    float *c_ptr = C.data();

    // shape check
    if (C.height() != A_height || C.width() != B_width)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_height * C_stride), 0.0f);

    for (int i = 0; i < A_height; ++i)
    {
        const float *a_row_ptr = &a_ptr[i * A_stride];
        float *c_row_ptr = &c_ptr[i * C_stride];
        for (int k = 0; k < A_width; ++k)
        {
            __m256 v_a = _mm256_set1_ps(a_row_ptr[k]);
            const float *b_row_ptr = &b_ptr[k * B_stride];
            int j = 0;
            for (; j <= B_width - 8; j += 8)
            {
                __m256 v_b = _mm256_load_ps(&b_row_ptr[j]);
                __m256 v_c = _mm256_load_ps(&c_row_ptr[j]);
                v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                _mm256_store_ps(&c_row_ptr[j], v_c);
            }
            for (; j < B_width; ++j)
            {
                c_row_ptr[j] += a_row_ptr[k] * b_ptr[k * B_stride + j];
            }
        }
    }
}

void register_blocking_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
{
    int A_height = A.height();
    int A_width = A.width();
    int B_width = B.width();

    int A_stride = A.stride();
    int B_stride = B.stride();
    int C_stride = C.stride();

    const float *a_ptr = A.data();
    const float *b_ptr = B.data();
    float *c_ptr = C.data();

    // shape check
    if (C.height() != A_height || C.width() != B_width)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    // std::fill(c_ptr, c_ptr + (A_height * C_stride), 0.0f);

    int i = 0;
    for (; i <= A_height - 4; i += 4)
    {
        const float *a_row_ptr_0 = &a_ptr[(i + 0) * A_stride];
        const float *a_row_ptr_1 = &a_ptr[(i + 1) * A_stride];
        const float *a_row_ptr_2 = &a_ptr[(i + 2) * A_stride];
        const float *a_row_ptr_3 = &a_ptr[(i + 3) * A_stride];

        int j = 0;
        for (; j <= B_width - 8; j += 8)
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int k = 0; k < A_width; ++k)
            {
                __m256 v_a0 = _mm256_set1_ps(a_row_ptr_0[k]);
                __m256 v_a1 = _mm256_set1_ps(a_row_ptr_1[k]);
                __m256 v_a2 = _mm256_set1_ps(a_row_ptr_2[k]);
                __m256 v_a3 = _mm256_set1_ps(a_row_ptr_3[k]);

                __m256 v_b = _mm256_load_ps(&b_ptr[k * B_stride + j]);

                acc0 = _mm256_fmadd_ps(v_a0, v_b, acc0);
                acc1 = _mm256_fmadd_ps(v_a1, v_b, acc1);
                acc2 = _mm256_fmadd_ps(v_a2, v_b, acc2);
                acc3 = _mm256_fmadd_ps(v_a3, v_b, acc3);
            }

            _mm256_store_ps(&c_ptr[(i + 0) * C_stride + j], acc0);
            _mm256_store_ps(&c_ptr[(i + 1) * C_stride + j], acc1);
            _mm256_store_ps(&c_ptr[(i + 2) * C_stride + j], acc2);
            _mm256_store_ps(&c_ptr[(i + 3) * C_stride + j], acc3);
        }

        for (; j < B_width; ++j)
        {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            for (int k = 0; k < A_width; ++k)
            {
                sum0 += a_row_ptr_0[k] * b_ptr[k * B_stride + j];
                sum1 += a_row_ptr_1[k] * b_ptr[k * B_stride + j];
                sum2 += a_row_ptr_2[k] * b_ptr[k * B_stride + j];
                sum3 += a_row_ptr_3[k] * b_ptr[k * B_stride + j];
            }
            c_ptr[(i + 0) * C_stride + j] = sum0;
            c_ptr[(i + 1) * C_stride + j] = sum1;
            c_ptr[(i + 2) * C_stride + j] = sum2;
            c_ptr[(i + 3) * C_stride + j] = sum3;
        }
    }

    for (; i < A_height; ++i)
    {
        const float *a_row_ptr = &a_ptr[i * A_stride];
        float *c_row_ptr = &c_ptr[i * C_stride];
        for (int j = 0; j < B_width; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < A_width; ++k)
            {
                sum += a_row_ptr[k] * b_ptr[k * B_stride + j];
            }
            c_row_ptr[j] = sum;
        }
    }
}

void cache_tiling_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
{
    int A_height = A.height();
    int A_width = A.width();
    int B_width = B.width();

    int A_stride = A.stride();
    int B_stride = B.stride();
    int C_stride = C.stride();

    const float *a_ptr = A.data();
    const float *b_ptr = B.data();
    float *c_ptr = C.data();

    // shape check
    if (C.height() != A_height || C.width() != B_width)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_height * C_stride), 0.0f);

    for (int jc = 0; jc < B_width; jc += NC)
    {
        int j_min = std::min(jc + NC, B_width);
        for (int kc = 0; kc < A_width; kc += KC)
        {
            int k_min = std::min(kc + KC, A_width);
            for (int ic = 0; ic < A_height; ic += MC)
            {
                int i_min = std::min(ic + MC, A_height);

                int i = ic;
                for (; i <= i_min - 4; i += 4)
                {
                    const float *a_row_ptr_0 = &a_ptr[(i + 0) * A_stride];
                    const float *a_row_ptr_1 = &a_ptr[(i + 1) * A_stride];
                    const float *a_row_ptr_2 = &a_ptr[(i + 2) * A_stride];
                    const float *a_row_ptr_3 = &a_ptr[(i + 3) * A_stride];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 acc0 = _mm256_load_ps(&c_ptr[(i + 0) * C_stride + j]);
                        __m256 acc1 = _mm256_load_ps(&c_ptr[(i + 1) * C_stride + j]);
                        __m256 acc2 = _mm256_load_ps(&c_ptr[(i + 2) * C_stride + j]);
                        __m256 acc3 = _mm256_load_ps(&c_ptr[(i + 3) * C_stride + j]);

                        for (int k = kc; k < k_min; ++k)
                        {
                            __m256 v_a0 = _mm256_set1_ps(a_row_ptr_0[k]);
                            __m256 v_a1 = _mm256_set1_ps(a_row_ptr_1[k]);
                            __m256 v_a2 = _mm256_set1_ps(a_row_ptr_2[k]);
                            __m256 v_a3 = _mm256_set1_ps(a_row_ptr_3[k]);

                            __m256 v_b = _mm256_load_ps(&b_ptr[k * B_stride + j]);

                            acc0 = _mm256_fmadd_ps(v_a0, v_b, acc0);
                            acc1 = _mm256_fmadd_ps(v_a1, v_b, acc1);
                            acc2 = _mm256_fmadd_ps(v_a2, v_b, acc2);
                            acc3 = _mm256_fmadd_ps(v_a3, v_b, acc3);
                        }

                        _mm256_store_ps(&c_ptr[(i + 0) * C_stride + j], acc0);
                        _mm256_store_ps(&c_ptr[(i + 1) * C_stride + j], acc1);
                        _mm256_store_ps(&c_ptr[(i + 2) * C_stride + j], acc2);
                        _mm256_store_ps(&c_ptr[(i + 3) * C_stride + j], acc3);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum0 += a_row_ptr_0[k] * b_ptr[k * B_stride + j];
                            sum1 += a_row_ptr_1[k] * b_ptr[k * B_stride + j];
                            sum2 += a_row_ptr_2[k] * b_ptr[k * B_stride + j];
                            sum3 += a_row_ptr_3[k] * b_ptr[k * B_stride + j];
                        }
                        c_ptr[(i + 0) * C_stride + j] += sum0;
                        c_ptr[(i + 1) * C_stride + j] += sum1;
                        c_ptr[(i + 2) * C_stride + j] += sum2;
                        c_ptr[(i + 3) * C_stride + j] += sum3;
                    }
                }

                for (; i < i_min; ++i)
                {
                    const float *a_row_ptr = &a_ptr[i * A_stride];
                    float *c_row_ptr = &c_ptr[i * C_stride];
                    for (int k = kc; k < k_min; ++k)
                    {
                        __m256 v_a = _mm256_set1_ps(a_row_ptr[k]);
                        const float *b_row_ptr = &b_ptr[k * B_stride];
                        int j = jc;
                        for (; j <= j_min - 8; j += 8)
                        {
                            __m256 v_b = _mm256_load_ps(&b_row_ptr[j]);
                            __m256 v_c = _mm256_load_ps(&c_row_ptr[j]);
                            v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                            _mm256_store_ps(&c_row_ptr[j], v_c);
                        }
                        for (; j < j_min; ++j)
                        {
                            c_row_ptr[j] += a_row_ptr[k] * b_ptr[k * B_stride + j];
                        }
                    }
                }
            }
        }
    }
}

void pack_B(int k_min, int k_max, int j_min, int j_max, int B_stride, const float *b_ptr, float *packed_buffer)
{
    float *pack_iter = packed_buffer;

    for (int j = j_min; j < j_max; j += 8)
    {
        for (int k = k_min; k < k_max; ++k)
        {
            const float *b_row_ptr = b_ptr + k * B_stride + j;

            if (j + 8 <= j_max)
            {
                _mm256_store_ps(pack_iter, _mm256_loadu_ps(b_row_ptr));
            }
            else
            {
                alignas(32) float temp[8] = {0};
                for (int n = 0; n < (j_max - j); ++n)
                {
                    temp[n] = b_row_ptr[n];
                }
                _mm256_store_ps(pack_iter, _mm256_load_ps(temp));
            }
            pack_iter += 8;
        }
    }
}

void cache_tiling_packed_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
{
    int A_height = A.height();
    int A_width = A.width();
    int B_width = B.width();

    int A_stride = A.stride();
    int B_stride = B.stride();
    int C_stride = C.stride();

    const float *a_ptr = A.data();
    const float *b_ptr = B.data();
    float *c_ptr = C.data();

    // shape check
    if (C.height() != A_height || C.width() != B_width)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_height * C_stride), 0.0f);

    alignas(32) float packed_B[NC * KC];

    for (int jc = 0; jc < B_width; jc += NC)
    {
        int j_min = std::min(jc + NC, B_width);
        for (int kc = 0; kc < A_width; kc += KC)
        {
            int k_min = std::min(kc + KC, A_width);

            pack_B(kc, k_min, jc, j_min, B_stride, b_ptr, packed_B);

            for (int ic = 0; ic < A_height; ic += MC)
            {
                int i_min = std::min(ic + MC, A_height);

                int i = ic;
                for (; i <= i_min - 4; i += 4)
                {
                    const float *b_pack_ptr = packed_B;

                    const float *a_row_ptr_0 = &a_ptr[(i + 0) * A_stride];
                    const float *a_row_ptr_1 = &a_ptr[(i + 1) * A_stride];
                    const float *a_row_ptr_2 = &a_ptr[(i + 2) * A_stride];
                    const float *a_row_ptr_3 = &a_ptr[(i + 3) * A_stride];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 acc0 = _mm256_load_ps(&c_ptr[(i + 0) * C_stride + j]);
                        __m256 acc1 = _mm256_load_ps(&c_ptr[(i + 1) * C_stride + j]);
                        __m256 acc2 = _mm256_load_ps(&c_ptr[(i + 2) * C_stride + j]);
                        __m256 acc3 = _mm256_load_ps(&c_ptr[(i + 3) * C_stride + j]);

                        for (int k = kc; k < k_min; ++k)
                        {
                            __m256 v_a0 = _mm256_set1_ps(a_row_ptr_0[k]);
                            __m256 v_a1 = _mm256_set1_ps(a_row_ptr_1[k]);
                            __m256 v_a2 = _mm256_set1_ps(a_row_ptr_2[k]);
                            __m256 v_a3 = _mm256_set1_ps(a_row_ptr_3[k]);

                            __m256 v_b = _mm256_load_ps(b_pack_ptr);
                            b_pack_ptr += 8;

                            acc0 = _mm256_fmadd_ps(v_a0, v_b, acc0);
                            acc1 = _mm256_fmadd_ps(v_a1, v_b, acc1);
                            acc2 = _mm256_fmadd_ps(v_a2, v_b, acc2);
                            acc3 = _mm256_fmadd_ps(v_a3, v_b, acc3);
                        }

                        _mm256_store_ps(&c_ptr[(i + 0) * C_stride + j], acc0);
                        _mm256_store_ps(&c_ptr[(i + 1) * C_stride + j], acc1);
                        _mm256_store_ps(&c_ptr[(i + 2) * C_stride + j], acc2);
                        _mm256_store_ps(&c_ptr[(i + 3) * C_stride + j], acc3);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum0 += a_row_ptr_0[k] * b_ptr[k * B_stride + j];
                            sum1 += a_row_ptr_1[k] * b_ptr[k * B_stride + j];
                            sum2 += a_row_ptr_2[k] * b_ptr[k * B_stride + j];
                            sum3 += a_row_ptr_3[k] * b_ptr[k * B_stride + j];
                        }
                        c_ptr[(i + 0) * C_stride + j] += sum0;
                        c_ptr[(i + 1) * C_stride + j] += sum1;
                        c_ptr[(i + 2) * C_stride + j] += sum2;
                        c_ptr[(i + 3) * C_stride + j] += sum3;
                    }
                }

                for (; i < i_min; ++i)
                {
                    const float *a_row_ptr = &a_ptr[i * A_stride];
                    const float *b_pack_ptr = packed_B;
                    float *c_row_ptr = &c_ptr[i * C_stride];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 v_c = _mm256_load_ps(&c_row_ptr[j]);
                        for (int k = kc; k < k_min; ++k)
                        {
                            __m256 v_a = _mm256_set1_ps(a_row_ptr[k]);
                            __m256 v_b = _mm256_load_ps(b_pack_ptr);
                            b_pack_ptr += 8;
                            v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                        }
                        _mm256_store_ps(&c_row_ptr[j], v_c);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum += a_row_ptr[k] * b_ptr[k * B_stride + j];
                        }
                        c_row_ptr[j] += sum;
                    }
                }
            }
        }
    }
}

void packed_parallel_gemm(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
{
    int A_height = A.height();
    int A_width = A.width();
    int B_width = B.width();

    int A_stride = A.stride();
    int B_stride = B.stride();
    int C_stride = C.stride();

    const float *a_ptr = A.data();
    const float *b_ptr = B.data();
    float *c_ptr = C.data();

    // shape check
    if (C.height() != A_height || C.width() != B_width)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_height * C_stride), 0.0f);

#pragma omp parallel for
    for (int jc = 0; jc < B_width; jc += NC)
    {
        alignas(32) float packed_B[NC * KC];

        int j_min = std::min(jc + NC, B_width);
        for (int kc = 0; kc < A_width; kc += KC)
        {
            int k_min = std::min(kc + KC, A_width);

            pack_B(kc, k_min, jc, j_min, B_stride, b_ptr, packed_B);

            for (int ic = 0; ic < A_height; ic += MC)
            {
                int i_min = std::min(ic + MC, A_height);

                int i = ic;
                for (; i <= i_min - 4; i += 4)
                {
                    const float *b_pack_ptr = packed_B;

                    const float *a_row_ptr_0 = &a_ptr[(i + 0) * A_stride];
                    const float *a_row_ptr_1 = &a_ptr[(i + 1) * A_stride];
                    const float *a_row_ptr_2 = &a_ptr[(i + 2) * A_stride];
                    const float *a_row_ptr_3 = &a_ptr[(i + 3) * A_stride];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 acc0 = _mm256_load_ps(&c_ptr[(i + 0) * C_stride + j]);
                        __m256 acc1 = _mm256_load_ps(&c_ptr[(i + 1) * C_stride + j]);
                        __m256 acc2 = _mm256_load_ps(&c_ptr[(i + 2) * C_stride + j]);
                        __m256 acc3 = _mm256_load_ps(&c_ptr[(i + 3) * C_stride + j]);

                        for (int k = kc; k < k_min; ++k)
                        {
                            __m256 v_a0 = _mm256_set1_ps(a_row_ptr_0[k]);
                            __m256 v_a1 = _mm256_set1_ps(a_row_ptr_1[k]);
                            __m256 v_a2 = _mm256_set1_ps(a_row_ptr_2[k]);
                            __m256 v_a3 = _mm256_set1_ps(a_row_ptr_3[k]);

                            __m256 v_b = _mm256_load_ps(b_pack_ptr);
                            b_pack_ptr += 8;

                            acc0 = _mm256_fmadd_ps(v_a0, v_b, acc0);
                            acc1 = _mm256_fmadd_ps(v_a1, v_b, acc1);
                            acc2 = _mm256_fmadd_ps(v_a2, v_b, acc2);
                            acc3 = _mm256_fmadd_ps(v_a3, v_b, acc3);
                        }

                        _mm256_store_ps(&c_ptr[(i + 0) * C_stride + j], acc0);
                        _mm256_store_ps(&c_ptr[(i + 1) * C_stride + j], acc1);
                        _mm256_store_ps(&c_ptr[(i + 2) * C_stride + j], acc2);
                        _mm256_store_ps(&c_ptr[(i + 3) * C_stride + j], acc3);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum0 += a_row_ptr_0[k] * b_ptr[k * B_stride + j];
                            sum1 += a_row_ptr_1[k] * b_ptr[k * B_stride + j];
                            sum2 += a_row_ptr_2[k] * b_ptr[k * B_stride + j];
                            sum3 += a_row_ptr_3[k] * b_ptr[k * B_stride + j];
                        }
                        c_ptr[(i + 0) * C_stride + j] += sum0;
                        c_ptr[(i + 1) * C_stride + j] += sum1;
                        c_ptr[(i + 2) * C_stride + j] += sum2;
                        c_ptr[(i + 3) * C_stride + j] += sum3;
                    }
                }

                for (; i < i_min; ++i)
                {
                    const float *a_row_ptr = &a_ptr[i * A_stride];
                    const float *b_pack_ptr = packed_B;
                    float *c_row_ptr = &c_ptr[i * C_stride];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 v_c = _mm256_load_ps(&c_row_ptr[j]);
                        for (int k = kc; k < k_min; ++k)
                        {
                            __m256 v_a = _mm256_set1_ps(a_row_ptr[k]);
                            __m256 v_b = _mm256_load_ps(b_pack_ptr);
                            b_pack_ptr += 8;
                            v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                        }
                        _mm256_store_ps(&c_row_ptr[j], v_c);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum += a_row_ptr[k] * b_ptr[k * B_stride + j];
                        }
                        c_row_ptr[j] += sum;
                    }
                }
            }
        }
    }
}