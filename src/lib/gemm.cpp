#include "core/tensor.h"
#include "lib/gemm.h"

#include <immintrin.h>

const int MC = 256, NC = 64, KC = 256;

void naive_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C)
{
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float *a_ptr = A.data().get();
    const float *b_ptr = B.data().get();
    float *c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    for (int i = 0; i < A_rows; ++i)
    {
        const float *a_row_ptr = &a_ptr[i * A_cols];
        float *c_row_ptr = &c_ptr[i * B_cols];
        for (int j = 0; j < B_cols; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; ++k)
            {
                sum += a_row_ptr[k] * b_ptr[k * B_cols + j];
            }
            c_row_ptr[j] = sum;
        }
    }
}

void ikj_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C)
{
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float *a_ptr = A.data().get();
    const float *b_ptr = B.data().get();
    float *c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    for (int i = 0; i < A_rows; ++i)
    {
        for (int k = 0; k < A_cols; ++k)
        {
            float a = a_ptr[i * A_cols + k];
            const float *b_row_ptr = &b_ptr[k * B_cols];
            float *c_row_ptr = &c_ptr[i * B_cols];
            for (int j = 0; j < B_cols; ++j)
            {
                c_row_ptr[j] += a * b_row_ptr[j];
            }
        }
    }
}

void ikj_broadcast_a_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C)
{
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float *a_ptr = A.data().get();
    const float *b_ptr = B.data().get();
    float *c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    for (int i = 0; i < A_rows; ++i)
    {
        const float *a_row_ptr = &a_ptr[i * A_cols];
        float *c_row_ptr = &c_ptr[i * B_cols];
        for (int k = 0; k < A_cols; ++k)
        {
            __m256 v_a = _mm256_set1_ps(a_row_ptr[k]);
            const float *b_row_ptr = &b_ptr[k * B_cols];
            int j = 0;
            for (; j <= B_cols - 8; j += 8)
            {
                __m256 v_b = _mm256_loadu_ps(&b_row_ptr[j]);
                __m256 v_c = _mm256_loadu_ps(&c_row_ptr[j]);
                v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                _mm256_storeu_ps(&c_row_ptr[j], v_c);
            }
            for (; j < B_cols; ++j)
            {
                c_row_ptr[j] += a_row_ptr[k] * b_ptr[k * B_cols + j];
            }
        }
    }
}

void register_blocking_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C)
{
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float *a_ptr = A.data().get();
    const float *b_ptr = B.data().get();
    float *c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    // std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    int i = 0;
    for (; i <= A_rows - 4; i += 4)
    {
        const float *a_row_ptr_0 = &a_ptr[(i + 0) * A_cols];
        const float *a_row_ptr_1 = &a_ptr[(i + 1) * A_cols];
        const float *a_row_ptr_2 = &a_ptr[(i + 2) * A_cols];
        const float *a_row_ptr_3 = &a_ptr[(i + 3) * A_cols];

        int j = 0;
        for (; j <= B_cols - 8; j += 8)
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int k = 0; k < A_cols; ++k)
            {
                __m256 v_a0 = _mm256_set1_ps(a_row_ptr_0[k]);
                __m256 v_a1 = _mm256_set1_ps(a_row_ptr_1[k]);
                __m256 v_a2 = _mm256_set1_ps(a_row_ptr_2[k]);
                __m256 v_a3 = _mm256_set1_ps(a_row_ptr_3[k]);

                __m256 v_b = _mm256_loadu_ps(&b_ptr[k * B_cols + j]);

                acc0 = _mm256_fmadd_ps(v_a0, v_b, acc0);
                acc1 = _mm256_fmadd_ps(v_a1, v_b, acc1);
                acc2 = _mm256_fmadd_ps(v_a2, v_b, acc2);
                acc3 = _mm256_fmadd_ps(v_a3, v_b, acc3);
            }

            _mm256_storeu_ps(&c_ptr[(i + 0) * B_cols + j], acc0);
            _mm256_storeu_ps(&c_ptr[(i + 1) * B_cols + j], acc1);
            _mm256_storeu_ps(&c_ptr[(i + 2) * B_cols + j], acc2);
            _mm256_storeu_ps(&c_ptr[(i + 3) * B_cols + j], acc3);
        }

        for (; j < B_cols; ++j)
        {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            for (int k = 0; k < A_cols; ++k)
            {
                sum0 += a_row_ptr_0[k] * b_ptr[k * B_cols + j];
                sum1 += a_row_ptr_1[k] * b_ptr[k * B_cols + j];
                sum2 += a_row_ptr_2[k] * b_ptr[k * B_cols + j];
                sum3 += a_row_ptr_3[k] * b_ptr[k * B_cols + j];
            }
            c_ptr[(i + 0) * B_cols + j] = sum0;
            c_ptr[(i + 1) * B_cols + j] = sum1;
            c_ptr[(i + 2) * B_cols + j] = sum2;
            c_ptr[(i + 3) * B_cols + j] = sum3;
        }
    }

    for (; i < A_rows; ++i)
    {
        const float *a_row_ptr = &a_ptr[i * A_cols];
        float *c_row_ptr = &c_ptr[i * B_cols];
        for (int j = 0; j < B_cols; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; ++k)
            {
                sum += a_row_ptr[k] * b_ptr[k * B_cols + j];
            }
            c_row_ptr[j] = sum;
        }
    }
}

void cache_tiling_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C)
{
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float *a_ptr = A.data().get();
    const float *b_ptr = B.data().get();
    float *c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    for (int jc = 0; jc < B_cols; jc += NC)
    {
        int j_min = std::min(jc + NC, B_cols);
        for (int kc = 0; kc < A_cols; kc += KC)
        {
            int k_min = std::min(kc + KC, A_cols);
            for (int ic = 0; ic < A_rows; ic += MC)
            {
                int i_min = std::min(ic + MC, A_rows);

                int i = ic;
                for (; i <= i_min - 4; i += 4)
                {
                    const float *a_row_ptr_0 = &a_ptr[(i + 0) * A_cols];
                    const float *a_row_ptr_1 = &a_ptr[(i + 1) * A_cols];
                    const float *a_row_ptr_2 = &a_ptr[(i + 2) * A_cols];
                    const float *a_row_ptr_3 = &a_ptr[(i + 3) * A_cols];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 acc0 = _mm256_loadu_ps(&c_ptr[(i + 0) * B_cols + j]);
                        __m256 acc1 = _mm256_loadu_ps(&c_ptr[(i + 1) * B_cols + j]);
                        __m256 acc2 = _mm256_loadu_ps(&c_ptr[(i + 2) * B_cols + j]);
                        __m256 acc3 = _mm256_loadu_ps(&c_ptr[(i + 3) * B_cols + j]);

                        for (int k = kc; k < k_min; ++k)
                        {
                            __m256 v_a0 = _mm256_set1_ps(a_row_ptr_0[k]);
                            __m256 v_a1 = _mm256_set1_ps(a_row_ptr_1[k]);
                            __m256 v_a2 = _mm256_set1_ps(a_row_ptr_2[k]);
                            __m256 v_a3 = _mm256_set1_ps(a_row_ptr_3[k]);

                            __m256 v_b = _mm256_loadu_ps(&b_ptr[k * B_cols + j]);

                            acc0 = _mm256_fmadd_ps(v_a0, v_b, acc0);
                            acc1 = _mm256_fmadd_ps(v_a1, v_b, acc1);
                            acc2 = _mm256_fmadd_ps(v_a2, v_b, acc2);
                            acc3 = _mm256_fmadd_ps(v_a3, v_b, acc3);
                        }

                        _mm256_storeu_ps(&c_ptr[(i + 0) * B_cols + j], acc0);
                        _mm256_storeu_ps(&c_ptr[(i + 1) * B_cols + j], acc1);
                        _mm256_storeu_ps(&c_ptr[(i + 2) * B_cols + j], acc2);
                        _mm256_storeu_ps(&c_ptr[(i + 3) * B_cols + j], acc3);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum0 += a_row_ptr_0[k] * b_ptr[k * B_cols + j];
                            sum1 += a_row_ptr_1[k] * b_ptr[k * B_cols + j];
                            sum2 += a_row_ptr_2[k] * b_ptr[k * B_cols + j];
                            sum3 += a_row_ptr_3[k] * b_ptr[k * B_cols + j];
                        }
                        c_ptr[(i + 0) * B_cols + j] += sum0;
                        c_ptr[(i + 1) * B_cols + j] += sum1;
                        c_ptr[(i + 2) * B_cols + j] += sum2;
                        c_ptr[(i + 3) * B_cols + j] += sum3;
                    }
                }

                for (; i < i_min; ++i)
                {
                    const float *a_row_ptr = &a_ptr[i * A_cols];
                    float *c_row_ptr = &c_ptr[i * B_cols];
                    for (int k = kc; k < k_min; ++k)
                    {
                        __m256 v_a = _mm256_set1_ps(a_row_ptr[k]);
                        const float *b_row_ptr = &b_ptr[k * B_cols];
                        int j = jc;
                        for (; j <= j_min - 8; j += 8)
                        {
                            __m256 v_b = _mm256_loadu_ps(&b_row_ptr[j]);
                            __m256 v_c = _mm256_loadu_ps(&c_row_ptr[j]);
                            v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                            _mm256_storeu_ps(&c_row_ptr[j], v_c);
                        }
                        for (; j < j_min; ++j)
                        {
                            c_row_ptr[j] += a_row_ptr[k] * b_ptr[k * B_cols + j];
                        }
                    }
                }
            }
        }
    }
}

void pack_B(int k_min, int k_max, int j_min, int j_max, int B_cols, const float *b_ptr, float *packed_buffer)
{
    float *pack_iter = packed_buffer;

    for (int j = j_min; j < j_max; j += 8)
    {
        for (int k = k_min; k < k_max; ++k)
        {
            const float *b_row_ptr = b_ptr + k * B_cols + j;

            if (j + 8 <= j_max)
            {
                _mm256_store_ps(pack_iter, _mm256_loadu_ps(b_row_ptr));
            }
            else
            {
                for (int n = 0; n < 8; ++n)
                {
                    if (j + n < j_max)
                        pack_iter[n] = b_row_ptr[n];
                    else
                        pack_iter[n] = 0.0f;
                }
            }
            pack_iter += 8;
        }
    }
}

void cache_tiling_packed_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C)
{
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float *a_ptr = A.data().get();
    const float *b_ptr = B.data().get();
    float *c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    alignas(32) float packed_B[NC * KC];

    for (int jc = 0; jc < B_cols; jc += NC)
    {
        int j_min = std::min(jc + NC, B_cols);
        for (int kc = 0; kc < A_cols; kc += KC)
        {
            int k_min = std::min(kc + KC, A_cols);

            pack_B(kc, k_min, jc, j_min, B_cols, b_ptr, packed_B);

            for (int ic = 0; ic < A_rows; ic += MC)
            {
                int i_min = std::min(ic + MC, A_rows);

                int i = ic;
                for (; i <= i_min - 4; i += 4)
                {
                    const float *b_pack_ptr = packed_B;

                    const float *a_row_ptr_0 = &a_ptr[(i + 0) * A_cols];
                    const float *a_row_ptr_1 = &a_ptr[(i + 1) * A_cols];
                    const float *a_row_ptr_2 = &a_ptr[(i + 2) * A_cols];
                    const float *a_row_ptr_3 = &a_ptr[(i + 3) * A_cols];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 acc0 = _mm256_loadu_ps(&c_ptr[(i + 0) * B_cols + j]);
                        __m256 acc1 = _mm256_loadu_ps(&c_ptr[(i + 1) * B_cols + j]);
                        __m256 acc2 = _mm256_loadu_ps(&c_ptr[(i + 2) * B_cols + j]);
                        __m256 acc3 = _mm256_loadu_ps(&c_ptr[(i + 3) * B_cols + j]);

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

                        _mm256_storeu_ps(&c_ptr[(i + 0) * B_cols + j], acc0);
                        _mm256_storeu_ps(&c_ptr[(i + 1) * B_cols + j], acc1);
                        _mm256_storeu_ps(&c_ptr[(i + 2) * B_cols + j], acc2);
                        _mm256_storeu_ps(&c_ptr[(i + 3) * B_cols + j], acc3);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum0 += a_row_ptr_0[k] * b_ptr[k * B_cols + j];
                            sum1 += a_row_ptr_1[k] * b_ptr[k * B_cols + j];
                            sum2 += a_row_ptr_2[k] * b_ptr[k * B_cols + j];
                            sum3 += a_row_ptr_3[k] * b_ptr[k * B_cols + j];
                        }
                        c_ptr[(i + 0) * B_cols + j] += sum0;
                        c_ptr[(i + 1) * B_cols + j] += sum1;
                        c_ptr[(i + 2) * B_cols + j] += sum2;
                        c_ptr[(i + 3) * B_cols + j] += sum3;
                    }
                }

                for (; i < i_min; ++i)
                {
                    const float *a_row_ptr = &a_ptr[i * A_cols];
                    const float *b_pack_ptr = packed_B;
                    float *c_row_ptr = &c_ptr[i * B_cols];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 v_c = _mm256_loadu_ps(&c_row_ptr[j]);
                        for (int k = kc; k < k_min; ++k)
                        {
                            __m256 v_a = _mm256_set1_ps(a_row_ptr[k]);
                            __m256 v_b = _mm256_load_ps(b_pack_ptr);
                            b_pack_ptr += 8;
                            v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                        }
                        _mm256_storeu_ps(&c_row_ptr[j], v_c);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum += a_row_ptr[k] * b_ptr[k * B_cols + j];
                        }
                        c_row_ptr[j] += sum;
                    }
                }
            }
        }
    }
}

void packed_parallel_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C)
{
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float *a_ptr = A.data().get();
    const float *b_ptr = B.data().get();
    float *c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols)
    {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

#pragma omp parallel for
    for (int jc = 0; jc < B_cols; jc += NC)
    {
        alignas(32) float packed_B[NC * KC];

        int j_min = std::min(jc + NC, B_cols);
        for (int kc = 0; kc < A_cols; kc += KC)
        {
            int k_min = std::min(kc + KC, A_cols);

            pack_B(kc, k_min, jc, j_min, B_cols, b_ptr, packed_B);

            for (int ic = 0; ic < A_rows; ic += MC)
            {
                int i_min = std::min(ic + MC, A_rows);

                int i = ic;
                for (; i <= i_min - 4; i += 4)
                {
                    const float *b_pack_ptr = packed_B;

                    const float *a_row_ptr_0 = &a_ptr[(i + 0) * A_cols];
                    const float *a_row_ptr_1 = &a_ptr[(i + 1) * A_cols];
                    const float *a_row_ptr_2 = &a_ptr[(i + 2) * A_cols];
                    const float *a_row_ptr_3 = &a_ptr[(i + 3) * A_cols];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 acc0 = _mm256_loadu_ps(&c_ptr[(i + 0) * B_cols + j]);
                        __m256 acc1 = _mm256_loadu_ps(&c_ptr[(i + 1) * B_cols + j]);
                        __m256 acc2 = _mm256_loadu_ps(&c_ptr[(i + 2) * B_cols + j]);
                        __m256 acc3 = _mm256_loadu_ps(&c_ptr[(i + 3) * B_cols + j]);

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

                        _mm256_storeu_ps(&c_ptr[(i + 0) * B_cols + j], acc0);
                        _mm256_storeu_ps(&c_ptr[(i + 1) * B_cols + j], acc1);
                        _mm256_storeu_ps(&c_ptr[(i + 2) * B_cols + j], acc2);
                        _mm256_storeu_ps(&c_ptr[(i + 3) * B_cols + j], acc3);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum0 += a_row_ptr_0[k] * b_ptr[k * B_cols + j];
                            sum1 += a_row_ptr_1[k] * b_ptr[k * B_cols + j];
                            sum2 += a_row_ptr_2[k] * b_ptr[k * B_cols + j];
                            sum3 += a_row_ptr_3[k] * b_ptr[k * B_cols + j];
                        }
                        c_ptr[(i + 0) * B_cols + j] += sum0;
                        c_ptr[(i + 1) * B_cols + j] += sum1;
                        c_ptr[(i + 2) * B_cols + j] += sum2;
                        c_ptr[(i + 3) * B_cols + j] += sum3;
                    }
                }

                for (; i < i_min; ++i)
                {
                    const float *a_row_ptr = &a_ptr[i * A_cols];
                    const float *b_pack_ptr = packed_B;
                    float *c_row_ptr = &c_ptr[i * B_cols];

                    int j = jc;
                    for (; j <= j_min - 8; j += 8)
                    {
                        __m256 v_c = _mm256_loadu_ps(&c_row_ptr[j]);
                        for (int k = kc; k < k_min; ++k)
                        {
                            __m256 v_a = _mm256_set1_ps(a_row_ptr[k]);
                            __m256 v_b = _mm256_load_ps(b_pack_ptr);
                            b_pack_ptr += 8;
                            v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                        }
                        _mm256_storeu_ps(&c_row_ptr[j], v_c);
                    }

                    for (; j < j_min; ++j)
                    {
                        float sum = 0.0f;
                        for (int k = kc; k < k_min; ++k)
                        {
                            sum += a_row_ptr[k] * b_ptr[k * B_cols + j];
                        }
                        c_row_ptr[j] += sum;
                    }
                }
            }
        }
    }
}