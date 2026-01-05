#include "core/tensor.h"
#include "lib/gemm.h"

#include <immintrin.h>

void naive_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C) {
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float* a_ptr = A.data().get();
    const float* b_ptr = B.data().get();
    float* c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols) {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    for (int i = 0; i < A_rows; ++i) {
        const float* a_row_ptr = &a_ptr[i * A_cols];
        float* c_row_ptr = &c_ptr[i * B_cols];
        for (int j = 0; j < B_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; ++k) {
                sum += a_row_ptr[k] * b_ptr[k * B_cols + j];
            }
            c_row_ptr[j] = sum;
        }
    }
}

void ikj_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C) {
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float* a_ptr = A.data().get();
    const float* b_ptr = B.data().get();
    float* c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols) {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    for (int i = 0; i < A_rows; ++i) {
        for (int k = 0; k < A_cols; ++k) {
            float a = a_ptr[i * A_cols + k];
            const float* b_row_ptr = &b_ptr[k * B_cols];
            float* c_row_ptr = &c_ptr[i * B_cols];
            for (int j = 0; j < B_cols; ++j) {
                c_row_ptr[j] += a * b_row_ptr[j];
            }
        }
    }
}

void ikj_broadcast_a_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C) {
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float* a_ptr = A.data().get();
    const float* b_ptr = B.data().get();
    float* c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols) {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    for (int i = 0; i < A_rows; ++i) {
        const float* a_row_ptr = &a_ptr[i * A_cols];
        float* c_row_ptr = &c_ptr[i * B_cols];
        for (int k = 0; k < A_cols; ++k) {
            __m256 v_a = _mm256_set1_ps(a_row_ptr[k]);
            const float* b_row_ptr = &b_ptr[k * B_cols];
            int j = 0;
            for (; j <= B_cols - 8; j += 8) {
                __m256 v_b = _mm256_loadu_ps(&b_row_ptr[j]);
                __m256 v_c = _mm256_loadu_ps(&c_row_ptr[j]);
                v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                _mm256_storeu_ps(&c_row_ptr[j], v_c);
            }
            for (; j < B_cols; ++j) {
                c_row_ptr[j] += a_row_ptr[k] * b_ptr[k * B_cols + j];
            }
        }
    }
}

void loop_unrolling_gemm(const Tensor2D &A, const Tensor2D &B, Tensor2D &C) {
    int A_rows = A.rows();
    int B_cols = B.cols();
    int A_cols = A.cols();

    const float* a_ptr = A.data().get();
    const float* b_ptr = B.data().get();
    float* c_ptr = C.data().get();

    // shape check
    if (C.rows() != A_rows || C.cols() != B_cols) {
        throw std::invalid_argument("C rows and cols must match A rows and B cols, respectively");
    }

    // std::fill(c_ptr, c_ptr + (A_rows * B_cols), 0.0f);

    int i = 0;
    for (; i <= A_rows - 4; i += 4) {
        const float* a_row_ptr_0 = &a_ptr[(i + 0) * A_cols];
        const float* a_row_ptr_1 = &a_ptr[(i + 1) * A_cols];
        const float* a_row_ptr_2 = &a_ptr[(i + 2) * A_cols];
        const float* a_row_ptr_3 = &a_ptr[(i + 3) * A_cols];

        int j = 0;
        for (; j <= B_cols - 8; j += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int k = 0; k < A_cols; ++k) {
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

        for (; j < B_cols; ++j) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            for (int k = 0; k < A_cols; ++k) {
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

    for (; i < A_rows; ++i) {
        const float* a_row_ptr = &a_ptr[i * A_cols];
        float* c_row_ptr = &c_ptr[i * B_cols];
        for (int j = 0; j < B_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; ++k) {
                sum += a_row_ptr[k] * b_ptr[k * B_cols + j];
            }
            c_row_ptr[j] = sum;
        }
    }
}