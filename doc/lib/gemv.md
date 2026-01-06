1. naive_gemv:
- Naive 2 loops ij calculates each cells of `y`
2. unroll_j_gemv:
- This implementation utilizes `_mm256_fmadd_ps`
3. unroll_i_j_gemv:
- This hides latency of loading `W`
- A reduction function will be needed to prevent storing sum to temp arrays
- The reduction function `hsum_avx` folds the upper and lower 128-bit lanes together, then shuffles and adds to reduce all 8 floats into 1 result
4. gemv:
- Add `#pragma omp parallel for`. This would require to reformat the `i` loop as `num_blocks`