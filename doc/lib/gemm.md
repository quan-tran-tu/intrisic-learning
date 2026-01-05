- Currently each implementations would need to fill `C` with zeros first, except register_blocking_gemm (since the registers are initialized with zeros to store the future values of `C`). But since `std::fill` takes O(M * N), which is a lot less than O(M * N * K), `std::fill` will be kept for now

1. naive_gemm:
- Naive 3 loops ijk calculates each cells of `C`
2. ikj_gemm:
- Naive ijk loops accesses `b_ptr` by jumping `B_cols`, which is inefficient
- Changing to ikj loops is better since `a` can be reused and `B` can now be accessed contiguously
- `j` will be processed in steps of 8 element
3. ikj_broadcast_a_gemm:
- `a` can be broadcasted
- This approach is "memory bound" on `C`
4. register_blocking_gemm:
- The current code is constantly reading and writing to `C` for every single `k` step
- So the idea would be to compute a 4x8 tile of C at once (4 YMM registers), `i` loop is then unrolled by 4
- Only write back to `C` memory after `k` loop is totally finished