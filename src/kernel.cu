#include "kernel.h"

__global__ void gemm(__nv_bfloat16 *__restrict__ out,
                     const __nv_bfloat16 *__restrict__ in_a,
                     const __nv_bfloat16 *__restrict__ in_b,
                     const __nv_bfloat16 *__restrict__ bias, std::size_t m,
                     std::size_t n, std::size_t k) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < m && y < n) {
    auto res = bias ? bias[x * n + y] : __nv_bfloat16{0};
    for (int i = 0; i < k; i++)
      res += in_a[x * k + i] * in_b[i * n + y];
    out[x * n + y] = res;
  }
}
