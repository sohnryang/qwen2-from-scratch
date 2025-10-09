#include "kernel.h"
#include "tensor.h"

#include <stdexcept>

void launch_gemm(Tensor &out, const Tensor &in_a, const Tensor &in_b,
                 const Tensor &bias) {
  if (in_a.dimensions != 2 || in_b.dimensions != 2 || out.dimensions != 2)
    throw std::runtime_error("invalid dimension");
  const auto m = in_a.shape[0];
  const auto k = in_a.shape[1];
  if (k != in_b.shape[0])
    throw std::runtime_error("incompatible dimension");
  const auto n = in_b.shape[1];
  if (m != out.shape[0] || n != out.shape[1])
    throw std::runtime_error("incompatible dimension");

  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks((m + 15) / 16, (n + 15) / 16);
  gemm<<<num_blocks, threads_per_block>>>(out.storage->data, in_a.storage->data,
                                          in_b.storage->data,
                                          bias.storage->data, m, n, k);
}

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
