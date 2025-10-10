#include "kernel.h"
#include "tensor.h"

#include <stdexcept>

__global__ void gemm(__nv_bfloat16 *__restrict__ out,
                     const __nv_bfloat16 *__restrict__ in_a,
                     const __nv_bfloat16 *__restrict__ in_b,
                     const __nv_bfloat16 *__restrict__ bias,
                     __nv_bfloat16 scale, std::size_t m, std::size_t n,
                     std::size_t k) {
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;
  const auto col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    auto res = bias ? bias[col] : __nv_bfloat16{0};
    for (int i = 0; i < k; i++)
      res += scale * in_a[row * k + i] * in_b[i * n + col];
    out[row * n + col] = res;
  }
}

__global__ void gemm_transposed(__nv_bfloat16 *__restrict__ out,
                                const __nv_bfloat16 *__restrict__ in_a,
                                const __nv_bfloat16 *__restrict__ in_b,
                                const __nv_bfloat16 *__restrict__ bias,
                                __nv_bfloat16 scale, std::size_t m,
                                std::size_t n, std::size_t k) {
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;
  const auto col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    auto res = bias ? bias[col] : __nv_bfloat16{0};
    for (int i = 0; i < k; i++)
      res += scale * in_a[row * k + i] * in_b[col * k + i];
    out[row * n + col] = res;
  }
}

__global__ void dense(__nv_bfloat16 *__restrict__ out,
                      const __nv_bfloat16 *__restrict__ x,
                      const __nv_bfloat16 *__restrict__ weight,
                      const __nv_bfloat16 *__restrict__ bias, std::size_t n,
                      std::size_t in_features, std::size_t out_features) {
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;
  const auto col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < n && col < out_features) {
    auto res = bias ? bias[col] : __nv_bfloat16{0};
    for (int i = 0; i < in_features; i++)
      res += x[row * in_features + i] * weight[col * in_features + i];
    const auto res_f32 = __bfloat162float(res);
    out[row * out_features + col] =
        __float2bfloat16(res_f32 / (1.0f + __expf(-res_f32)));
  }
}

void launch_gemm(Tensor &out, const Tensor &in_a, const Tensor &in_b,
                 const Tensor &bias, __nv_bfloat16 scale,
                 bool transpose_second) {
  if (in_a.dimensions != 2 || in_b.dimensions != 2 || out.dimensions != 2)
    throw std::runtime_error("invalid dimension");
  const auto m = in_a.shape[0];
  const auto k = in_a.shape[1];
  const auto n = transpose_second ? in_b.shape[0] : in_b.shape[1];
  if (k != in_b.shape[transpose_second ? 1 : 0])
    throw std::runtime_error("incompatible dimension");
  if (m != out.shape[0] || n != out.shape[1])
    throw std::runtime_error("incompatible dimension");

  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks((m + 15) / 16, (n + 15) / 16);
  if (transpose_second) {
    gemm_transposed<<<num_blocks, threads_per_block>>>(
        out.storage->data, in_a.storage->data, in_b.storage->data,
        bias.storage->data, scale, m, n, k);
  } else {
    gemm<<<num_blocks, threads_per_block>>>(
        out.storage->data, in_a.storage->data, in_b.storage->data,
        bias.storage->data, scale, m, n, k);
  }
}
