#include "cuda_utils.h"
#include "kernel.h"
#include "tensor.h"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

__global__ void gemm(__nv_bfloat16 *__restrict__ out,
                     const __nv_bfloat16 *__restrict__ in_a,
                     const __nv_bfloat16 *__restrict__ in_b,
                     const __nv_bfloat16 *__restrict__ bias,
                     __nv_bfloat16 scale, std::size_t m, std::size_t n,
                     std::size_t k) {
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;
  const auto col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    float res = bias ? __bfloat162float(bias[col]) : 0.0f;
    for (int i = 0; i < k; i++)
      res += __bfloat162float(scale) * __bfloat162float(in_a[row * k + i]) *
             __bfloat162float(in_b[i * n + col]);
    out[row * n + col] = __float2bfloat16(res);
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
    float res = bias ? __bfloat162float(bias[col]) : 0.0f;
    for (int i = 0; i < k; i++)
      res += __bfloat162float(scale) * __bfloat162float(in_a[row * k + i]) *
             __bfloat162float(in_b[col * k + i]);
    out[row * n + col] = __float2bfloat16(res);
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
    auto res = bias ? __bfloat162float(bias[col]) : 0.0f;
    for (int i = 0; i < in_features; i++)
      res += __bfloat162float(x[row * in_features + i]) *
             __bfloat162float(weight[col * in_features + i]);
    out[row * out_features + col] =
        __float2bfloat16(res / (1.0f + __expf(-res)));
  }
}

void launch_gemm(Tensor<__nv_bfloat16> &out, const Tensor<__nv_bfloat16> &in_a,
                 const Tensor<__nv_bfloat16> &in_b,
                 const Tensor<__nv_bfloat16> &bias, __nv_bfloat16 scale,
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
  const dim3 num_blocks(ceil_div(m, threads_per_block.x),
                        ceil_div(n, threads_per_block.y));
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

template <typename T>
__global__ void square_sum_reduce(float *__restrict__ out,
                                  const T *__restrict__ x, std::size_t n) {
  extern __shared__ float sdata[];
  const auto tid = threadIdx.x;
  const auto idx = blockIdx.x * blockDim.x + tid;
  sdata[tid] = idx < n ? [&] {
    if constexpr (std::is_same<T, float>::value)
      return x[idx];
    else {
      float x_value;
      x_value = __bfloat162float(x[idx]);
      return x_value * x_value;
    }
  }()
                       : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    out[blockIdx.x] = sdata[0];
}

template __global__ void square_sum_reduce<float>(float *__restrict__ out,
                                                  const float *__restrict__ x,
                                                  std::size_t n);
template __global__ void
square_sum_reduce<__nv_bfloat16>(float *__restrict__ out,
                                 const __nv_bfloat16 *__restrict__ x,
                                 std::size_t n);

float launch_square_sum_reduce(const Tensor<__nv_bfloat16> &x) {
  const dim3 threads_per_block(1024);
  dim3 num_blocks(ceil_div(x.storage->elems, threads_per_block.x));
  float *out_arr;
  CHECK_CUDA(cudaMalloc(&out_arr, num_blocks.x * sizeof(float)));
  square_sum_reduce<<<num_blocks, threads_per_block,
                      threads_per_block.x * sizeof(float)>>>(
      out_arr, x.storage->data, x.storage->elems);
  if (num_blocks.x == 1) {
    float res;
    CHECK_CUDA(
        cudaMemcpy(&res, out_arr, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(out_arr));
    return res;
  }

  float *intermediate_arr;
  CHECK_CUDA(cudaMalloc(&intermediate_arr, num_blocks.x * sizeof(float)));
  while (num_blocks.x > 1) {
    CHECK_CUDA(cudaMemcpy(intermediate_arr, out_arr,
                          num_blocks.x * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    square_sum_reduce<<<num_blocks, threads_per_block,
                        threads_per_block.x * sizeof(float)>>>(
        out_arr, intermediate_arr, num_blocks.x);
    num_blocks.x = ceil_div(num_blocks.x, threads_per_block.x);
  }
  CHECK_CUDA(cudaFree(intermediate_arr));

  float res;
  CHECK_CUDA(cudaMemcpy(&res, out_arr, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(out_arr));
  return res;
}

__global__ void elementwise_product(__nv_bfloat16 *__restrict__ out,
                                    const __nv_bfloat16 *__restrict__ x,
                                    const __nv_bfloat16 *__restrict__ y,
                                    float scale, std::size_t n) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = __float2bfloat16(scale * __bfloat162float(x[idx]) *
                                __bfloat162float(y[idx]));
}

__global__ void elementwise_add(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                                const __nv_bfloat16 *y, std::size_t n) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] =
        __float2bfloat16(__bfloat162float(x[idx]) + __bfloat162float(y[idx]));
}

__global__ void softmax(float *out, const float *x, std::size_t batches,
                        std::size_t n) {
  const auto batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < batches) {
    float x_max = -INFINITY;
    float norm = 0.0f;
    for (int i = 0; i < n; i++) {
      const auto x_cur = x[batch * n + i];
      if (x_cur > x_max) {
        norm = norm * __expf(x_max - x_cur);
        x_max = x_cur;
      }
      norm += __expf(x_cur - x_max);
    }
    for (int i = 0; i < n; i++)
      out[batch * n + i] = __expf(x[batch * n + i] - x_max) / norm;
  }
}

__global__ void softmax(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                        std::size_t batches, std::size_t n) {
  const auto batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < batches) {
    __nv_bfloat16 x_max = -INFINITY;
    float norm = 0.0f;
    for (int i = 0; i < n; i++) {
      const auto x_cur = x[batch * n + i];
      if (x_cur > x_max) {
        norm = norm * __expf(__bfloat162float(x_max - x_cur));
        x_max = x_cur;
      }
      norm += __expf(x_cur - x_max);
    }
    for (int i = 0; i < n; i++)
      out[batch * n + i] = __float2bfloat16(
          __expf(__bfloat162float(x[batch * n + i] - x_max)) / norm);
  }
}

void launch_softmax(Tensor<__nv_bfloat16> &out,
                    const Tensor<__nv_bfloat16> &x) {
  if (out.storage->elems != x.storage->elems)
    throw std::runtime_error("incompatible dimension");
  if (x.dimensions < 1)
    throw std::runtime_error("invalid dimension");

  const auto n = x.shape[x.dimensions - 1];
  const auto batches = x.storage->elems / n;

  const dim3 threads_per_block(256);
  const dim3 num_blocks(ceil_div(batches, threads_per_block.x));
  softmax<<<num_blocks, threads_per_block>>>(out.storage->data, x.storage->data,
                                             batches, n);
}

__global__ void grouped_query_attention_scores(
    float *__restrict__ out, const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k, std::size_t batches,
    std::size_t sequence_length_q, std::size_t sequence_length_kv,
    std::size_t dimension, std::size_t kv_heads, std::size_t groups) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto q_heads = groups * kv_heads;
  const auto batch = idx / (q_heads * sequence_length_q * sequence_length_kv);
  const auto q_head = idx / (sequence_length_q * sequence_length_kv) % q_heads;
  const auto k_head = q_head / groups;
  const auto row = idx / sequence_length_kv % sequence_length_q;
  const auto col = idx % sequence_length_kv;
  if (batch < batches) {
    float score = 0.0f;
    for (int i = 0; i < dimension; i++)
      score += __bfloat162float(
                   q[batch * sequence_length_q * q_heads * dimension +
                     row * q_heads * dimension + q_head * dimension + i]) *
               __bfloat162float(
                   k[batch * sequence_length_kv * kv_heads * dimension +
                     col * kv_heads * dimension + k_head * dimension + i]);
    out[batch * sequence_length_q * q_heads * sequence_length_kv +
        row * q_heads * sequence_length_kv + q_head * sequence_length_kv +
        col] = score / sqrtf(dimension);
  }
}

__global__ void grouped_query_attention_scores_masked(
    float *__restrict__ out, const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k, std::size_t batches,
    std::size_t sequence_length_q, std::size_t sequence_length_kv,
    std::size_t dimension, std::size_t kv_heads, std::size_t groups) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto q_heads = groups * kv_heads;
  const auto batch = idx / (q_heads * sequence_length_q * sequence_length_kv);
  const auto q_head = idx / (sequence_length_q * sequence_length_kv) % q_heads;
  const auto k_head = q_head / groups;
  const auto row = idx / sequence_length_kv % sequence_length_q;
  const auto col = idx % sequence_length_kv;
  if (batch < batches) {
    float score = 0.0f;
    for (int i = 0; i < dimension; i++)
      score += __bfloat162float(
                   q[batch * sequence_length_q * q_heads * dimension +
                     row * q_heads * dimension + q_head * dimension + i]) *
               __bfloat162float(
                   k[batch * sequence_length_kv * kv_heads * dimension +
                     col * kv_heads * dimension + k_head * dimension + i]);
    out[batch * sequence_length_q * q_heads * sequence_length_kv +
        row * q_heads * sequence_length_kv + q_head * sequence_length_kv +
        col] = score / sqrtf(dimension) +
               (static_cast<int>(col) - static_cast<int>(row) >
                        static_cast<int>(sequence_length_kv) -
                            static_cast<int>(sequence_length_q)
                    ? -INFINITY
                    : 0);
  }
}

__global__ void grouped_query_attention_output(
    __nv_bfloat16 *__restrict__ out, const float *__restrict__ p,
    const __nv_bfloat16 *__restrict__ v, std::size_t batches,
    std::size_t sequence_length_q, std::size_t sequence_length_kv,
    std::size_t dimension, std::size_t kv_heads, std::size_t groups) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto q_heads = groups * kv_heads;
  const auto batch = idx / (q_heads * sequence_length_q * dimension);
  const auto q_head = idx / (sequence_length_q * dimension) % q_heads;
  const auto v_head = q_head / groups;
  const auto row = idx / dimension % sequence_length_q;
  const auto col = idx % dimension;
  if (batch < batches) {
    float o = 0.0f;
    for (int i = 0; i < sequence_length_kv; i++)
      o += p[batch * sequence_length_q * q_heads * sequence_length_kv +
             row * q_heads * sequence_length_kv + q_head * sequence_length_kv +
             i] *
           __bfloat162float(
               v[batch * sequence_length_kv * kv_heads * dimension +
                 i * kv_heads * dimension + v_head * dimension + col]);
    out[batch * sequence_length_q * q_heads * dimension +
        row * q_heads * dimension + q_head * dimension + col] =
        __float2bfloat16(o);
  }
}

__global__ void precompute_rope_bases(float *__restrict__ cos_basis_out,
                                      float *__restrict__ sin_basis_out,
                                      int base, std::size_t max_sequence_length,
                                      std::size_t half_dimension) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto row = idx / half_dimension;
  const auto col = idx % half_dimension;
  if (row >= max_sequence_length)
    return;

  const auto freq = powf(base, -static_cast<float>(col) / half_dimension);
  cos_basis_out[row * half_dimension + col] = cosf(row * freq);
  sin_basis_out[row * half_dimension + col] = sinf(row * freq);
}

__global__ void rope(__nv_bfloat16 *__restrict__ out,
                     const __nv_bfloat16 *__restrict__ x,
                     const float *__restrict__ cos_basis,
                     const float *__restrict__ sin_basis, std::size_t batches,
                     std::size_t sequence_length, std::size_t heads,
                     std::size_t half_dimension) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto batch = idx / (sequence_length * heads * half_dimension);
  const auto sequence_idx = idx / (heads * half_dimension) % sequence_length;
  const auto head = idx / half_dimension % heads;
  const auto vector_idx1 = idx % half_dimension;
  const auto vector_idx2 = vector_idx1 + half_dimension;
  if (batch >= batches)
    return;

  const auto dimension = 2 * half_dimension;
  const auto x1_idx = batch * sequence_length * heads * dimension +
                      sequence_idx * heads * dimension + head * dimension +
                      vector_idx1;
  const auto x1 = __bfloat162float(x[x1_idx]);
  const auto x2_idx = batch * sequence_length * heads * dimension +
                      sequence_idx * heads * dimension + head * dimension +
                      vector_idx2;
  const auto x2 = __bfloat162float(x[x2_idx]);
  const auto cos_basis_elem =
      cos_basis[sequence_idx * half_dimension + vector_idx1];
  const auto sin_basis_elem =
      sin_basis[sequence_idx * half_dimension + vector_idx1];
  out[x1_idx] = __float2bfloat16(x1 * cos_basis_elem - x2 * sin_basis_elem);
  out[x2_idx] = __float2bfloat16(x2 * cos_basis_elem + x1 * sin_basis_elem);
}
