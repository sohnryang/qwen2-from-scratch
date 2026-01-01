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
  if (row >= m || col >= n)
    return;

  float res = bias ? __bfloat162float(bias[col]) : 0.0f;
  for (int i = 0; i < k; i++)
    res += __bfloat162float(scale) * __bfloat162float(in_a[row * k + i]) *
           __bfloat162float(in_b[i * n + col]);
  out[row * n + col] = __float2bfloat16(res);
}

__global__ void gemm_transposed(__nv_bfloat16 *__restrict__ out,
                                const __nv_bfloat16 *__restrict__ in_a,
                                const __nv_bfloat16 *__restrict__ in_b,
                                const __nv_bfloat16 *__restrict__ bias,
                                __nv_bfloat16 scale, std::size_t m,
                                std::size_t n, std::size_t k) {
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;
  const auto col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n)
    return;

  float res = bias ? __bfloat162float(bias[col]) : 0.0f;
  for (int i = 0; i < k; i++)
    res += __bfloat162float(scale) * __bfloat162float(in_a[row * k + i]) *
           __bfloat162float(in_b[col * k + i]);
  out[row * n + col] = __float2bfloat16(res);
}

__global__ void dense(__nv_bfloat16 *__restrict__ out,
                      const __nv_bfloat16 *__restrict__ x,
                      const __nv_bfloat16 *__restrict__ weight,
                      const __nv_bfloat16 *__restrict__ bias, std::size_t n,
                      std::size_t in_features, std::size_t out_features) {
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;
  const auto col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= n || col >= out_features)
    return;

  auto res = bias ? __bfloat162float(bias[col]) : 0.0f;
  for (int i = 0; i < in_features; i++)
    res += __bfloat162float(x[row * in_features + i]) *
           __bfloat162float(weight[col * in_features + i]);
  out[row * out_features + col] = __float2bfloat16(res / (1.0f + __expf(-res)));
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
  if (idx >= n)
    return;

  out[idx] = __float2bfloat16(scale * __bfloat162float(x[idx]) *
                              __bfloat162float(y[idx]));
}

__global__ void elementwise_add(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                                const __nv_bfloat16 *y, std::size_t n) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  out[idx] =
      __float2bfloat16(__bfloat162float(x[idx]) + __bfloat162float(y[idx]));
}

__global__ void softmax(float *out, const float *x, std::size_t batches,
                        std::size_t n) {
  const auto batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= batches)
    return;

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

__global__ void softmax(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                        std::size_t batches, std::size_t n) {
  const auto batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= batches)
    return;

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
    const __nv_bfloat16 *__restrict__ k, std::size_t sequence_length_q,
    std::size_t sequence_length_kv, std::size_t dimension, std::size_t kv_heads,
    std::size_t groups) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto q_heads = groups * kv_heads;
  const auto q_head = idx / (sequence_length_q * sequence_length_kv);
  const auto k_head = q_head / groups;
  const auto row = idx / sequence_length_kv % sequence_length_q;
  const auto col = idx % sequence_length_kv;
  if (q_head >= q_heads)
    return;

  float score = 0.0f;
  for (int i = 0; i < dimension; i++)
    score += __bfloat162float(
                 q[row * q_heads * dimension + q_head * dimension + i]) *
             __bfloat162float(
                 k[col * kv_heads * dimension + k_head * dimension + i]);
  out[row * q_heads * sequence_length_kv + q_head * sequence_length_kv + col] =
      score / sqrtf(dimension);
}

__global__ void grouped_query_attention_scores_masked(
    float *__restrict__ out, const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k, std::size_t sequence_length_q,
    std::size_t sequence_length_kv, std::size_t dimension, std::size_t kv_heads,
    std::size_t groups) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto q_heads = groups * kv_heads;
  const auto q_head = idx / (sequence_length_q * sequence_length_kv);
  const auto k_head = q_head / groups;
  const auto row = idx / sequence_length_kv % sequence_length_q;
  const auto col = idx % sequence_length_kv;
  if (q_head >= q_heads)
    return;

  float score = 0.0f;
  for (int i = 0; i < dimension; i++)
    score += __bfloat162float(
                 q[row * q_heads * dimension + q_head * dimension + i]) *
             __bfloat162float(
                 k[col * kv_heads * dimension + k_head * dimension + i]);
  out[row * q_heads * sequence_length_kv + q_head * sequence_length_kv + col] =
      score / sqrtf(dimension) +
      (static_cast<int>(col) - static_cast<int>(row) >
               static_cast<int>(sequence_length_kv) -
                   static_cast<int>(sequence_length_q)
           ? -INFINITY
           : 0);
}

__global__ void grouped_query_attention_output(
    __nv_bfloat16 *__restrict__ out, const float *__restrict__ p,
    const __nv_bfloat16 *__restrict__ v, std::size_t sequence_length_q,
    std::size_t sequence_length_kv, std::size_t dimension, std::size_t kv_heads,
    std::size_t groups) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto q_heads = groups * kv_heads;
  const auto q_head = idx / (sequence_length_q * dimension);
  const auto v_head = q_head / groups;
  const auto row = idx / dimension % sequence_length_q;
  const auto col = idx % dimension;
  if (q_head >= q_heads)
    return;

  float o = 0.0f;
  for (int i = 0; i < sequence_length_kv; i++)
    o += p[row * q_heads * sequence_length_kv + q_head * sequence_length_kv +
           i] *
         __bfloat162float(
             v[i * kv_heads * dimension + v_head * dimension + col]);
  out[row * q_heads * dimension + q_head * dimension + col] =
      __float2bfloat16(o);
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

__global__ void rope(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                     const float *__restrict__ cos_basis,
                     const float *__restrict__ sin_basis, std::size_t offset,
                     std::size_t sequence_length, std::size_t heads,
                     std::size_t half_dimension) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto sequence_idx = idx / (heads * half_dimension);
  const auto head = idx / half_dimension % heads;
  const auto vector_idx1 = idx % half_dimension;
  const auto vector_idx2 = vector_idx1 + half_dimension;
  if (sequence_idx >= sequence_length)
    return;

  const auto dimension = 2 * half_dimension;
  const auto x1_idx =
      sequence_idx * heads * dimension + head * dimension + vector_idx1;
  const auto x1 = __bfloat162float(x[x1_idx]);
  const auto x2_idx =
      sequence_idx * heads * dimension + head * dimension + vector_idx2;
  const auto x2 = __bfloat162float(x[x2_idx]);
  const auto cos_basis_elem =
      cos_basis[(sequence_idx + offset) * half_dimension + vector_idx1];
  const auto sin_basis_elem =
      sin_basis[(sequence_idx + offset) * half_dimension + vector_idx1];
  out[x1_idx] = __float2bfloat16(x1 * cos_basis_elem - x2 * sin_basis_elem);
  out[x2_idx] = __float2bfloat16(x2 * cos_basis_elem + x1 * sin_basis_elem);
}

__global__ void
lookup_embeddings(__nv_bfloat16 *__restrict__ out,
                  const int *__restrict__ input_ids,
                  const __nv_bfloat16 *__restrict__ embedding_table,
                  std::size_t sequence_length, std::size_t dimension) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= sequence_length)
    return;

  const auto token_id = input_ids[idx];
  for (int i = 0; i < dimension; i++)
    out[idx * dimension + i] = embedding_table[token_id * dimension + i];
}

__global__ void argmax_first(const __nv_bfloat16 *__restrict__ logits,
                             float *__restrict__ block_max_vals,
                             int *__restrict__ block_max_indices,
                             std::size_t vocab_size) {
  const auto batch = blockIdx.y;
  const auto block = blockIdx.x;

  const auto global_offset = batch * vocab_size;
  const auto idx = block * blockDim.x + threadIdx.x;
  const float val = idx < vocab_size
                        ? __bfloat162float(logits[global_offset + idx])
                        : -INFINITY;

  extern __shared__ unsigned char shared_raw[];
  float *shared_max_val = reinterpret_cast<float *>(shared_raw);
  int *shared_max_idx =
      reinterpret_cast<int *>(shared_raw + blockDim.x * sizeof(float));
  shared_max_val[threadIdx.x] = val;
  shared_max_idx[threadIdx.x] = idx;
  __syncthreads();

  for (std::size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      const float other_val = shared_max_val[threadIdx.x + offset];
      const int other_idx = shared_max_idx[threadIdx.x + offset];
      if (other_val > shared_max_val[threadIdx.x] ||
          (other_val == shared_max_val[threadIdx.x] &&
           other_idx < shared_max_idx[threadIdx.x])) {
        shared_max_val[threadIdx.x] = other_val;
        shared_max_idx[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const auto out_idx = batch * gridDim.x + block;
    block_max_vals[out_idx] = shared_max_val[0];
    block_max_indices[out_idx] = shared_max_idx[0];
  }
}

__global__ void argmax_reduce(const float *__restrict__ in_vals,
                              const int *__restrict__ in_indices,
                              float *__restrict__ out_vals,
                              int *__restrict__ out_indices,
                              std::size_t blocks_in) {
  const auto batch = blockIdx.y;
  const auto block = blockIdx.x;

  const auto base = batch * blocks_in;
  const auto idx = block * blockDim.x + threadIdx.x;

  const bool valid = idx < blocks_in;
  const float val = valid ? in_vals[base + idx] : -INFINITY;
  const int id = valid ? in_indices[base + idx] : 0;

  extern __shared__ unsigned char shared_raw[];
  float *shared_max_val = reinterpret_cast<float *>(shared_raw);
  int *shared_max_idx =
      reinterpret_cast<int *>(shared_raw + blockDim.x * sizeof(float));
  shared_max_val[threadIdx.x] = val;
  shared_max_idx[threadIdx.x] = id;
  __syncthreads();

  for (std::size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      const float other_val = shared_max_val[threadIdx.x + offset];
      const int other_idx = shared_max_idx[threadIdx.x + offset];
      if (other_val > shared_max_val[threadIdx.x] ||
          (other_val == shared_max_val[threadIdx.x] &&
           other_idx < shared_max_idx[threadIdx.x])) {
        shared_max_val[threadIdx.x] = other_val;
        shared_max_idx[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const auto out_idx = batch * gridDim.x + block;
    out_vals[out_idx] = shared_max_val[0];
    out_indices[out_idx] = shared_max_idx[0];
  }
}

__global__ void rmsnorm(__nv_bfloat16 *__restrict__ out,
                        const __nv_bfloat16 *__restrict__ x,
                        const __nv_bfloat16 *__restrict__ weight,
                        std::size_t batches, std::size_t n, __nv_bfloat16 eps) {
  const auto batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= batches)
    return;

  float squared_sum = 0.0f;
  for (int i = 0; i < n; i++) {
    const auto elem = __bfloat162float(x[n * batch + i]);
    squared_sum += elem * elem;
  }
  const auto factor = rsqrtf(squared_sum / n + __bfloat162float(eps));
  for (int i = 0; i < n; i++)
    out[n * batch + i] = __float2bfloat16(__bfloat162float(x[n * batch + i]) *
                                          factor * __bfloat162float(weight[i]));
}

__global__ void step(int *last_token_index, int *is_stopped,
                     std::size_t max_sequence_length) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  if (atomicAdd(is_stopped, 0) == 1)
    return;
  const int next_index = atomicAdd(last_token_index, 1) + 1;
  if (next_index >= static_cast<int>(max_sequence_length))
    atomicExch(is_stopped, 1);
}
