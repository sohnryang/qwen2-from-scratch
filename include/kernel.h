#pragma once

#include "tensor.h"

#include <cstddef>
#include <optional>

#include <cuda_bf16.h>

__global__ void gemm(__nv_bfloat16 *__restrict__ out,
                     const __nv_bfloat16 *__restrict__ in_a,
                     const __nv_bfloat16 *__restrict__ in_b,
                     const __nv_bfloat16 *__restrict__ bias,
                     __nv_bfloat16 scale, std::size_t m, std::size_t n,
                     std::size_t k);

__global__ void gemm_transposed(__nv_bfloat16 *__restrict__ out,
                                const __nv_bfloat16 *__restrict__ in_a,
                                const __nv_bfloat16 *__restrict__ in_b,
                                const __nv_bfloat16 *__restrict__ bias,
                                __nv_bfloat16 scale, std::size_t m,
                                std::size_t n, std::size_t k);

__global__ void dense(__nv_bfloat16 *__restrict__ out,
                      const __nv_bfloat16 *__restrict__ x,
                      const __nv_bfloat16 *__restrict__ weight,
                      const __nv_bfloat16 *__restrict__ bias, std::size_t n,
                      std::size_t in_features, std::size_t out_features);

__global__ void gemv_transposed(__nv_bfloat16 *__restrict__ out,
                                const __nv_bfloat16 *__restrict__ mat,
                                const __nv_bfloat16 *__restrict__ vec,
                                const __nv_bfloat16 *__restrict__ bias,
                                std::size_t m, std::size_t n,
                                bool use_activation);

void launch_gemm(Tensor<__nv_bfloat16> &out, const Tensor<__nv_bfloat16> &in_a,
                 const Tensor<__nv_bfloat16> &in_b,
                 const Tensor<__nv_bfloat16> &bias, __nv_bfloat16 scale,
                 bool transpose_second = false);

void launch_gemv(Tensor<__nv_bfloat16> &out, const Tensor<__nv_bfloat16> &mat,
                 const Tensor<__nv_bfloat16> &vec,
                 const std::optional<Tensor<__nv_bfloat16>> &bias,
                 int block_dim_x, int block_dim_y, bool use_activation);

template <typename T>
__global__ void square_sum_reduce(float *__restrict__ out,
                                  const T *__restrict__ x, std::size_t n);

float launch_square_sum_reduce(const Tensor<__nv_bfloat16> &x);

__global__ void elementwise_product(__nv_bfloat16 *__restrict__ out,
                                    const __nv_bfloat16 *__restrict__ x,
                                    const __nv_bfloat16 *__restrict__ y,
                                    float scale, std::size_t n);

__global__ void elementwise_add(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                                const __nv_bfloat16 *y, std::size_t n);

__global__ void softmax(float *out, const float *x, std::size_t batches,
                        std::size_t n);

__global__ void softmax(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                        std::size_t batches, std::size_t n);

void launch_softmax(Tensor<__nv_bfloat16> &out, const Tensor<__nv_bfloat16> &x);

__global__ void grouped_query_attention_scores(
    float *__restrict__ out, const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k, std::size_t sequence_length_q,
    std::size_t sequence_length_kv, std::size_t dimension, std::size_t kv_heads,
    std::size_t groups);

__global__ void grouped_query_attention_scores_masked(
    float *__restrict__ out, const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k, std::size_t sequence_length_q,
    std::size_t sequence_length_kv, std::size_t dimension, std::size_t kv_heads,
    std::size_t groups);

__global__ void grouped_query_attention_output(
    __nv_bfloat16 *__restrict__ out, const float *__restrict__ p,
    const __nv_bfloat16 *__restrict__ v, std::size_t sequence_length_q,
    std::size_t sequence_length_kv, std::size_t dimension, std::size_t kv_heads,
    std::size_t groups);

__global__ void precompute_rope_bases(float *__restrict__ cos_basis_out,
                                      float *__restrict__ sin_basis_out,
                                      int base, std::size_t max_sequence_length,
                                      std::size_t half_dimension);

__global__ void rope(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                     const float *__restrict__ cos_basis,
                     const float *__restrict__ sin_basis, std::size_t offset,
                     std::size_t sequence_length, std::size_t heads,
                     std::size_t half_dimension);

__global__ void
lookup_embeddings(__nv_bfloat16 *__restrict__ out,
                  const int *__restrict__ input_ids,
                  const __nv_bfloat16 *__restrict__ embedding_table,
                  std::size_t sequence_length, std::size_t dimension);

__global__ void argmax_first(const __nv_bfloat16 *__restrict__ logits,
                             float *__restrict__ block_max_vals,
                             int *__restrict__ block_max_indices,
                             std::size_t vocab_size);

__global__ void argmax_reduce(const float *__restrict__ in_vals,
                              const int *__restrict__ in_indices,
                              float *__restrict__ out_vals,
                              int *__restrict__ out_indices,
                              std::size_t blocks_in);

__global__ void rmsnorm(__nv_bfloat16 *__restrict__ out,
                        const __nv_bfloat16 *__restrict__ x,
                        const __nv_bfloat16 *__restrict__ weight,
                        std::size_t batches, std::size_t n, __nv_bfloat16 eps);

__global__ void update_step(int *__restrict__ valid_tokens_ptr,
                            int *__restrict__ tokens_buffer,
                            const int *__restrict__ out_buffer,
                            std::size_t max_sequence_length);
