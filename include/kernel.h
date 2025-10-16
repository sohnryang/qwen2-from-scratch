#pragma once

#include "tensor.h"

#include <cstddef>
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

void launch_gemm(Tensor<__nv_bfloat16> &out, const Tensor<__nv_bfloat16> &in_a,
                 const Tensor<__nv_bfloat16> &in_b,
                 const Tensor<__nv_bfloat16> &bias, __nv_bfloat16 scale,
                 bool transpose_second = false);

template <typename T>
__global__ void square_sum_reduce(float *__restrict__ out,
                                  const T *__restrict__ x, std::size_t n);

float launch_square_sum_reduce(const Tensor<__nv_bfloat16> &x);

__global__ void elementwise_product(__nv_bfloat16 *__restrict__ out,
                                    const __nv_bfloat16 *__restrict__ x,
                                    const __nv_bfloat16 *__restrict__ y,
                                    float scale, std::size_t n);

__global__ void softmax(float *out, const float *x, std::size_t batches,
                        std::size_t n);

__global__ void softmax(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                        std::size_t batches, std::size_t n);

void launch_softmax(Tensor<__nv_bfloat16> &out, const Tensor<__nv_bfloat16> &x);

__global__ void grouped_query_attention_scores(
    float *__restrict__ out, const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k, std::size_t batches,
    std::size_t sequence_length_q, std::size_t sequence_length_kv,
    std::size_t dimension, std::size_t kv_heads, std::size_t groups);

__global__ void grouped_query_attention_scores_masked(
    float *__restrict__ out, const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k, std::size_t batches,
    std::size_t sequence_length_q, std::size_t sequence_length_kv,
    std::size_t dimension, std::size_t kv_heads, std::size_t groups);

__global__ void grouped_query_attention_output(
    __nv_bfloat16 *__restrict__ out, const float *__restrict__ p,
    const __nv_bfloat16 *__restrict__ v, std::size_t batches,
    std::size_t sequence_length_q, std::size_t sequence_length_kv,
    std::size_t dimension, std::size_t kv_heads, std::size_t groups);
