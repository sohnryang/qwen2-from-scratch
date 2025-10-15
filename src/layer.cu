#include "cuda_utils.h"
#include "kernel.h"
#include "layer.h"
#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>

#include <cuda_bf16.h>

Tensor<__nv_bfloat16>
Dense::operator()(const Tensor<__nv_bfloat16> &input,
                  std::shared_ptr<Storage<__nv_bfloat16>> out_storage) {
  assert(input.shape[input.dimensions - 1] == _in_features &&
         "invalid input dimension");
  const auto batches = input.storage->elems / _in_features;
  const auto out_elems = batches * _out_features;
  if (out_storage)
    assert(out_storage->elems ==
               input.storage->elems / _in_features * _out_features &&
           "invalid output storage size");

  out_storage = out_storage
                    ? out_storage
                    : std::make_shared<Storage<__nv_bfloat16>>(out_elems);
  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks(ceil_div(batches, threads_per_block.x),
                        ceil_div(_out_features, threads_per_block.y));
  if (_use_activation)
    dense<<<num_blocks, threads_per_block>>>(
        out_storage->data, input.storage->data, _weight.storage->data,
        _bias ? _bias->storage->data : nullptr, batches, _in_features,
        _out_features);
  else
    gemm_transposed<<<num_blocks, threads_per_block>>>(
        out_storage->data, input.storage->data, _weight.storage->data,
        _bias ? _bias->storage->data : nullptr, 1.0f, batches, _out_features,
        _in_features);

  Tensor<__nv_bfloat16> res = {.dimensions = input.dimensions,
                               .storage = out_storage};
  std::copy_n(input.shape.begin(), input.dimensions - 1, res.shape.begin());
  res.shape[input.dimensions - 1] = _out_features;
  return res;
}

Tensor<__nv_bfloat16>
RMSNorm::operator()(const Tensor<__nv_bfloat16> &input,
                    std::shared_ptr<Storage<__nv_bfloat16>> out_storage) {
  assert(input.shape[input.dimensions - 1] == _dimensions &&
         "invalid input dimension");
  const auto batches = input.storage->elems / _dimensions;
  if (out_storage)
    assert(out_storage->elems == input.storage->elems &&
           "invalid output storage size");

  out_storage =
      out_storage
          ? out_storage
          : std::make_shared<Storage<__nv_bfloat16>>(input.storage->elems);

  Tensor<__nv_bfloat16> reshaped =
      input.reshape({-1, static_cast<int>(_dimensions)});
  const dim3 threads_per_block(1024);
  const dim3 num_blocks(ceil_div(_dimensions, threads_per_block.x));
  float *out_arr, *intermediate_arr;
  CHECK_CUDA(cudaMalloc(&out_arr, num_blocks.x * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&intermediate_arr, num_blocks.x * sizeof(float)));
  for (int batch = 0; batch < reshaped.shape[0]; batch++) {
    square_sum_reduce<<<num_blocks, threads_per_block,
                        threads_per_block.x * sizeof(float)>>>(
        out_arr, reshaped.storage->data + batch * _dimensions, _dimensions);
    if (num_blocks.x != 1) {
      dim3 num_blocks_reduced(ceil_div(_dimensions, threads_per_block.x));
      while (num_blocks_reduced.x > 1) {
        CHECK_CUDA(cudaMemcpy(intermediate_arr, out_arr,
                              num_blocks_reduced.x * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        square_sum_reduce<<<num_blocks_reduced, threads_per_block,
                            threads_per_block.x * sizeof(float)>>>(
            out_arr, intermediate_arr, num_blocks_reduced.x);
        num_blocks_reduced.x =
            ceil_div(num_blocks_reduced.x, threads_per_block.x);
      }
    }
    float res;
    CHECK_CUDA(
        cudaMemcpy(&res, out_arr, sizeof(float), cudaMemcpyDeviceToHost));

    elementwise_product<<<num_blocks, threads_per_block>>>(
        out_storage->data + batch * _dimensions,
        reshaped.storage->data + batch * _dimensions, _weight.storage->data,
        rsqrtf(res / _dimensions + _epsilon), _dimensions);
  }
  CHECK_CUDA(cudaFree(intermediate_arr));
  CHECK_CUDA(cudaFree(out_arr));

  return {.shape = input.shape,
          .dimensions = input.dimensions,
          .storage = out_storage};
}

Tensor<__nv_bfloat16> GroupedQueryAttention::operator()(
    const Tensor<__nv_bfloat16> &input_q, const Tensor<__nv_bfloat16> &input_k,
    const Tensor<__nv_bfloat16> &input_v,
    std::optional<std::reference_wrapper<const Tensor<__nv_bfloat16>>>
        input_mask,
    std::shared_ptr<Storage<__nv_bfloat16>> q_proj_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> k_proj_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> v_proj_out_storage,
    std::shared_ptr<Storage<float>> scores_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> attention_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> o_proj_out_storage) {
  const auto q_proj = _q_layer(input_q, q_proj_out_storage);
  const auto k_proj = _k_layer(input_k, k_proj_out_storage);
  const auto v_proj = _v_layer(input_v, v_proj_out_storage);

  assert(q_proj.dimensions == k_proj.dimensions &&
         k_proj.dimensions == v_proj.dimensions &&
         "QKV dimension should match");
  assert(k_proj.shape[k_proj.dimensions - 2] ==
             v_proj.shape[v_proj.dimensions - 2] &&
         "KV sequence length should match");
  const auto sequence_length_q = q_proj.shape[q_proj.dimensions - 2];
  const auto sequence_length_kv = k_proj.shape[q_proj.dimensions - 2];
  assert(k_proj.storage->elems == v_proj.storage->elems &&
         q_proj.storage->elems / sequence_length_q ==
             k_proj.storage->elems / sequence_length_kv * _groups &&
         "QKV element count should match");
  const auto dimension = _k_layer.out_features() / _kv_heads;
  const auto batches =
      k_proj.storage->elems / _k_layer.out_features() / sequence_length_kv;
  if (!scores_out_storage)
    scores_out_storage = std::make_shared<Storage<float>>(
        batches * _kv_heads * _groups * sequence_length_q * sequence_length_kv);
  const dim3 threads_per_block(1024);
  {
    const dim3 num_blocks(
        ceil_div(scores_out_storage->elems, threads_per_block.x));
    grouped_query_attention_scores<<<num_blocks, threads_per_block>>>(
        scores_out_storage->data, q_proj.storage->data, k_proj.storage->data,
        input_mask ? input_mask->get().storage->data : nullptr, batches,
        sequence_length_q, sequence_length_kv, dimension, _kv_heads, _groups);
  }
  {
    const dim3 num_blocks(ceil_div(
        scores_out_storage->elems / sequence_length_kv, threads_per_block.x));
    softmax<<<num_blocks, threads_per_block>>>(
        scores_out_storage->data, scores_out_storage->data,
        scores_out_storage->elems / sequence_length_kv, sequence_length_kv);
  }
  if (!attention_out_storage)
    attention_out_storage = std::make_shared<Storage<__nv_bfloat16>>(
        batches * _kv_heads * _groups * sequence_length_q * dimension);
  {
    const dim3 num_blocks(ceil_div(attention_out_storage->elems, 1024));
    grouped_query_attention_output<<<num_blocks, threads_per_block>>>(
        attention_out_storage->data, scores_out_storage->data,
        v_proj.storage->data, batches, sequence_length_q, sequence_length_kv,
        dimension, _kv_heads, _groups);
  }
  const auto o_proj = _o_layer(
      Tensor<__nv_bfloat16>{.shape = {attention_out_storage->elems},
                            .dimensions = 1,
                            .storage = attention_out_storage}
          .reshape({static_cast<int>(batches),
                    static_cast<int>(sequence_length_q),
                    static_cast<int>(_kv_heads * _groups * dimension)}),
      o_proj_out_storage);
  return o_proj;
}
