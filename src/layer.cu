#include "cuda_utils.h"
#include "kernel.h"
#include "layer.h"
#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>

#include <cuda_bf16.h>

Tensor<__nv_bfloat16>
Dense::operator()(const Tensor<__nv_bfloat16> &input,
                  std::shared_ptr<Storage<__nv_bfloat16>> out_storage) {
  assert(input.shape[input.dimensions - 1] == _in_features &&
         "invalid input dimension");
  const auto batches = input.elems() / _in_features;
  const auto out_elems = batches * _out_features;
  if (out_storage)
    assert(out_storage->elems < out_elems &&
           "insufficient output storage size");
  else
    out_storage = std::make_shared<Storage<__nv_bfloat16>>(out_elems);
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
  const auto batches = input.elems() / _dimensions;
  if (out_storage)
    assert(out_storage->elems < input.elems() &&
           "insufficient output storage size");
  else
    out_storage = std::make_shared<Storage<__nv_bfloat16>>(input.elems());

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

GroupedQueryAttention::GroupedQueryAttention(
    std::size_t kv_heads, std::size_t groups, std::size_t max_sequence_length,
    int encoding_base, const Dense &q_layer, const Dense &k_layer,
    const Dense &v_layer, const Dense &o_layer)
    : _kv_heads{kv_heads}, _groups{groups},
      _max_sequence_length{max_sequence_length}, _encoding_base{encoding_base},
      _q_layer(q_layer), _k_layer(k_layer), _v_layer(v_layer),
      _o_layer(o_layer),
      _cos_basis({.shape = {_max_sequence_length,
                            _k_layer.out_features() / _kv_heads / 2},
                  .dimensions = 2,
                  .storage = std::make_shared<Storage<float>>(
                      _max_sequence_length * _k_layer.out_features() /
                      _kv_heads / 2)}),
      _sin_basis(
          {.shape = _cos_basis.shape,
           .dimensions = 2,
           .storage = std::make_shared<Storage<float>>(_cos_basis.elems())}) {
  assert(_k_layer.out_features() == _v_layer.out_features() &&
         "K and V dimensions mismatch");
  assert(_q_layer.out_features() % _k_layer.out_features() == 0 &&
         "Q head count should be multiple of KV head count");
  assert(_k_layer.out_features() % kv_heads == 0 &&
         "KV layer output dimension should be multiple of KV heads");
  assert(_q_layer.out_features() % (kv_heads * groups) == 0 &&
         "Q layer output dimension should be multiple of Q heads");

  const auto half_dimension = _k_layer.out_features() / _kv_heads / 2;
  const dim3 threads_per_block(1024);
  const dim3 num_blocks(
      ceil_div(_max_sequence_length * half_dimension, threads_per_block.x));
  precompute_rope_bases<<<num_blocks, threads_per_block>>>(
      _cos_basis.storage->data, _sin_basis.storage->data, _encoding_base,
      _max_sequence_length, half_dimension);
}

Tensor<__nv_bfloat16> GroupedQueryAttention::operator()(
    const Tensor<__nv_bfloat16> &input_q, const Tensor<__nv_bfloat16> &input_k,
    const Tensor<__nv_bfloat16> &input_v, bool causal_mask,
    std::shared_ptr<Storage<__nv_bfloat16>> q_proj_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> q_proj_rope_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> k_proj_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> k_proj_rope_out_storage,
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
  assert(q_proj.shape[k_proj.dimensions - 2] ==
             k_proj.shape[v_proj.dimensions - 2] &&
         k_proj.shape[k_proj.dimensions - 2] ==
             v_proj.shape[v_proj.dimensions - 2] &&
         "QKV sequence length should match");
  const auto sequence_length_q = q_proj.shape[q_proj.dimensions - 2];
  const auto sequence_length_kv = k_proj.shape[q_proj.dimensions - 2];
  const auto dimension = _k_layer.out_features() / _kv_heads;
  assert(dimension % 2 == 0 && "Q and K dimension should be even");
  const auto batches =
      k_proj.elems() / _k_layer.out_features() / sequence_length_kv;
  const dim3 threads_per_block(1024);
  if (!q_proj_rope_out_storage)
    q_proj_rope_out_storage =
        std::make_shared<Storage<__nv_bfloat16>>(q_proj.elems());
  if (!k_proj_rope_out_storage)
    k_proj_rope_out_storage =
        std::make_shared<Storage<__nv_bfloat16>>(k_proj.elems());
  {
    const dim3 num_blocks(ceil_div(q_proj.elems() / 2, threads_per_block.x));
    rope<<<num_blocks, threads_per_block>>>(
        q_proj_rope_out_storage->data, q_proj.storage->data,
        _cos_basis.storage->data, _sin_basis.storage->data, batches,
        sequence_length_q, _kv_heads * _groups, dimension / 2);
  }
  const auto q_proj_rope =
      Tensor<__nv_bfloat16>{.shape = q_proj.shape,
                            .dimensions = q_proj.dimensions,
                            .storage = q_proj_rope_out_storage};
  {
    const dim3 num_blocks(ceil_div(k_proj.elems() / 2, threads_per_block.x));
    rope<<<num_blocks, threads_per_block>>>(
        k_proj_rope_out_storage->data, k_proj.storage->data,
        _cos_basis.storage->data, _sin_basis.storage->data, batches,
        sequence_length_kv, _kv_heads, dimension / 2);
  }
  const auto k_proj_rope =
      Tensor<__nv_bfloat16>{.shape = k_proj.shape,
                            .dimensions = k_proj.dimensions,
                            .storage = k_proj_rope_out_storage};
  assert(k_proj_rope.elems() == v_proj.elems() &&
         q_proj_rope.elems() / sequence_length_q ==
             k_proj_rope.elems() / sequence_length_kv * _groups &&
         "QKV element count should match");
  if (!scores_out_storage)
    scores_out_storage = std::make_shared<Storage<float>>(
        batches * _kv_heads * _groups * sequence_length_q * sequence_length_kv);
  {
    const dim3 num_blocks(
        ceil_div(scores_out_storage->elems, threads_per_block.x));
    if (causal_mask)
      grouped_query_attention_scores_masked<<<num_blocks, threads_per_block>>>(
          scores_out_storage->data, q_proj_rope.storage->data,
          k_proj_rope.storage->data, batches, sequence_length_q,
          sequence_length_kv, dimension, _kv_heads, _groups);
    else
      grouped_query_attention_scores<<<num_blocks, threads_per_block>>>(
          scores_out_storage->data, q_proj_rope.storage->data,
          k_proj_rope.storage->data, batches, sequence_length_q,
          sequence_length_kv, dimension, _kv_heads, _groups);
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

Tensor<__nv_bfloat16> Qwen2TransformerBlock::operator()(
    const Tensor<__nv_bfloat16> &input, bool causal_mask,
    std::shared_ptr<Storage<__nv_bfloat16>> input_norm_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> q_proj_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> q_proj_rope_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> k_proj_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> k_proj_rope_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> v_proj_out_storage,
    std::shared_ptr<Storage<float>> scores_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> attention_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> o_proj_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> post_attention_norm_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> gate_proj_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> up_proj_out_storage,
    std::shared_ptr<Storage<__nv_bfloat16>> down_proj_out_storage) {
  const auto input_normalized =
      _input_norm_layer(input, input_norm_out_storage);
  const auto attention_output = _attention_layer(
      input_normalized, input_normalized, input_normalized, causal_mask,
      q_proj_out_storage, q_proj_rope_out_storage, k_proj_out_storage,
      k_proj_rope_out_storage, v_proj_out_storage, scores_out_storage,
      attention_out_storage, o_proj_out_storage);
  assert(input.shape == attention_output.shape &&
         "input and attention output shape should match");
  const dim3 threads_per_block(1024);
  {
    const dim3 num_blocks(
        ceil_div(attention_output.elems(), threads_per_block.x));
    elementwise_add<<<num_blocks, threads_per_block>>>(
        attention_output.storage->data, attention_output.storage->data,
        input.storage->data, attention_output.elems());
  }
  const auto attention_output_normalized = _post_attention_norm_layer(
      attention_output, post_attention_norm_out_storage);
  const auto gate_proj_output =
      _gate_proj_layer(attention_output_normalized, gate_proj_out_storage);
  const auto up_proj_output =
      _up_proj_layer(attention_output_normalized, up_proj_out_storage);
  assert(gate_proj_output.shape == up_proj_output.shape &&
         "gate and up projection shape should match");
  {
    const dim3 num_blocks(
        ceil_div(gate_proj_output.elems(), threads_per_block.x));
    elementwise_product<<<num_blocks, threads_per_block>>>(
        gate_proj_output.storage->data, gate_proj_output.storage->data,
        up_proj_output.storage->data, 1.0f, gate_proj_output.elems());
  }
  const auto down_proj_output =
      _down_proj_layer(gate_proj_output, down_proj_out_storage);
  assert(down_proj_output.shape == attention_output.shape &&
         "down projection and attention output shape should match");
  {
    const dim3 num_blocks(
        ceil_div(down_proj_output.elems(), threads_per_block.x));
    elementwise_add<<<num_blocks, threads_per_block>>>(
        down_proj_output.storage->data, down_proj_output.storage->data,
        attention_output.storage->data, down_proj_output.elems());
  }
  return down_proj_output;
}

Tensor<__nv_bfloat16>
Embedding::operator()(const Tensor<int> &input,
                      std::shared_ptr<Storage<__nv_bfloat16>> out_storage) {
  if (!out_storage)
    out_storage =
        std::make_shared<Storage<__nv_bfloat16>>(input.elems() * _dimension);

  const auto sequence_length = input.shape[input.dimensions - 1];
  const auto batches = input.elems() / sequence_length;
  const dim3 threads_per_block(1024);
  const dim3 num_blocks(ceil_div(input.elems(), threads_per_block.x));
  lookup_embeddings<<<num_blocks, threads_per_block>>>(
      out_storage->data, input.storage->data, _embedding_table.storage->data,
      batches, sequence_length, _dimension);

  auto shape = input.shape;
  shape[input.dimensions] = _dimension;
  return {.shape = shape,
          .dimensions = input.dimensions + 1,
          .storage = out_storage};
}
