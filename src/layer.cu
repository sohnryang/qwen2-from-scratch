#include "cuda_utils.h"
#include "kernel.h"
#include "layer.h"
#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>

#include <cuda_bf16.h>

Tensor<__nv_bfloat16> Dense::operator()(const Tensor<__nv_bfloat16> &input) {
  assert(input.shape[input.dimensions - 1] == _in_features &&
         "invalid input dimension");
  const auto batches = input.elems() / _in_features;
  assert(batches <= _max_sequence_length &&
         "input sequence length exceeds preallocated capacity");
  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks(ceil_div(batches, threads_per_block.x),
                        ceil_div(_out_features, threads_per_block.y));
  if (_use_activation)
    dense<<<num_blocks, threads_per_block>>>(
        _out_storage->data, input.storage->data, _weight.storage->data,
        _bias ? _bias->storage->data : nullptr, batches, _in_features,
        _out_features);
  else
    gemm_transposed<<<num_blocks, threads_per_block>>>(
        _out_storage->data, input.storage->data, _weight.storage->data,
        _bias ? _bias->storage->data : nullptr, 1.0f, batches, _out_features,
        _in_features);

  Tensor<__nv_bfloat16> res = {.dimensions = input.dimensions,
                               .storage = _out_storage};
  std::copy_n(input.shape.begin(), input.dimensions - 1, res.shape.begin());
  res.shape[input.dimensions - 1] = _out_features;
  return res;
}

Tensor<__nv_bfloat16> RMSNorm::operator()(const Tensor<__nv_bfloat16> &input) {
  assert(input.shape[input.dimensions - 1] == _dimensions &&
         "invalid input dimension");
  const auto batches = input.elems() / _dimensions;
  assert(batches <= _max_sequence_length &&
         "input sequence length exceeds preallocated capacity");

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
        _out_storage->data + batch * _dimensions,
        reshaped.storage->data + batch * _dimensions, _weight.storage->data,
        rsqrtf(res / _dimensions + _epsilon), _dimensions);
  }
  CHECK_CUDA(cudaFree(intermediate_arr));
  CHECK_CUDA(cudaFree(out_arr));

  return {.shape = input.shape,
          .dimensions = input.dimensions,
          .storage = _out_storage};
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
           .storage = std::make_shared<Storage<float>>(_cos_basis.elems())}),
      _q_proj_rope_out_storage(std::make_shared<Storage<__nv_bfloat16>>(
          _max_sequence_length * _q_layer.out_features())),
      _k_proj_rope_out_storage(std::make_shared<Storage<__nv_bfloat16>>(
          _max_sequence_length * _k_layer.out_features())),
      _attention_out_storage(std::make_shared<Storage<__nv_bfloat16>>(
          _kv_heads * _groups * _max_sequence_length *
          (_k_layer.out_features() / _kv_heads))) {
  assert(_k_layer.out_features() == _v_layer.out_features() &&
         "K and V dimensions mismatch");
  assert(_q_layer.out_features() % _k_layer.out_features() == 0 &&
         "Q head count should be multiple of KV head count");
  assert(_k_layer.out_features() % kv_heads == 0 &&
         "KV layer output dimension should be multiple of KV heads");
  assert(_q_layer.out_features() % (kv_heads * groups) == 0 &&
         "Q layer output dimension should be multiple of Q heads");
  assert(_q_layer.max_sequence_length() == _max_sequence_length &&
         _k_layer.max_sequence_length() == _max_sequence_length &&
         _v_layer.max_sequence_length() == _max_sequence_length &&
         _o_layer.max_sequence_length() == _max_sequence_length &&
         "dense layer max sequence length should match attention max length");

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
    const Tensor<__nv_bfloat16> &input_v, bool causal_mask) {
  const auto q_proj = _q_layer(input_q);
  const auto k_proj = _k_layer(input_k);
  const auto v_proj = _v_layer(input_v);

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
  assert(sequence_length_q <= _max_sequence_length &&
         sequence_length_kv <= _max_sequence_length &&
         "sequence length exceeds preallocated capacity");
  const auto dimension = _k_layer.out_features() / _kv_heads;
  assert(dimension % 2 == 0 && "Q and K dimension should be even");
  const dim3 threads_per_block(1024);
  {
    const dim3 num_blocks(ceil_div(q_proj.elems() / 2, threads_per_block.x));
    rope<<<num_blocks, threads_per_block>>>(
        _q_proj_rope_out_storage->data, q_proj.storage->data,
        _cos_basis.storage->data, _sin_basis.storage->data, sequence_length_q,
        _kv_heads * _groups, dimension / 2);
  }
  const auto q_proj_rope =
      Tensor<__nv_bfloat16>{.shape = q_proj.shape,
                            .dimensions = q_proj.dimensions,
                            .storage = _q_proj_rope_out_storage};
  {
    const dim3 num_blocks(ceil_div(k_proj.elems() / 2, threads_per_block.x));
    rope<<<num_blocks, threads_per_block>>>(
        _k_proj_rope_out_storage->data, k_proj.storage->data,
        _cos_basis.storage->data, _sin_basis.storage->data, sequence_length_kv,
        _kv_heads, dimension / 2);
  }
  const auto k_proj_rope =
      Tensor<__nv_bfloat16>{.shape = k_proj.shape,
                            .dimensions = k_proj.dimensions,
                            .storage = _k_proj_rope_out_storage};
  assert(k_proj_rope.elems() == v_proj.elems() &&
         q_proj_rope.elems() / sequence_length_q ==
             k_proj_rope.elems() / sequence_length_kv * _groups &&
         "QKV element count should match");
  const auto scores_elems =
      _kv_heads * _groups * sequence_length_q * sequence_length_kv;
  auto scores_out_storage = std::make_shared<Storage<float>>(
      _kv_heads * _groups * _max_sequence_length * _max_sequence_length);
  {
    const dim3 num_blocks(ceil_div(scores_elems, threads_per_block.x));
    if (causal_mask)
      grouped_query_attention_scores_masked<<<num_blocks, threads_per_block>>>(
          scores_out_storage->data, q_proj_rope.storage->data,
          k_proj_rope.storage->data, sequence_length_q, sequence_length_kv,
          dimension, _kv_heads, _groups);
    else
      grouped_query_attention_scores<<<num_blocks, threads_per_block>>>(
          scores_out_storage->data, q_proj_rope.storage->data,
          k_proj_rope.storage->data, sequence_length_q, sequence_length_kv,
          dimension, _kv_heads, _groups);
  }
  {
    const dim3 num_blocks(
        ceil_div(scores_elems / sequence_length_kv, threads_per_block.x));
    softmax<<<num_blocks, threads_per_block>>>(
        scores_out_storage->data, scores_out_storage->data,
        scores_elems / sequence_length_kv, sequence_length_kv);
  }
  const auto attention_elems =
      _kv_heads * _groups * sequence_length_q * dimension;
  {
    const dim3 num_blocks(ceil_div(attention_elems, 1024));
    grouped_query_attention_output<<<num_blocks, threads_per_block>>>(
        _attention_out_storage->data, scores_out_storage->data,
        v_proj.storage->data, sequence_length_q, sequence_length_kv, dimension,
        _kv_heads, _groups);
  }
  const auto o_proj = _o_layer(
      Tensor<__nv_bfloat16>{.shape = {attention_elems},
                            .dimensions = 1,
                            .storage = _attention_out_storage}
          .reshape({static_cast<int>(sequence_length_q),
                    static_cast<int>(_kv_heads * _groups * dimension)}));
  return o_proj;
}

Tensor<__nv_bfloat16>
Qwen2TransformerBlock::operator()(const Tensor<__nv_bfloat16> &input,
                                  bool causal_mask) {
  const auto input_normalized = _input_norm_layer(input);
  const auto attention_output = _attention_layer(
      input_normalized, input_normalized, input_normalized, causal_mask);
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
  const auto attention_output_normalized =
      _post_attention_norm_layer(attention_output);
  const auto gate_proj_output = _gate_proj_layer(attention_output_normalized);
  const auto up_proj_output = _up_proj_layer(attention_output_normalized);
  assert(gate_proj_output.shape == up_proj_output.shape &&
         "gate and up projection shape should match");
  {
    const dim3 num_blocks(
        ceil_div(gate_proj_output.elems(), threads_per_block.x));
    elementwise_product<<<num_blocks, threads_per_block>>>(
        gate_proj_output.storage->data, gate_proj_output.storage->data,
        up_proj_output.storage->data, 1.0f, gate_proj_output.elems());
  }
  const auto down_proj_output = _down_proj_layer(gate_proj_output);
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

Tensor<__nv_bfloat16> Embedding::operator()(const Tensor<int> &input) {
  assert(input.elems() <= _max_sequence_length &&
         "input sequence length exceeds preallocated capacity");

  const auto sequence_length = input.elems();
  const dim3 threads_per_block(1024);
  const dim3 num_blocks(ceil_div(input.elems(), threads_per_block.x));
  lookup_embeddings<<<num_blocks, threads_per_block>>>(
      _out_storage->data, input.storage->data, _embedding_table.storage->data,
      sequence_length, _dimension);

  auto shape = input.shape;
  shape[input.dimensions] = _dimension;
  return {.shape = shape,
          .dimensions = input.dimensions + 1,
          .storage = _out_storage};
}

Tensor<int> Sampler::operator()(const Tensor<__nv_bfloat16> &logits) {
  assert(logits.shape[logits.dimensions - 1] == _vocab_size &&
         "vocab dimension should match sampler");

  const auto batches = logits.dimensions > 1 ? logits.shape[0] : 1;
  const std::size_t threads_per_block = 1024;
  std::size_t blocks = ceil_div(_vocab_size, threads_per_block);

  if (!_vals_storage || _vals_storage->elems < blocks * batches)
    _vals_storage = std::make_shared<Storage<float>>(
        blocks * static_cast<std::size_t>(batches));
  if (!_indices_storage || _indices_storage->elems < blocks * batches)
    _indices_storage = std::make_shared<Storage<int>>(
        blocks * static_cast<std::size_t>(batches));

  auto current_vals = _vals_storage;
  auto current_indices = _indices_storage;
  const auto shared_bytes = threads_per_block * (sizeof(float) + sizeof(int));
  argmax_first<<<dim3(blocks, batches), threads_per_block, shared_bytes>>>(
      logits.storage->data, current_vals->data, current_indices->data,
      _vocab_size);

  while (blocks > 1) {
    const auto next_blocks = ceil_div(blocks, threads_per_block);
    if (!_vals_storage_next ||
        _vals_storage_next->elems < next_blocks * batches)
      _vals_storage_next = std::make_shared<Storage<float>>(
          next_blocks * static_cast<std::size_t>(batches));
    if (!_indices_storage_next ||
        _indices_storage_next->elems < next_blocks * batches)
      _indices_storage_next = std::make_shared<Storage<int>>(
          next_blocks * static_cast<std::size_t>(batches));
    auto next_vals = _vals_storage_next;
    auto next_indices = _indices_storage_next;
    argmax_reduce<<<dim3(next_blocks, batches), threads_per_block,
                    shared_bytes>>>(current_vals->data, current_indices->data,
                                    next_vals->data, next_indices->data,
                                    blocks);
    blocks = next_blocks;
    current_vals = std::move(next_vals);
    current_indices = std::move(next_indices);
  }

  _out_storage = current_indices;
  return {.shape = {static_cast<std::size_t>(batches)},
          .dimensions = 1,
          .storage = _out_storage};
}

Tensor<__nv_bfloat16>
LmHeadDense::operator()(const Tensor<__nv_bfloat16> &input) {
  assert(input.dimensions >= 1 && "input should have at least 1 dimension");
  const auto last_dim = input.shape[input.dimensions - 1];
  assert(last_dim == _in_features && "invalid input dimension");

  const std::size_t tokens = input.elems() / last_dim;
  assert(tokens > 0 && "input should have at least one token");
  const std::size_t offset = (tokens - 1) * last_dim;
  const __nv_bfloat16 *last_token = input.storage->data + offset;

  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks(1, ceil_div(_out_features, threads_per_block.y));
  gemm_transposed<<<num_blocks, threads_per_block>>>(
      _out_storage->data, last_token, _weight.storage->data, nullptr, 1.0f, 1,
      _out_features, _in_features);
  return {.shape = {_out_features}, .dimensions = 1, .storage = _out_storage};
}
