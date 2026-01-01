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
Dense::operator()(LayerContext &ctx, const Tensor<__nv_bfloat16> &input,
                  const std::unique_ptr<InOutBuffer> &iobuf) {
  auto &scratchpad = ctx.scratchpad();
  assert(input.shape[input.dimensions - 1] == _in_features &&
         "invalid input dimension");
  const auto batches = input.elems() / _in_features;
  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks(ceil_div(batches, threads_per_block.x),
                        ceil_div(_out_features, threads_per_block.y));
  auto out_storage = _cache;
  const auto out_offset = _cached_batches * _out_features;
  if (!_cache->elems) {
    if (iobuf)
      out_storage =
          iobuf->output_for<__nv_bfloat16>(input, batches * _out_features);
    else
      out_storage =
          scratchpad.make_storage<__nv_bfloat16>(batches * _out_features);
  } else
    _cached_batches += batches;

  if (_use_activation)
    dense<<<num_blocks, threads_per_block>>>(
        out_storage->data + out_offset, input.storage->data,
        _weight.storage->data, _bias ? _bias->storage->data : nullptr, batches,
        _in_features, _out_features);
  else
    gemm_transposed<<<num_blocks, threads_per_block>>>(
        out_storage->data + out_offset, input.storage->data,
        _weight.storage->data, _bias ? _bias->storage->data : nullptr, 1.0f,
        batches, _out_features, _in_features);

  Tensor<__nv_bfloat16> res = {.dimensions = input.dimensions,
                               .storage = out_storage};
  std::copy_n(input.shape.begin(), input.dimensions - 1, res.shape.begin());
  res.shape[input.dimensions - 1] = _out_features;
  if (_cache->elems) {
    assert(res.dimensions == 2 && "cached dense layer only allow 2D tensors");
    res.shape[0] = _cached_batches;
  }
  return res;
}

Tensor<__nv_bfloat16>
RMSNorm::operator()(LayerContext &ctx, const Tensor<__nv_bfloat16> &input,
                    const std::unique_ptr<InOutBuffer> &iobuf) {
  auto &scratchpad = ctx.scratchpad();
  assert(input.shape[input.dimensions - 1] == _dimensions &&
         "invalid input dimension");
  const auto batches = input.elems() / _dimensions;

  Tensor<__nv_bfloat16> reshaped =
      input.reshape({-1, static_cast<int>(_dimensions)});
  const dim3 threads_per_block(1024);
  const dim3 num_blocks(ceil_div(reshaped.shape[0], threads_per_block.x));
  std::shared_ptr<Storage<__nv_bfloat16>> out_storage;
  if (iobuf)
    out_storage = iobuf->output_for<__nv_bfloat16>(input, input.elems());
  else
    out_storage = scratchpad.make_storage<__nv_bfloat16>(input.elems());
  rmsnorm<<<num_blocks, threads_per_block>>>(
      out_storage->data, input.storage->data, _weight.storage->data, batches,
      _dimensions, _epsilon);

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
    LayerContext &ctx, const Tensor<__nv_bfloat16> &input_q,
    const Tensor<__nv_bfloat16> &input_k, const Tensor<__nv_bfloat16> &input_v,
    bool causal_mask, const std::unique_ptr<InOutBuffer> &iobuf) {
  auto &scratchpad = ctx.scratchpad();
  assert(_k_layer.cached_batches() == _v_layer.cached_batches() &&
         "K and V caches should hold equal number of tokens");
  const auto cached_tokens = _k_layer.cached_batches();
  const auto q_proj = _q_layer(ctx, input_q, iobuf);
  const auto k_proj = _k_layer(ctx, input_k);
  const auto v_proj = _v_layer(ctx, input_v);

  assert(q_proj.dimensions == k_proj.dimensions &&
         k_proj.dimensions == v_proj.dimensions &&
         "QKV dimension should match");
  assert(k_proj.shape[k_proj.dimensions - 2] ==
             v_proj.shape[v_proj.dimensions - 2] &&
         "KV sequence length should match");
  const auto sequence_length_q = q_proj.shape[q_proj.dimensions - 2];
  const auto sequence_length_kv = k_proj.shape[q_proj.dimensions - 2];
  const auto dimension = _k_layer.out_features() / _kv_heads;
  assert(dimension % 2 == 0 && "Q and K dimension should be even");
  const dim3 threads_per_block(1024);
  {
    const dim3 num_blocks(ceil_div(q_proj.elems() / 2, threads_per_block.x));
    rope<<<num_blocks, threads_per_block>>>(
        q_proj.storage->data, q_proj.storage->data, _cos_basis.storage->data,
        _sin_basis.storage->data, cached_tokens, sequence_length_q,
        _kv_heads * _groups, dimension / 2);
  }
  const auto q_proj_rope =
      Tensor<__nv_bfloat16>{.shape = q_proj.shape,
                            .dimensions = q_proj.dimensions,
                            .storage = q_proj.storage};
  {
    const dim3 num_blocks(
        ceil_div((k_proj.elems() - cached_tokens * _k_layer.out_features()) / 2,
                 threads_per_block.x));
    rope<<<num_blocks, threads_per_block>>>(
        k_proj.storage->data + cached_tokens * _k_layer.out_features(),
        k_proj.storage->data + cached_tokens * _k_layer.out_features(),
        _cos_basis.storage->data, _sin_basis.storage->data, cached_tokens,
        sequence_length_kv - cached_tokens, _kv_heads, dimension / 2);
  }
  const auto k_proj_rope =
      Tensor<__nv_bfloat16>{.shape = k_proj.shape,
                            .dimensions = k_proj.dimensions,
                            .storage = k_proj.storage};
  assert(k_proj_rope.elems() == v_proj.elems() &&
         "KV element count should match");
  const auto scores_elems =
      _kv_heads * _groups * sequence_length_q * sequence_length_kv;
  auto scores_out_storage = scratchpad.make_storage<float>(
      _kv_heads * _groups * sequence_length_q * sequence_length_kv);
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
  auto attention_out_storage =
      scratchpad.make_storage<__nv_bfloat16>(attention_elems);
  {
    const dim3 num_blocks(ceil_div(attention_elems, 1024));
    grouped_query_attention_output<<<num_blocks, threads_per_block>>>(
        attention_out_storage->data, scores_out_storage->data,
        v_proj.storage->data, sequence_length_q, sequence_length_kv, dimension,
        _kv_heads, _groups);
  }
  const auto o_proj = _o_layer(
      ctx, Tensor{.shape = {attention_elems},
                  .dimensions = 1,
                  .storage = attention_out_storage}
               .reshape({static_cast<int>(sequence_length_q),
                         static_cast<int>(_kv_heads * _groups * dimension)}));
  return o_proj;
}

Tensor<__nv_bfloat16>
Qwen2TransformerBlock::operator()(LayerContext &ctx,
                                  const Tensor<__nv_bfloat16> &input,
                                  const std::unique_ptr<InOutBuffer> &iobuf) {
  auto &scratchpad = ctx.scratchpad();
  ScratchPadScope scope{scratchpad};

  const auto input_normalized = _input_norm_layer(ctx, input);
  const auto attention_output = _attention_layer(
      ctx, input_normalized, input_normalized, input_normalized, true);
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
      _post_attention_norm_layer(ctx, attention_output);
  const auto gate_proj_output =
      _gate_proj_layer(ctx, attention_output_normalized);
  const auto up_proj_output = _up_proj_layer(ctx, attention_output_normalized);
  assert(gate_proj_output.shape == up_proj_output.shape &&
         "gate and up projection shape should match");
  {
    const dim3 num_blocks(
        ceil_div(gate_proj_output.elems(), threads_per_block.x));
    elementwise_product<<<num_blocks, threads_per_block>>>(
        gate_proj_output.storage->data, gate_proj_output.storage->data,
        up_proj_output.storage->data, 1.0f, gate_proj_output.elems());
  }
  const auto down_proj_output = _down_proj_layer(ctx, gate_proj_output);
  assert(down_proj_output.shape == attention_output.shape &&
         "down projection and attention output shape should match");
  std::shared_ptr<Storage<__nv_bfloat16>> out_storage;
  if (iobuf)
    out_storage =
        iobuf->output_for<__nv_bfloat16>(input, down_proj_output.elems());
  else
    out_storage =
        scratchpad.make_storage<__nv_bfloat16>(down_proj_output.elems());
  {
    const dim3 num_blocks(
        ceil_div(down_proj_output.elems(), threads_per_block.x));
    elementwise_add<<<num_blocks, threads_per_block>>>(
        out_storage->data, down_proj_output.storage->data,
        attention_output.storage->data, down_proj_output.elems());
  }
  return {.shape = down_proj_output.shape,
          .dimensions = down_proj_output.dimensions,
          .storage = out_storage};
}

Tensor<__nv_bfloat16>
Embedding::operator()(LayerContext &ctx, const Tensor<int> &input,
                      const std::unique_ptr<InOutBuffer> &iobuf) {
  auto &scratchpad = ctx.scratchpad();
  const auto sequence_length = input.elems();
  const dim3 threads_per_block(1024);
  const dim3 num_blocks(ceil_div(input.elems(), threads_per_block.x));
  std::shared_ptr<Storage<__nv_bfloat16>> out_storage;
  if (iobuf)
    out_storage =
        iobuf->output_for<__nv_bfloat16>(input, sequence_length * _dimension);
  else
    out_storage =
        scratchpad.make_storage<__nv_bfloat16>(sequence_length * _dimension);
  lookup_embeddings<<<num_blocks, threads_per_block>>>(
      out_storage->data, input.storage->data, _embedding_table.storage->data,
      sequence_length, _dimension);

  auto shape = input.shape;
  shape[input.dimensions] = _dimension;
  return {.shape = shape,
          .dimensions = input.dimensions + 1,
          .storage = out_storage};
}

Tensor<int> Sampler::operator()(LayerContext &ctx,
                                const Tensor<__nv_bfloat16> &logits,
                                const std::unique_ptr<InOutBuffer> &iobuf) {
  auto &scratchpad = ctx.scratchpad();
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

  std::shared_ptr<Storage<int>> out_storage;
  if (iobuf)
    out_storage = iobuf->output_for<int>(logits, batches);
  else
    out_storage = scratchpad.make_storage<int>(batches);
  CHECK_CUDA(cudaMemcpy(out_storage->data, current_indices->data,
                        batches * sizeof(int), cudaMemcpyDeviceToDevice));
  return {.shape = {static_cast<std::size_t>(batches)},
          .dimensions = 1,
          .storage = out_storage};
}

Tensor<__nv_bfloat16>
LmHeadDense::operator()(LayerContext &ctx, const Tensor<__nv_bfloat16> &input,
                        const std::unique_ptr<InOutBuffer> &iobuf) {
  auto &scratchpad = ctx.scratchpad();
  assert(input.dimensions >= 1 && "input should have at least 1 dimension");
  const auto last_dim = input.shape[input.dimensions - 1];
  assert(last_dim == _in_features && "invalid input dimension");

  const std::size_t tokens = input.elems() / last_dim;
  assert(tokens > 0 && "input should have at least one token");
  const std::size_t offset = (tokens - 1) * last_dim;
  const __nv_bfloat16 *last_token = input.storage->data + offset;

  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks(1, ceil_div(_out_features, threads_per_block.y));
  std::shared_ptr<Storage<__nv_bfloat16>> out_storage;
  if (iobuf)
    out_storage = iobuf->output_for<__nv_bfloat16>(input, _out_features);
  else
    out_storage = scratchpad.make_storage<__nv_bfloat16>(_out_features);
  gemm_transposed<<<num_blocks, threads_per_block>>>(
      out_storage->data, last_token, _weight.storage->data, nullptr, 1.0f, 1,
      _out_features, _in_features);
  return {.shape = {_out_features}, .dimensions = 1, .storage = out_storage};
}
