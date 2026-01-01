#include "model.h"
#include "cuda_utils.h"
#include "layer.h"
#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <format>
#include <memory>
#include <vector>

Qwen2Model::Qwen2Model(
    const Embedding &embedding_layer,
    const std::vector<Qwen2TransformerBlock> &transformer_blocks,
    const RMSNorm &rmsnorm, const LmHeadDense &lm_head, const Sampler &sampler,
    std::size_t scratchpad_size, std::size_t iobuf_size, int eos_token)
    : _embedding_layer(embedding_layer),
      _transformer_blocks(transformer_blocks), _rmsnorm(rmsnorm),
      _lm_head(lm_head), _sampler(sampler), _layer_ctx(scratchpad_size),
      _iobuf{std::make_unique<InOutBuffer>(iobuf_size)}, _eos_token{eos_token},
      _max_sequence_length{
          _transformer_blocks[0].attention_layer().max_sequence_length()} {
  assert(std::all_of(_transformer_blocks.begin(), _transformer_blocks.end(),
                     [&](const auto &x) {
                       return x.attention_layer().max_sequence_length() ==
                              _max_sequence_length;
                     }) &&
         "max sequence length mismatch");
}

Qwen2Model Qwen2Model::from_parameters(
    const std::map<std::string, Tensor<__nv_bfloat16>> &weights,
    std::size_t max_sequence_length, int eos_token) {
  const auto embedding_weight = weights.at("model.embed_tokens.weight");
  const auto embedding_layer = Embedding::from_parameter(embedding_weight);
  const auto iobuf_size = std::max(
      max_sequence_length * embedding_layer.dimension() * sizeof(__nv_bfloat16),
      max_sequence_length * sizeof(int));

  const std::size_t num_kv_heads = 2;
  const std::size_t groups = 6;
  const int encoding_base = 1000000;
  const auto k_weight0 = weights.at("model.layers.0.self_attn.k_proj.weight");
  const auto head_dimension = k_weight0.shape[0] / num_kv_heads;
  const auto rope_basis = GroupedQueryAttention::make_rope_bases(
      max_sequence_length, head_dimension, encoding_base);

  std::vector<Qwen2TransformerBlock> transformer_blocks;
  for (int i = 0; i < 28; i++) {
    const auto key_prefix = std::format("model.layers.{}", i);

    const auto input_layernorm_weight =
        weights.at(key_prefix + ".input_layernorm.weight");
    const auto input_layernorm =
        RMSNorm::from_parameter(input_layernorm_weight, 1e-6);

    const auto q_weight = weights.at(key_prefix + ".self_attn.q_proj.weight");
    const auto q_bias = weights.at(key_prefix + ".self_attn.q_proj.bias");
    const auto q_layer = Dense::from_parameters(q_weight, q_bias, false);
    const auto k_weight = weights.at(key_prefix + ".self_attn.k_proj.weight");
    const auto k_bias = weights.at(key_prefix + ".self_attn.k_proj.bias");
    const auto k_layer =
        Dense::from_parameters(k_weight, k_bias, false, max_sequence_length);
    const auto v_weight = weights.at(key_prefix + ".self_attn.v_proj.weight");
    const auto v_bias = weights.at(key_prefix + ".self_attn.v_proj.bias");
    const auto v_layer =
        Dense::from_parameters(v_weight, v_bias, false, max_sequence_length);
    const auto o_weight = weights.at(key_prefix + ".self_attn.o_proj.weight");
    const auto o_layer = Dense::from_parameters(o_weight, false);

    const auto post_attention_layernorm_weight =
        weights.at(key_prefix + ".post_attention_layernorm.weight");
    const auto post_attention_layernorm =
        RMSNorm::from_parameter(post_attention_layernorm_weight, 1e-6);
    const auto attention_layer = GroupedQueryAttention(
        num_kv_heads, groups, max_sequence_length, encoding_base, q_layer,
        k_layer, v_layer, o_layer, rope_basis);

    const auto gate_proj_weight =
        weights.at(key_prefix + ".mlp.gate_proj.weight");
    const auto gate_proj = Dense::from_parameters(gate_proj_weight, true);
    const auto up_proj_weight = weights.at(key_prefix + ".mlp.up_proj.weight");
    const auto up_proj = Dense::from_parameters(up_proj_weight, false);
    const auto down_proj_weight =
        weights.at(key_prefix + ".mlp.down_proj.weight");
    const auto down_proj = Dense::from_parameters(down_proj_weight, false);

    transformer_blocks.emplace_back(input_layernorm, attention_layer,
                                    post_attention_layernorm, gate_proj,
                                    up_proj, down_proj);
  }
  const auto scratchpad_size =
      2 * transformer_blocks[0].attention_layer().groups() *
      transformer_blocks[0].attention_layer().kv_heads() * max_sequence_length *
      max_sequence_length * sizeof(float);

  const auto rmsnorm_weight = weights.at("model.norm.weight");
  const auto rmsnorm = RMSNorm::from_parameter(rmsnorm_weight, 1e-6);

  const auto lm_head = LmHeadDense::from_parameters(embedding_weight);
  const auto sampler = Sampler(lm_head.out_features());
  return Qwen2Model(embedding_layer, transformer_blocks, rmsnorm, lm_head,
                    sampler, scratchpad_size, iobuf_size, eos_token);
}

GenerationResult Qwen2Model::generate(const std::vector<int> &user_prompt) {
  int next_token;
  const auto prev_cached_tokens = _cached_tokens;
  auto model_input = Tensor{.shape = {user_prompt.size()},
                            .dimensions = 1,
                            .storage = _iobuf->make_input_storage(user_prompt)};
  GenerationResult result;
  auto &generated_tokens = result.tokens;
  const auto start = std::chrono::steady_clock::now();
  do {
    if (_cached_tokens + model_input.elems() > _max_sequence_length) {
      _cached_tokens = prev_cached_tokens;
      for (auto &block : _transformer_blocks)
        block.rollback(prev_cached_tokens);
      const auto elapsed = std::chrono::steady_clock::now() - start;
      result.metrics.time_to_first_token = elapsed;
      result.metrics.total_duration = elapsed;
      return result;
    }
    _cached_tokens += model_input.elems();

    const auto embedding_out =
        _embedding_layer(_layer_ctx, model_input, _iobuf);
    auto blocks_out = embedding_out;
    for (auto &block : _transformer_blocks)
      blocks_out = block(_layer_ctx, blocks_out, _iobuf);
    const auto rmsnorm_out = _rmsnorm(_layer_ctx, blocks_out, _iobuf);
    const auto lm_head_out = _lm_head(_layer_ctx, rmsnorm_out, _iobuf);
    const auto sampler_out = _sampler(_layer_ctx, lm_head_out, _iobuf);

    CHECK_CUDA(cudaStreamSynchronize(_layer_ctx.stream()));
    CHECK_CUDA(cudaMemcpy(&next_token, sampler_out.storage->data, sizeof(int),
                          cudaMemcpyDeviceToHost));
    generated_tokens.push_back(next_token);
    if (result.metrics.time_to_first_token ==
        std::chrono::steady_clock::duration{})
      result.metrics.time_to_first_token =
          std::chrono::steady_clock::now() - start;
    model_input = sampler_out;
  } while (next_token != _eos_token);
  result.metrics.total_duration = std::chrono::steady_clock::now() - start;
  return result;
}
