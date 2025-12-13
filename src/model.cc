#include "model.h"
#include "cuda_utils.h"
#include "layer.h"
#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <format>
#include <memory>
#include <vector>

Qwen2Model::Qwen2Model(
    const Embedding &embedding_layer,
    const std::vector<Qwen2TransformerBlock> &transformer_blocks,
    const RMSNorm &rmsnorm, const LmHeadDense &lm_head, const Sampler &sampler,
    int eos_token)
    : _embedding_layer(embedding_layer),
      _transformer_blocks(transformer_blocks), _rmsnorm(rmsnorm),
      _lm_head(lm_head), _sampler(sampler), _eos_token{eos_token},
      _max_sequence_length{
          _transformer_blocks[0].attention_layer().max_sequence_length()} {
  assert(std::all_of(_transformer_blocks.begin(), _transformer_blocks.end(),
                     [&](const auto &x) {
                       return x.attention_layer().max_sequence_length() ==
                              _max_sequence_length;
                     }) &&
         "max sequence length mismatch");
  _prompt = {.shape = {0},
             .dimensions = 1,
             .storage = std::make_shared<Storage<int>>(_max_sequence_length)};
}

Qwen2Model Qwen2Model::from_parameters(
    const std::map<std::string, Tensor<__nv_bfloat16>> &weights,
    std::size_t max_sequence_length, int eos_token) {
  const auto embedding_weight = weights.at("model.embed_tokens.weight");
  const auto embedding_layer =
      Embedding::from_parameter(embedding_weight, max_sequence_length);

  std::vector<Qwen2TransformerBlock> transformer_blocks;
  for (int i = 0; i < 28; i++) {
    const auto key_prefix = std::format("model.layers.{}", i);

    const auto input_layernorm_weight =
        weights.at(key_prefix + ".input_layernorm.weight");
    const auto input_layernorm = RMSNorm::from_parameter(
        input_layernorm_weight, 1e-6, max_sequence_length);

    const auto q_weight = weights.at(key_prefix + ".self_attn.q_proj.weight");
    const auto q_bias = weights.at(key_prefix + ".self_attn.q_proj.bias");
    const auto q_layer =
        Dense::from_parameters(q_weight, q_bias, false, max_sequence_length);
    const auto k_weight = weights.at(key_prefix + ".self_attn.k_proj.weight");
    const auto k_bias = weights.at(key_prefix + ".self_attn.k_proj.bias");
    const auto k_layer =
        Dense::from_parameters(k_weight, k_bias, false, max_sequence_length);
    const auto v_weight = weights.at(key_prefix + ".self_attn.v_proj.weight");
    const auto v_bias = weights.at(key_prefix + ".self_attn.v_proj.bias");
    const auto v_layer =
        Dense::from_parameters(v_weight, v_bias, false, max_sequence_length);
    const auto o_weight = weights.at(key_prefix + ".self_attn.o_proj.weight");
    const auto o_layer =
        Dense::from_parameters(o_weight, false, max_sequence_length);

    const auto post_attention_layernorm_weight =
        weights.at(key_prefix + ".post_attention_layernorm.weight");
    const auto post_attention_layernorm = RMSNorm::from_parameter(
        post_attention_layernorm_weight, 1e-6, max_sequence_length);
    const auto attention_layer = GroupedQueryAttention(
        2, 6, max_sequence_length, 1000000, q_layer, k_layer, v_layer, o_layer);

    const auto gate_proj_weight =
        weights.at(key_prefix + ".mlp.gate_proj.weight");
    const auto gate_proj =
        Dense::from_parameters(gate_proj_weight, true, max_sequence_length);
    const auto up_proj_weight = weights.at(key_prefix + ".mlp.up_proj.weight");
    const auto up_proj =
        Dense::from_parameters(up_proj_weight, false, max_sequence_length);
    const auto down_proj_weight =
        weights.at(key_prefix + ".mlp.down_proj.weight");
    const auto down_proj =
        Dense::from_parameters(down_proj_weight, false, max_sequence_length);

    transformer_blocks.emplace_back(input_layernorm, attention_layer,
                                    post_attention_layernorm, gate_proj,
                                    up_proj, down_proj);
  }

  const auto rmsnorm_weight = weights.at("model.norm.weight");
  const auto rmsnorm =
      RMSNorm::from_parameter(rmsnorm_weight, 1e-6, max_sequence_length);

  const auto lm_head = LmHeadDense::from_parameters(embedding_weight);
  const auto sampler = Sampler(lm_head.out_features());
  return Qwen2Model(embedding_layer, transformer_blocks, rmsnorm, lm_head,
                    sampler, eos_token);
}

bool Qwen2Model::prefill(const std::vector<int> &prompt) {
  if (prompt.size() >= _prompt.storage->elems)
    return false;
  CHECK_CUDA(cudaMemcpy(_prompt.storage->data, prompt.data(),
                        prompt.size() * sizeof(int), cudaMemcpyHostToDevice));
  _prompt.shape[0] += prompt.size();
  return true;
}

std::vector<int> Qwen2Model::generate(const std::vector<int> &user_prompt) {
  int next_token = -1;
  const auto prev_input_length = _prompt.shape[0];
  if (prev_input_length + user_prompt.size() >= _prompt.storage->elems)
    return {};

  CHECK_CUDA(cudaMemcpy(_prompt.storage->data + prev_input_length,
                        user_prompt.data(), user_prompt.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  _prompt.shape[0] += user_prompt.size();

  std::vector<int> generated_tokens;
  do {
    const auto current_input_length = _prompt.shape[0];
    if (current_input_length == _max_sequence_length) {
      _prompt.shape[0] = prev_input_length;
      return {};
    }

    const auto embedding_out = _embedding_layer(_prompt);
    auto blocks_out = embedding_out;
    for (auto &block : _transformer_blocks)
      blocks_out = block(blocks_out);
    const auto rmsnorm_out = _rmsnorm(blocks_out);
    const auto lm_head_out = _lm_head(rmsnorm_out);
    const auto sampler_out = _sampler(lm_head_out);

    CHECK_CUDA(cudaMemcpy(&next_token, sampler_out.storage->data, sizeof(int),
                          cudaMemcpyDeviceToHost));
    generated_tokens.push_back(next_token);
    CHECK_CUDA(cudaMemcpy(_prompt.storage->data + current_input_length,
                          sampler_out.storage->data, sizeof(int),
                          cudaMemcpyDeviceToDevice));
    _prompt.shape[0]++;
  } while (next_token != _eos_token);
  return generated_tokens;
}
