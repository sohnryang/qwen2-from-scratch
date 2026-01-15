#include "model.h"
#include "cuda_utils.h"
#include "layer.h"
#include "tensor.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <format>
#include <iterator>
#include <memory>
#include <thread>
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
  CHECK_CUDA(cudaMallocHost(&_published_tokens_ptr, sizeof(int)));
  CHECK_CUDA(cudaMallocHost(&_published_tokens_buf,
                            _max_sequence_length * sizeof(int)));
  CHECK_CUDA(cudaEventCreate(&_publish_event));
  CHECK_CUDA(cudaStreamCreate(&_copy_stream));
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
    const auto q_layer =
        Dense::from_parameters(q_weight, q_bias, false, 0, dim3(32, 4));
    const auto k_weight = weights.at(key_prefix + ".self_attn.k_proj.weight");
    const auto k_bias = weights.at(key_prefix + ".self_attn.k_proj.bias");
    const auto k_layer = Dense::from_parameters(
        k_weight, k_bias, false, max_sequence_length, dim3(32, 4));
    const auto v_weight = weights.at(key_prefix + ".self_attn.v_proj.weight");
    const auto v_bias = weights.at(key_prefix + ".self_attn.v_proj.bias");
    const auto v_layer = Dense::from_parameters(
        v_weight, v_bias, false, max_sequence_length, dim3(32, 4));
    const auto o_weight = weights.at(key_prefix + ".self_attn.o_proj.weight");
    const auto o_layer =
        Dense::from_parameters(o_weight, false, 0, dim3(32, 4));

    const auto post_attention_layernorm_weight =
        weights.at(key_prefix + ".post_attention_layernorm.weight");
    const auto post_attention_layernorm =
        RMSNorm::from_parameter(post_attention_layernorm_weight, 1e-6);
    const auto attention_layer = GroupedQueryAttention(
        num_kv_heads, groups, max_sequence_length, encoding_base, q_layer,
        k_layer, v_layer, o_layer, rope_basis);

    const auto gate_proj_weight =
        weights.at(key_prefix + ".mlp.gate_proj.weight");
    const auto gate_proj =
        Dense::from_parameters(gate_proj_weight, true, 0, dim3(32, 4));
    const auto up_proj_weight = weights.at(key_prefix + ".mlp.up_proj.weight");
    const auto up_proj =
        Dense::from_parameters(up_proj_weight, false, 0, dim3(32, 4));
    const auto down_proj_weight =
        weights.at(key_prefix + ".mlp.down_proj.weight");
    const auto down_proj =
        Dense::from_parameters(down_proj_weight, false, 0, dim3(32, 4));

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

  const auto lm_head =
      LmHeadDense::from_parameters(embedding_weight, dim3(32, 4));
  const auto sampler = Sampler(lm_head.out_features(), max_sequence_length);
  return Qwen2Model(embedding_layer, transformer_blocks, rmsnorm, lm_head,
                    sampler, scratchpad_size, iobuf_size, eos_token);
}

std::thread Qwen2Model::spawn_producer() {
  return std::thread([&] {
    while (true) {
      _prompt_ready.wait(false, std::memory_order_acquire);
      _prompt_ready.store(false, std::memory_order_release);
      _stop_generating.store(false, std::memory_order_release);

      auto model_input =
          Tensor{.shape = {_appended_prompt.size()},
                 .dimensions = 1,
                 .storage = _iobuf->make_input_storage(_appended_prompt)};
      std::size_t response_token_index = 0;
      const auto streamed_tokens =
          _streamed_tokens.load(std::memory_order_acquire);
      _layer_ctx.set_valid_tokens(streamed_tokens);
      while (!_stop_generating.load(std::memory_order_acquire) &&
             !_stop_producer.load(std::memory_order_acquire)) {
        if (streamed_tokens + response_token_index < _max_sequence_length) {
          const auto embedding_out =
              _embedding_layer(_layer_ctx, model_input, _iobuf);
          auto blocks_out = embedding_out;
          for (auto &block : _transformer_blocks)
            blocks_out = block(_layer_ctx, blocks_out, _iobuf);
          const auto rmsnorm_out = _rmsnorm(_layer_ctx, blocks_out, _iobuf);
          const auto lm_head_out = _lm_head(_layer_ctx, rmsnorm_out, _iobuf);
          const auto sampler_out = _sampler(_layer_ctx, lm_head_out, _iobuf);
          model_input = sampler_out;
          response_token_index++;
        }
        if (_publish_available.load(std::memory_order_acquire)) {
          CHECK_CUDA(cudaMemcpyAsync(
              _published_tokens_ptr, _layer_ctx.valid_tokens_ptr(), sizeof(int),
              cudaMemcpyDeviceToHost, _layer_ctx.stream()));
          CHECK_CUDA(cudaEventRecord(_publish_event, _layer_ctx.stream()));
          _published.store(true, std::memory_order_release);
          _publish_available.store(false, std::memory_order_release);
          _published.notify_one();
        }
      }
      if (_stop_producer.load(std::memory_order_acquire))
        return;
      for (auto &block : _transformer_blocks)
        block.rollback(_streamed_tokens.load(std::memory_order_acquire));
    }
  });
}

bool Qwen2Model::append_prompt(const std::vector<int> &prompt) {
  if (prompt.size() + _streamed_tokens.load(std::memory_order_acquire) >=
      _max_sequence_length)
    return false;

  _appended_prompt = prompt;
  _streamed_tokens.fetch_add(prompt.size(), std::memory_order_acq_rel);
  _prompt_ready.store(true, std::memory_order_release);
  _prompt_ready.notify_one();
  return true;
}

Qwen2Model::StreamResult Qwen2Model::stream_response() {
  _published.wait(false, std::memory_order_acquire);
  CHECK_CUDA(cudaStreamWaitEvent(_copy_stream, _publish_event));

  const auto streamed_tokens = _streamed_tokens.load(std::memory_order_acquire);
  int new_tokens = *_published_tokens_ptr - streamed_tokens;
  std::vector<int> tokens;
  std::optional<std::chrono::time_point<std::chrono::steady_clock>> timestamp;
  auto done = false;
  auto out_of_space = false;
  if (new_tokens >= 0) {
    tokens.resize(new_tokens);
    CHECK_CUDA(cudaMemcpyAsync(
        _published_tokens_buf + streamed_tokens,
        _sampler.generated_tokens()->data + streamed_tokens,
        new_tokens * sizeof(int), cudaMemcpyDeviceToHost, _copy_stream));
    CHECK_CUDA(cudaStreamSynchronize(_copy_stream));
    timestamp = std::chrono::steady_clock::now();
    std::copy_n(_published_tokens_buf + streamed_tokens, new_tokens,
                tokens.begin());
    const auto eos_it = std::find(tokens.begin(), tokens.end(), _eos_token);
    if (eos_it != tokens.end()) {
      const auto generated_length = std::distance(tokens.begin(), eos_it) + 1;
      tokens.resize(generated_length);
      done = true;
    } else if (*_published_tokens_ptr >= _max_sequence_length)
      out_of_space = true;
    _streamed_tokens.fetch_add(tokens.size(), std::memory_order_acq_rel);
    if (done) {
      _stop_generating.store(true, std::memory_order_release);
      _stop_generating.notify_one();
    }
  }

  _published.store(false, std::memory_order_release);
  _publish_available.store(true, std::memory_order_release);
  _publish_available.notify_one();
  return {tokens, timestamp, done, out_of_space};
}

void Qwen2Model::quit() {
  _stop_producer.store(true, std::memory_order_release);
}
