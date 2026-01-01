#pragma once

#include "layer.h"
#include "tensor.h"

#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct GenerationMetrics {
  std::chrono::steady_clock::duration time_to_first_token{};
  std::chrono::steady_clock::duration total_duration{};
};

struct GenerationResult {
  std::vector<int> tokens;
  GenerationMetrics metrics;
};

class Qwen2Model {
private:
  Embedding _embedding_layer;
  std::vector<Qwen2TransformerBlock> _transformer_blocks;
  RMSNorm _rmsnorm;
  LmHeadDense _lm_head;
  Sampler _sampler;

  LayerContext _layer_ctx;
  std::unique_ptr<InOutBuffer> _iobuf;

  const int _eos_token;
  const std::size_t _max_sequence_length;
  std::size_t _cached_tokens = 0;

public:
  Qwen2Model(const Embedding &embedding_layer,
             const std::vector<Qwen2TransformerBlock> &transformer_blocks,
             const RMSNorm &rmsnorm, const LmHeadDense &lm_head,
             const Sampler &sampler, std::size_t scratchpad_size,
             std::size_t iobuf_size, int eos_token);

  static Qwen2Model
  from_parameters(const std::map<std::string, Tensor<__nv_bfloat16>> &weights,
                  std::size_t max_sequence_length = 8192,
                  int eos_token = 151645);

  GenerationResult generate(const std::vector<int> &user_prompt);
};
