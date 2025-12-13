#pragma once

#include "layer.h"
#include "tensor.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

class Qwen2Model {
private:
  Embedding _embedding_layer;
  std::vector<Qwen2TransformerBlock> _transformer_blocks;
  RMSNorm _rmsnorm;
  LmHeadDense _lm_head;
  Sampler _sampler;

  const int _eos_token;
  const std::size_t _max_sequence_length;
  Tensor<int> _prompt;

public:
  Qwen2Model(const Embedding &embedding_layer,
             const std::vector<Qwen2TransformerBlock> &transformer_blocks,
             const RMSNorm &rmsnorm, const LmHeadDense &lm_head,
             const Sampler &sampler, int eos_token);

  static Qwen2Model
  from_parameters(const std::map<std::string, Tensor<__nv_bfloat16>> &weights,
                  std::size_t max_sequence_length = 8192,
                  int eos_token = 151645);

  bool prefill(const std::vector<int> &prompt);

  std::vector<int> generate(const std::vector<int> &user_prompt);
};
