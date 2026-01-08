#pragma once

#include "layer.h"
#include "tensor.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <thread>
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
public:
  struct StreamResult {
    std::vector<int> tokens;
    bool done;
    bool out_of_space;
  };

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
  std::atomic<std::size_t> _streamed_tokens = 0;

  std::vector<int> _appended_prompt;
  // TODO: align atomics to avoid false sharing
  std::atomic<bool> _prompt_ready = false;

  int *_published_tokens_ptr;
  int *_published_tokens_buf;
  cudaEvent_t _publish_event;
  cudaStream_t _copy_stream;
  std::atomic<bool> _published = false;
  std::atomic<bool> _stop_generating = false;
  std::atomic<bool> _publish_available = true;

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

  std::thread spawn_producer();

  bool append_prompt(const std::vector<int> &prompt);

  StreamResult stream_response();
};
