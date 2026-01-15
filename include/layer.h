#pragma once

#include "cuda_utils.h"
#include "tensor.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

class ScratchPad {
private:
  void *_pool = nullptr;
  std::size_t _size = 0;
  std::size_t _used = 0;

public:
  ~ScratchPad();
  ScratchPad() = default;
  ScratchPad(const ScratchPad &) = delete;
  ScratchPad(ScratchPad &&) noexcept;
  ScratchPad &operator=(ScratchPad &&) noexcept;

  explicit ScratchPad(std::size_t size);

  void reset();

  template <typename T>
  std::shared_ptr<Storage<T>> make_storage(std::size_t elems,
                                           std::size_t align = 256);
};

class ScratchPadScope {
private:
  ScratchPad &_scratchpad;

public:
  ~ScratchPadScope();
  ScratchPadScope() = delete;
  ScratchPadScope(const ScratchPadScope &) = default;
  ScratchPadScope &operator=(const ScratchPadScope &) = delete;

  ScratchPadScope(ScratchPad &scratchpad);
};

class LayerContext {
private:
  ScratchPad _scratchpad;
  cudaStream_t _stream = nullptr;
  int *_valid_tokens_ptr = nullptr;

public:
  ~LayerContext();
  LayerContext() = delete;
  LayerContext(const LayerContext &) = delete;
  LayerContext &operator=(const LayerContext &) = delete;
  LayerContext(LayerContext &&other) noexcept;
  LayerContext &operator=(LayerContext &&other) noexcept;

  explicit LayerContext(std::size_t scratchpad_size);

  ScratchPad &scratchpad() { return _scratchpad; }
  const ScratchPad &scratchpad() const { return _scratchpad; }
  cudaStream_t stream() const { return _stream; }
  int *valid_tokens_ptr() const { return _valid_tokens_ptr; }

  void set_valid_tokens(int valid_tokens);
};

class InOutBuffer {
private:
  std::array<void *, 2> _inout;
  std::size_t _size;

public:
  ~InOutBuffer();
  InOutBuffer() = delete;
  InOutBuffer(const InOutBuffer &) = delete;
  InOutBuffer(InOutBuffer &&) noexcept;
  InOutBuffer &operator=(InOutBuffer &&) noexcept;

  explicit InOutBuffer(std::size_t size);

  template <typename T>
  std::shared_ptr<Storage<T>> make_input_storage(std::size_t elems) const {
    assert(elems * sizeof(T) <= _size &&
           "requested size should fit in preallocated space");
    return std::make_shared<Storage<T>>((T *)_inout[0], elems);
  }

  template <typename T>
  std::shared_ptr<Storage<T>>
  make_input_storage(const std::vector<T> &host_data) {
    auto input_storage = make_input_storage<T>(host_data.size());
    CHECK_CUDA(cudaMemcpy(input_storage->data, host_data.data(),
                          host_data.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return input_storage;
  }

  template <typename T, typename U>
  std::shared_ptr<Storage<T>> output_for(const Tensor<U> &input,
                                         std::size_t elems) const {
    assert((_inout[0] == input.storage->data ||
            _inout[1] == input.storage->data) &&
           "provided tensor should use inout buffer");
    assert(elems * sizeof(T) <= _size &&
           "requested size should fit in preallocated space");
    return std::make_shared<Storage<T>>(
        (T *)(input.storage->data == _inout[0] ? _inout[1] : _inout[0]), elems);
  }
};

class Dense {
private:
  std::size_t _in_features;
  std::size_t _out_features;
  bool _use_activation;
  Tensor<__nv_bfloat16> _weight;
  std::optional<Tensor<__nv_bfloat16>> _bias;
  std::shared_ptr<Storage<__nv_bfloat16>> _cache;
  std::size_t _cached_batches = 0;
  std::optional<dim3> _gemv_block_dim;

public:
  Dense(std::size_t in_features, std::size_t out_features, bool use_activation,
        std::size_t cache_size, std::optional<dim3> gemv_block_dim = {});

  std::size_t in_features() const { return _in_features; }
  std::size_t out_features() const { return _out_features; }
  std::size_t cached_batches() const { return _cached_batches; }

  static Dense from_parameters(const Tensor<__nv_bfloat16> &weight,
                               bool use_activation, std::size_t cache_size = 0,
                               std::optional<dim3> gemv_block_dim = {});

  static Dense from_parameters(const Tensor<__nv_bfloat16> &weight,
                               const Tensor<__nv_bfloat16> &bias,
                               bool use_activation, std::size_t cache_size = 0,
                               std::optional<dim3> gemv_block_dim = {});

  Tensor<__nv_bfloat16>
  operator()(LayerContext &ctx, const Tensor<__nv_bfloat16> &input,
             const std::unique_ptr<InOutBuffer> &iobuf = nullptr);

  void rollback(std::size_t previous_cached_batches);
};

class RMSNorm {
private:
  std::size_t _dimensions;
  float _epsilon;
  Tensor<__nv_bfloat16> _weight;

public:
  RMSNorm(std::size_t dimensions, float epsilon);

  std::size_t dimensions() const { return _dimensions; }

  static RMSNorm from_parameter(const Tensor<__nv_bfloat16> &weight,
                                float epsilon);

  Tensor<__nv_bfloat16>
  operator()(LayerContext &ctx, const Tensor<__nv_bfloat16> &input,
             const std::unique_ptr<InOutBuffer> &iobuf = nullptr);
};

class GroupedQueryAttention {
public:
  using RopeBasis = std::pair<Tensor<float>, Tensor<float>>;

private:
  std::size_t _kv_heads;
  std::size_t _groups;
  std::size_t _max_sequence_length;
  int _encoding_base;
  Dense _q_layer;
  Dense _k_layer;
  Dense _v_layer;
  Dense _o_layer;
  Tensor<float> _cos_basis;
  Tensor<float> _sin_basis;

public:
  static RopeBasis make_rope_bases(std::size_t max_sequence_length,
                                   std::size_t head_dimension,
                                   int encoding_base,
                                   cudaStream_t stream = nullptr);

  GroupedQueryAttention(std::size_t kv_heads, std::size_t groups,
                        std::size_t max_sequence_length, int encoding_base,
                        const Dense &q_layer, const Dense &k_layer,
                        const Dense &v_layer, const Dense &o_layer,
                        const RopeBasis &rope_basis);

  std::size_t kv_heads() const { return _kv_heads; }
  std::size_t groups() const { return _groups; }
  std::size_t max_sequence_length() const { return _max_sequence_length; }
  const Dense &q_layer() const { return _q_layer; }
  const Dense &k_layer() const { return _k_layer; }
  const Dense &v_layer() const { return _v_layer; }
  const Dense &o_layer() const { return _o_layer; }

  Tensor<__nv_bfloat16>
  operator()(LayerContext &ctx, const Tensor<__nv_bfloat16> &input_q,
             const Tensor<__nv_bfloat16> &input_k,
             const Tensor<__nv_bfloat16> &input_v, bool causal_mask,
             const std::unique_ptr<InOutBuffer> &iobuf = nullptr);

  void rollback(std::size_t previous_cached_batches);
};

class Qwen2TransformerBlock {
private:
  RMSNorm _input_norm_layer;
  GroupedQueryAttention _attention_layer;
  RMSNorm _post_attention_norm_layer;
  Dense _gate_proj_layer;
  Dense _up_proj_layer;
  Dense _down_proj_layer;

public:
  Qwen2TransformerBlock(const RMSNorm &input_norm_layer,
                        const GroupedQueryAttention &attention_layer,
                        const RMSNorm &post_attention_norm_layer,
                        const Dense &gate_proj_layer,
                        const Dense &up_proj_layer,
                        const Dense &down_proj_layer);

  const RMSNorm &input_norm_layer() const { return _input_norm_layer; }
  const GroupedQueryAttention &attention_layer() const {
    return _attention_layer;
  }

  Tensor<__nv_bfloat16>
  operator()(LayerContext &ctx, const Tensor<__nv_bfloat16> &input,
             const std::unique_ptr<InOutBuffer> &iobuf = nullptr);

  void rollback(std::size_t previous_cached_batches);
};

class Embedding {
private:
  std::size_t _table_size;
  std::size_t _dimension;
  Tensor<__nv_bfloat16> _embedding_table;

public:
  Embedding(std::size_t table_size, std::size_t dimension);

  std::size_t table_size() const { return _table_size; }
  std::size_t dimension() const { return _dimension; }

  static Embedding from_parameter(const Tensor<__nv_bfloat16> &embedding_table);

  Tensor<__nv_bfloat16>
  operator()(LayerContext &ctx, const Tensor<int> &input,
             const std::unique_ptr<InOutBuffer> &iobuf = nullptr);
};

class Sampler {
private:
  std::size_t _vocab_size;
  std::size_t _max_sequence_length;
  std::shared_ptr<Storage<float>> _vals_storage;
  std::shared_ptr<Storage<float>> _vals_storage_next;
  std::shared_ptr<Storage<int>> _indices_storage;
  std::shared_ptr<Storage<int>> _indices_storage_next;
  std::shared_ptr<Storage<int>> _generated_tokens;

public:
  explicit Sampler(std::size_t vocab_size, std::size_t max_sequence_length);

  std::size_t vocab_size() const { return _vocab_size; }
  std::shared_ptr<Storage<int>> generated_tokens() const {
    return _generated_tokens;
  }

  Tensor<int> operator()(LayerContext &ctx, const Tensor<__nv_bfloat16> &logits,
                         const std::unique_ptr<InOutBuffer> &iobuf = nullptr);
};

class LmHeadDense {
private:
  std::size_t _in_features;
  std::size_t _out_features;
  Tensor<__nv_bfloat16> _weight;
  std::optional<dim3> _gemv_block_dim;

public:
  LmHeadDense(std::size_t in_features, std::size_t out_features,
              std::optional<dim3> gemv_block_dim = {});

  std::size_t in_features() const { return _in_features; }
  std::size_t out_features() const { return _out_features; }

  static LmHeadDense from_parameters(const Tensor<__nv_bfloat16> &weight,
                                     std::optional<dim3> gemv_block_dim = {});

  Tensor<__nv_bfloat16>
  operator()(LayerContext &ctx, const Tensor<__nv_bfloat16> &input,
             const std::unique_ptr<InOutBuffer> &iobuf = nullptr);
};
