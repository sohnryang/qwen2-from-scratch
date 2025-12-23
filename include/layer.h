#pragma once

#include "tensor.h"

#include <cstddef>
#include <memory>
#include <optional>

#include <cuda_bf16.h>

class Dense {
private:
  std::size_t _in_features;
  std::size_t _out_features;
  bool _use_activation;
  Tensor<__nv_bfloat16> _weight;
  std::optional<Tensor<__nv_bfloat16>> _bias;
  std::shared_ptr<Storage<__nv_bfloat16>> _cache;
  std::size_t _cached_batches = 0;

public:
  Dense(std::size_t in_features, std::size_t out_features, bool use_activation,
        std::size_t cache_size);

  std::size_t in_features() const { return _in_features; }
  std::size_t out_features() const { return _out_features; }
  std::size_t cached_batches() const { return _cached_batches; }

  static Dense from_parameters(const Tensor<__nv_bfloat16> &weight,
                               bool use_activation, std::size_t cache_size = 0);

  static Dense from_parameters(const Tensor<__nv_bfloat16> &weight,
                               const Tensor<__nv_bfloat16> &bias,
                               bool use_activation, std::size_t cache_size = 0);

  Tensor<__nv_bfloat16> operator()(const Tensor<__nv_bfloat16> &input);
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

  Tensor<__nv_bfloat16> operator()(const Tensor<__nv_bfloat16> &input);
};

class GroupedQueryAttention {
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
  std::shared_ptr<Storage<__nv_bfloat16>> _attention_out_storage;

public:
  GroupedQueryAttention(std::size_t kv_heads, std::size_t groups,
                        std::size_t max_sequence_length, int encoding_base,
                        const Dense &q_layer, const Dense &k_layer,
                        const Dense &v_layer, const Dense &o_layer);

  std::size_t max_sequence_length() const { return _max_sequence_length; }
  const Dense &q_layer() const { return _q_layer; }
  const Dense &k_layer() const { return _k_layer; }
  const Dense &v_layer() const { return _v_layer; }
  const Dense &o_layer() const { return _o_layer; }

  Tensor<__nv_bfloat16> operator()(const Tensor<__nv_bfloat16> &input_q,
                                   const Tensor<__nv_bfloat16> &input_k,
                                   const Tensor<__nv_bfloat16> &input_v,
                                   bool causal_mask = false);
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

  Tensor<__nv_bfloat16> operator()(const Tensor<__nv_bfloat16> &input,
                                   bool causal_mask = true);
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

  Tensor<__nv_bfloat16> operator()(const Tensor<int> &input);
};

class Sampler {
private:
  std::size_t _vocab_size;
  std::shared_ptr<Storage<float>> _vals_storage;
  std::shared_ptr<Storage<float>> _vals_storage_next;
  std::shared_ptr<Storage<int>> _indices_storage;
  std::shared_ptr<Storage<int>> _indices_storage_next;

public:
  explicit Sampler(std::size_t vocab_size);

  std::size_t vocab_size() const { return _vocab_size; }

  Tensor<int> operator()(const Tensor<__nv_bfloat16> &logits);
};

class LmHeadDense {
private:
  std::size_t _in_features;
  std::size_t _out_features;
  Tensor<__nv_bfloat16> _weight;

public:
  LmHeadDense(std::size_t in_features, std::size_t out_features);

  std::size_t in_features() const { return _in_features; }
  std::size_t out_features() const { return _out_features; }

  static LmHeadDense from_parameters(const Tensor<__nv_bfloat16> &weight);

  Tensor<__nv_bfloat16> operator()(const Tensor<__nv_bfloat16> &input);
};
