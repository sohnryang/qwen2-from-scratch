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

public:
  Dense(std::size_t in_features, std::size_t out_features, bool use_activation);

  std::size_t in_features() { return _in_features; }
  std::size_t out_features() { return _out_features; }

  static Dense from_parameters(const Tensor<__nv_bfloat16> &weight,
                               bool use_activation);

  static Dense from_parameters(const Tensor<__nv_bfloat16> &weight,
                               const Tensor<__nv_bfloat16> &bias,
                               bool use_activation);

  Tensor<__nv_bfloat16>
  operator()(const Tensor<__nv_bfloat16> &input,
             std::shared_ptr<Storage<__nv_bfloat16>> out_storage = nullptr);
};

class RMSNorm {
private:
  std::size_t _dimensions;
  float _epsilon;
  Tensor<__nv_bfloat16> _weight;

public:
  RMSNorm(std::size_t dimensions, float epsilon);

  static RMSNorm from_parameter(const Tensor<__nv_bfloat16> &weight,
                                float epsilon);

  Tensor<__nv_bfloat16>
  operator()(const Tensor<__nv_bfloat16> &input,
             std::shared_ptr<Storage<__nv_bfloat16>> out_storage = nullptr);
};

class GroupedQueryAttention {
private:
  std::size_t _kv_heads;
  std::size_t _groups;
  Dense _q_layer;
  Dense _k_layer;
  Dense _v_layer;
  Dense _o_layer;

public:
  GroupedQueryAttention(std::size_t kv_heads, std::size_t groups,
                        const Dense &q_layer, const Dense &k_layer,
                        const Dense &v_layer, const Dense &o_layer);

  Tensor<__nv_bfloat16> operator()(
      const Tensor<__nv_bfloat16> &input_q,
      const Tensor<__nv_bfloat16> &input_k,
      const Tensor<__nv_bfloat16> &input_v, bool causal_mask = false,
      std::shared_ptr<Storage<__nv_bfloat16>> q_proj_out_storage = nullptr,
      std::shared_ptr<Storage<__nv_bfloat16>> k_proj_out_storage = nullptr,
      std::shared_ptr<Storage<__nv_bfloat16>> v_proj_out_storage = nullptr,
      std::shared_ptr<Storage<float>> scores_out_storage = nullptr,
      std::shared_ptr<Storage<__nv_bfloat16>> attention_out_storage = nullptr,
      std::shared_ptr<Storage<__nv_bfloat16>> o_proj_out_storage = nullptr);
};
