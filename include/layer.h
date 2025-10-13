#pragma once

#include "tensor.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

class Dense {
private:
  std::size_t _in_features;
  std::size_t _out_features;
  bool _use_activation;
  Tensor _weight;
  std::optional<Tensor> _bias;

public:
  Dense(std::size_t in_features, std::size_t out_features, bool use_activation);

  std::size_t in_features() { return _in_features; }
  std::size_t out_features() { return _out_features; }

  static Dense from_parameters(const Tensor &weight, bool use_activation);

  static Dense from_parameters(const Tensor &weight, const Tensor &bias,
                               bool use_activation);

  Tensor operator()(const Tensor &input,
                    std::shared_ptr<Storage> out_storage = nullptr);
};

class RMSNorm {
private:
  std::size_t _dimensions;
  float _epsilon;
  Tensor _weight;

public:
  RMSNorm(std::size_t dimensions, float epsilon);

  static RMSNorm from_parameter(const Tensor &weight, float epsilon);

  Tensor operator()(const Tensor &input,
                    std::shared_ptr<Storage> out_storage = nullptr);
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

  Tensor operator()(
      const Tensor &input_q, const Tensor &input_k, const Tensor &input_v,
      std::optional<std::reference_wrapper<const Tensor>> input_mask = {},
      const std::vector<std::shared_ptr<Storage>> &out_storages = {});
};
