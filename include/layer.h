#pragma once

#include "tensor.h"

#include <cstddef>
#include <memory>
#include <optional>

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
