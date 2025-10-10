#pragma once

#include "tensor.h"

#include <cstddef>
#include <optional>

class Dense {
private:
  std::size_t in_features;
  std::size_t out_features;
  bool use_activation;
  Tensor weight;
  std::optional<Tensor> bias;

public:
  Dense(std::size_t in_features_, std::size_t out_features_,
        bool use_activation_);

  static Dense from_parameters(const Tensor &weight, bool use_activation);

  static Dense from_parameters(const Tensor &weight, const Tensor &bias,
                               bool use_activation);

  Tensor operator()(const Tensor &input,
                    std::shared_ptr<Storage> out_storage = nullptr);
};
