#include "layer.h"

#include <cassert>
#include <cstddef>

Dense::Dense(std::size_t in_features_, std::size_t out_features_,
             bool use_activation_)
    : in_features{in_features_}, out_features{out_features_},
      use_activation{use_activation_},
      weight{.shape = {out_features, in_features},
             .dimensions = 2,
             .storage = std::make_shared<Storage>(in_features * out_features)},
      bias{} {}

Dense Dense::from_parameters(const Tensor &weight, bool use_activation) {
  assert(weight.dimensions == 2 && "invalid weight dimension");
  const auto in_features = weight.shape[1], out_features = weight.shape[0];

  Dense dense(in_features, out_features, use_activation);
  dense.weight = weight;
  return dense;
}

Dense Dense::from_parameters(const Tensor &weight, const Tensor &bias,
                             bool use_activation) {
  assert(weight.dimensions == 2 && "invalid weight dimension");
  assert(bias.dimensions == 1 && "invalid bias dimension");
  assert(weight.shape[0] == bias.shape[0] && "weight and bias shape mismatch");
  const auto in_features = weight.shape[1], out_features = weight.shape[0];

  Dense dense(in_features, out_features, use_activation);
  dense.weight = weight;
  dense.bias = bias;
  return dense;
}
