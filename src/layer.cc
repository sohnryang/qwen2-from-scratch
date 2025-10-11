#include "layer.h"
#include "tensor.h"

#include <cassert>
#include <cstddef>
#include <memory>

Dense::Dense(std::size_t in_features, std::size_t out_features,
             bool use_activation)
    : _in_features{in_features}, _out_features{out_features},
      _use_activation{use_activation},
      _weight{.shape = {_out_features, _in_features},
              .dimensions = 2,
              .storage =
                  std::make_shared<Storage>(_in_features * _out_features)},
      _bias{} {}

Dense Dense::from_parameters(const Tensor &weight, bool use_activation) {
  assert(weight.dimensions == 2 && "invalid weight dimension");
  const auto in_features = weight.shape[1], out_features = weight.shape[0];
  Dense dense(in_features, out_features, use_activation);
  dense._weight = weight;
  return dense;
}

Dense Dense::from_parameters(const Tensor &weight, const Tensor &bias,
                             bool use_activation) {
  assert(weight.dimensions == 2 && "invalid weight dimension");
  assert(bias.dimensions == 1 && "invalid bias dimension");
  assert(weight.shape[0] == bias.shape[0] && "weight and bias shape mismatch");
  const auto in_features = weight.shape[1], out_features = weight.shape[0];

  Dense dense(in_features, out_features, use_activation);
  dense._weight = weight;
  dense._bias = bias;
  return dense;
}

RMSNorm::RMSNorm(std::size_t dimensions, float epsilon)
    : _dimensions{dimensions}, _epsilon{epsilon},
      _weight{.shape = {_dimensions},
              .dimensions = 1,
              .storage = std::make_shared<Storage>(_dimensions)} {}

RMSNorm RMSNorm::from_parameter(const Tensor &weight, float epsilon) {
  assert(weight.dimensions == 1 && "invalid weight dimension");
  RMSNorm rmsnorm(weight.shape[0], epsilon);
  rmsnorm._weight = weight;
  return rmsnorm;
}
