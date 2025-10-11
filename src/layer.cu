#include "cuda_utils.h"
#include "kernel.h"
#include "layer.h"
#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <memory>

Tensor Dense::operator()(const Tensor &input,
                         std::shared_ptr<Storage> out_storage) {
  assert(input.shape[input.dimensions - 1] == in_features &&
         "invalid input dimension");
  const auto batches = input.storage->elems / in_features;
  const auto out_elems = batches * out_features;
  if (out_storage)
    assert(out_storage->elems ==
               input.storage->elems / in_features * out_features &&
           "invalid output storage size");

  out_storage =
      out_storage ? out_storage : std::make_shared<Storage>(out_elems);
  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks(ceil_div(batches, threads_per_block.x),
                        ceil_div(out_features, threads_per_block.y));
  if (use_activation)
    dense<<<num_blocks, threads_per_block>>>(
        out_storage->data, input.storage->data, weight.storage->data,
        bias ? bias->storage->data : nullptr, batches, in_features,
        out_features);
  else
    gemm_transposed<<<num_blocks, threads_per_block>>>(
        out_storage->data, input.storage->data, weight.storage->data,
        bias ? bias->storage->data : nullptr, 1.0f, batches, out_features,
        in_features);

  Tensor res = {.dimensions = input.dimensions, .storage = out_storage};
  std::copy_n(input.shape.begin(), input.dimensions - 1, res.shape.begin());
  res.shape[input.dimensions - 1] = out_features;
  return res;
}
