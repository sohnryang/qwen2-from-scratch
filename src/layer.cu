#include "cuda_utils.h"
#include "kernel.h"
#include "layer.h"
#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <memory>

Tensor Dense::operator()(const Tensor &input,
                         std::shared_ptr<Storage> out_storage) {
  assert(input.shape[input.dimensions - 1] == _in_features &&
         "invalid input dimension");
  const auto batches = input.storage->elems / _in_features;
  const auto out_elems = batches * _out_features;
  if (out_storage)
    assert(out_storage->elems ==
               input.storage->elems / _in_features * _out_features &&
           "invalid output storage size");

  out_storage =
      out_storage ? out_storage : std::make_shared<Storage>(out_elems);
  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks(ceil_div(batches, threads_per_block.x),
                        ceil_div(_out_features, threads_per_block.y));
  if (_use_activation)
    dense<<<num_blocks, threads_per_block>>>(
        out_storage->data, input.storage->data, _weight.storage->data,
        _bias ? _bias->storage->data : nullptr, batches, _in_features,
        _out_features);
  else
    gemm_transposed<<<num_blocks, threads_per_block>>>(
        out_storage->data, input.storage->data, _weight.storage->data,
        _bias ? _bias->storage->data : nullptr, 1.0f, batches, _out_features,
        _in_features);

  Tensor res = {.dimensions = input.dimensions, .storage = out_storage};
  std::copy_n(input.shape.begin(), input.dimensions - 1, res.shape.begin());
  res.shape[input.dimensions - 1] = _out_features;
  return res;
}

Tensor RMSNorm::operator()(const Tensor &input,
                           std::shared_ptr<Storage> out_storage) {
  assert(input.shape[input.dimensions - 1] == _dimensions &&
         "invalid input dimension");
  const auto batches = input.storage->elems / _dimensions;
  if (out_storage)
    assert(out_storage->elems == input.storage->elems &&
           "invalid output storage size");

  out_storage = out_storage ? out_storage
                            : std::make_shared<Storage>(input.storage->elems);

  Tensor reshaped = input.reshape({-1, static_cast<int>(_dimensions)});
  const dim3 threads_per_block(1024);
  const dim3 num_blocks(ceil_div(_dimensions, threads_per_block.x));
  float *out_arr, *intermediate_arr;
  CHECK_CUDA(cudaMalloc(&out_arr, num_blocks.x * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&intermediate_arr, num_blocks.x * sizeof(float)));
  for (int batch = 0; batch < reshaped.shape[0]; batch++) {
    square_sum_reduce<<<num_blocks, threads_per_block,
                        threads_per_block.x * sizeof(float)>>>(
        out_arr, reshaped.storage->data + batch * _dimensions, _dimensions);
    if (num_blocks.x != 1) {
      dim3 num_blocks_reduced(ceil_div(_dimensions, threads_per_block.x));
      while (num_blocks_reduced.x > 1) {
        CHECK_CUDA(cudaMemcpy(intermediate_arr, out_arr,
                              num_blocks_reduced.x * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        square_sum_reduce<<<num_blocks_reduced, threads_per_block,
                            threads_per_block.x * sizeof(float)>>>(
            out_arr, intermediate_arr, num_blocks_reduced.x);
        num_blocks_reduced.x =
            ceil_div(num_blocks_reduced.x, threads_per_block.x);
      }
    }
    float res;
    CHECK_CUDA(
        cudaMemcpy(&res, out_arr, sizeof(float), cudaMemcpyDeviceToHost));

    elementwise_product<<<num_blocks, threads_per_block>>>(
        out_storage->data + batch * _dimensions,
        reshaped.storage->data + batch * _dimensions, _weight.storage->data,
        rsqrtf(res / _dimensions + _epsilon), _dimensions);
  }
  CHECK_CUDA(cudaFree(intermediate_arr));
  CHECK_CUDA(cudaFree(out_arr));

  return {.shape = input.shape,
          .dimensions = input.dimensions,
          .storage = out_storage};
}
