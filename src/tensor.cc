#include "tensor.h"
#include "cuda_utils.h"

#include <utility>

#include <cuda_bf16.h>

Storage::~Storage() { CHECK_CUDA(cudaFree(data)); }

Storage::Storage(const Storage &other) : elems(other.elems) {
  CHECK_CUDA(cudaMemcpy(data, other.data, sizeof(__nv_bfloat16) * other.elems,
                        cudaMemcpyDeviceToDevice));
}

Storage::Storage(Storage &&other) {
  std::swap(data, other.data);
  std::swap(elems, other.elems);
}

Storage Storage::load_from_offset(const std::uint8_t *buf, std::size_t begin,
                                  std::size_t end) {
  Storage loaded;
  const auto bytes = end - begin;
  loaded.elems = bytes / sizeof(__nv_bfloat16);
  CHECK_CUDA(cudaMalloc((void **)&loaded.data, bytes));
  CHECK_CUDA(
      cudaMemcpy(loaded.data, buf + begin, bytes, cudaMemcpyHostToDevice));
  return loaded;
}
