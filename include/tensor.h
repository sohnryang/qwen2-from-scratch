#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_bf16.h>

struct Storage {
  __nv_bfloat16 *data = nullptr;
  std::size_t elems = 0;

  ~Storage();
  Storage() = default;
  Storage(const Storage &);
  Storage(Storage &&);
  Storage &operator=(const Storage &) = default;
  Storage &operator=(Storage &&) = default;

  static Storage load_from_offset(const std::uint8_t *buf, std::size_t begin,
                                  std::size_t end);
};

struct Tensor {
  std::size_t shape[4];
  std::size_t dimensions;
  std::shared_ptr<Storage> storage;
};
