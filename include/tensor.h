#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>

struct Storage {
  __nv_bfloat16 *data = nullptr;
  std::size_t elems = 0;

  ~Storage();
  Storage() = default;
  Storage(const Storage &);
  Storage(Storage &&) noexcept;
  Storage &operator=(const Storage &);
  Storage &operator=(Storage &&) noexcept;

  Storage(std::size_t elems_);

  static Storage load_from_offset(const std::uint8_t *buf, std::size_t begin,
                                  std::size_t end);

  std::vector<__nv_bfloat16> to_host();
};

struct Tensor {
  std::array<std::size_t, 4> shape = {0};
  std::size_t dimensions = 0;
  std::shared_ptr<Storage> storage;

  Tensor reshape(std::vector<int> new_shape) const;
};

std::map<std::string, Tensor>
load_from_safetensors(const std::string &filename);
