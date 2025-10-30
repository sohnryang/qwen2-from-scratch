#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>

template <typename T> struct Storage {
  T *data = nullptr;
  std::size_t elems = 0;

  ~Storage();
  Storage() = default;
  Storage(const Storage &);
  Storage(Storage &&) noexcept;
  Storage &operator=(const Storage &);
  Storage &operator=(Storage &&) noexcept;

  Storage(std::size_t elems_);
  Storage(const std::vector<T> &host_data);

  static Storage load_from_offset(const std::uint8_t *buf, std::size_t begin,
                                  std::size_t end);

  std::vector<T> to_host();
};

template <typename T> struct Tensor {
  std::array<std::size_t, 4> shape = {0};
  std::size_t dimensions = 0;
  std::shared_ptr<Storage<T>> storage;

  Tensor reshape(std::vector<int> new_shape) const;

  std::size_t elems() const;
};

std::map<std::string, Tensor<__nv_bfloat16>>
load_from_safetensors(const std::string &filename);
