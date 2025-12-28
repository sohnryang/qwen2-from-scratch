#pragma once

#include "cuda_utils.h"
#include <array>
#include <cassert>
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
  bool owned = true;

  ~Storage();
  Storage() = default;
  Storage(const Storage &) = delete;
  Storage(Storage &&) noexcept;
  Storage &operator=(Storage &&) noexcept;

  explicit Storage(std::size_t elems_);
  Storage(const std::vector<T> &host_data);
  Storage(T *data_, std::size_t elems_);

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

class ScratchPad {
private:
  void *_pool = nullptr;
  std::size_t _size = 0;
  std::size_t _used = 0;

public:
  ~ScratchPad();
  ScratchPad() = default;
  ScratchPad(const ScratchPad &) = delete;
  ScratchPad(ScratchPad &&) noexcept;
  ScratchPad &operator=(ScratchPad &&) noexcept;

  explicit ScratchPad(std::size_t size);

  void reset();

  template <typename T>
  std::shared_ptr<Storage<T>> make_storage(std::size_t elems,
                                           std::size_t align = 256);
};

class ScratchPadScope {
private:
  ScratchPad &_scratchpad;

public:
  ~ScratchPadScope();
  ScratchPadScope() = delete;
  ScratchPadScope(const ScratchPadScope &) = default;
  ScratchPadScope &operator=(const ScratchPadScope &) = delete;

  ScratchPadScope(ScratchPad &scratchpad);
};

class InOutBuffer {
private:
  std::array<void *, 2> _inout;
  std::size_t _size;

public:
  ~InOutBuffer();
  InOutBuffer() = delete;
  InOutBuffer(const InOutBuffer &) = delete;
  InOutBuffer(InOutBuffer &&) noexcept;
  InOutBuffer &operator=(InOutBuffer &&) noexcept;

  explicit InOutBuffer(std::size_t size);

  template <typename T>
  std::shared_ptr<Storage<T>> make_input_storage(std::size_t elems) const {
    assert(elems * sizeof(T) <= _size &&
           "requested size should fit in preallocated space");
    return std::make_shared<Storage<T>>((T *)_inout[0], elems);
  }

  template <typename T>
  std::shared_ptr<Storage<T>>
  make_input_storage(const std::vector<T> &host_data) {
    auto input_storage = make_input_storage<T>(host_data.size());
    CHECK_CUDA(cudaMemcpy(input_storage->data, host_data.data(),
                          host_data.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return input_storage;
  }

  template <typename T, typename U>
  std::shared_ptr<Storage<T>> output_for(const Tensor<U> &input,
                                         std::size_t elems) const {
    assert((_inout[0] == input.storage->data ||
            _inout[1] == input.storage->data) &&
           "provided tensor should use inout buffer");
    assert(elems * sizeof(T) <= _size &&
           "requested size should fit in preallocated space");
    return std::make_shared<Storage<T>>(
        (T *)(input.storage->data == _inout[0] ? _inout[1] : _inout[0]), elems);
  }
};

std::map<std::string, Tensor<__nv_bfloat16>>
load_from_safetensors(const std::string &filename);
