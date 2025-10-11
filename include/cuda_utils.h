#pragma once

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      std::cerr << "[" << __FILE__ << ":" << __LINE__                          \
                << "] CUDA error: " << cudaGetErrorName(status_) << ":"        \
                << cudaGetErrorString(status_);                                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (false)

inline constexpr unsigned int ceil_div(unsigned int x, unsigned int y) {
  return (x + y - 1) / y;
}
