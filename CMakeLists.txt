cmake_minimum_required(VERSION 3.20)
project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find external packages
find_package(CUDAToolkit REQUIRED)
include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
FetchContent_MakeAvailable(googletest)

# simdjson
FetchContent_Declare(
  simdjson
  URL https://github.com/simdjson/simdjson/archive/refs/tags/v4.0.7.zip)
FetchContent_MakeAvailable(simdjson)

# tokenizers-cpp
FetchContent_Declare(
  tokenizers-cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp
  GIT_TAG 55d53aa38dc8df7d9c8bd9ed50907e82ae83ce66)
FetchContent_MakeAvailable(tokenizers-cpp)

# Kernel library
set(KERNEL_SRCS src/tensor.cc src/kernel.cu src/layer.cc src/layer.cu
                src/model.cc)
add_library(qwen2kern STATIC ${KERNEL_SRCS})
target_include_directories(qwen2kern PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qwen2kern PUBLIC CUDA::cudart)
target_link_libraries(qwen2kern PRIVATE simdjson)
set_property(TARGET qwen2kern PROPERTY CUDA_ARCHITECTURES 89)

# Main executable
add_executable(qwen2chat src/main.cc)
target_link_libraries(
  qwen2chat
  PRIVATE qwen2kern
  PRIVATE tokenizers_cpp)

# Unit tests
enable_testing()
add_executable(unit_tests tests/kernel_test.cc tests/tensor_test.cc
                          tests/layer_test.cc)
target_link_libraries(unit_tests PRIVATE gtest_main gmock qwen2kern)
target_compile_definitions(
  unit_tests PRIVATE "TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data\"")
include(GoogleTest)
gtest_discover_tests(unit_tests)
