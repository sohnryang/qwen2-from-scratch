cmake_minimum_required(VERSION 3.20)
project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add executable
add_executable(app src/main.cpp src/kernel.cu)

target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find CUDA and link to it
find_package(CUDAToolkit REQUIRED)
target_link_libraries(app PRIVATE CUDA::cudart)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(unit_tests tests/kernel_test.cpp src/kernel.cu)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(unit_tests PRIVATE gtest_main CUDA::cudart)

include(GoogleTest)
gtest_discover_tests(unit_tests)
