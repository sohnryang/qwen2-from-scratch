#include "cuda_utils.h"
#include "kernel.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <vector>

TEST(KernelTest, Gemm) {
  auto tensors = load_from_safetensors(std::string(TEST_DATA_DIR) +
                                       "/matmul_test.safetensors");
  const auto &in_a = tensors.at("in_a");
  const auto &in_b = tensors.at("in_b");
  const auto &bias = tensors.at("bias");
  const auto &expected_out = tensors.at("out");

  auto storage = std::make_shared<Storage>(in_a.shape[0] * in_b.shape[1]);
  Tensor actual_out{.shape = {in_a.shape[0], in_b.shape[1]},
                    .dimensions = 2,
                    .storage = storage};

  launch_gemm(actual_out, in_a, in_b, bias, 1.0);

  auto actual_host_data = actual_out.storage->to_host();
  auto expected_host_data = expected_out.storage->to_host();

  ASSERT_EQ(actual_host_data.size(), expected_host_data.size());
  for (size_t i = 0; i < actual_host_data.size(); ++i) {
    EXPECT_FLOAT_EQ(__bfloat162float(actual_host_data[i]),
                    __bfloat162float(expected_host_data[i]));
  }
}

TEST(KernelTest, GemmTransposed) {
  auto tensors = load_from_safetensors(std::string(TEST_DATA_DIR) +
                                       "/matmul_test.safetensors");
  const auto &in_a = tensors.at("in_a");
  const auto &in_b_transposed = tensors.at("in_b_transposed");
  const auto &bias = tensors.at("bias");
  const auto &expected_out = tensors.at("out");

  auto storage =
      std::make_shared<Storage>(in_a.shape[0] * in_b_transposed.shape[0]);
  Tensor actual_out{.shape = {in_a.shape[0], in_b_transposed.shape[0]},
                    .dimensions = 2,
                    .storage = storage};

  launch_gemm(actual_out, in_a, in_b_transposed, bias, 1.0, true);

  auto actual_host_data = actual_out.storage->to_host();
  auto expected_host_data = expected_out.storage->to_host();

  ASSERT_EQ(actual_host_data.size(), expected_host_data.size());
  for (size_t i = 0; i < actual_host_data.size(); ++i) {
    EXPECT_FLOAT_EQ(__bfloat162float(actual_host_data[i]),
                    __bfloat162float(expected_host_data[i]));
  }
}

TEST(KernelTest, SquareSumReduce) {
  const size_t num_elements = 2080; // Not a power of 2, and > 1024
  std::vector<__nv_bfloat16> host_data(num_elements);
  float expected_sum = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    const auto elem = i % 2 ? 1.0f : 0.5f;
    host_data[i] = __float2bfloat16(elem);
    expected_sum += elem * elem;
  }

  auto storage = std::make_shared<Storage>(num_elements);
  CHECK_CUDA(cudaMemcpy(storage->data, host_data.data(),
                        num_elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  Tensor input_tensor{
      .shape = {num_elements}, .dimensions = 1, .storage = storage};

  float actual_sum = launch_square_sum_reduce(input_tensor);
  EXPECT_FLOAT_EQ(actual_sum, expected_sum);
}
