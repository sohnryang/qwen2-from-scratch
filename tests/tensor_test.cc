#include "cuda_utils.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

// Helper function to copy data from device to host
std::vector<__nv_bfloat16> get_tensor_data(const Tensor &tensor) {
  if (!tensor.storage) {
    return {};
  }
  std::vector<__nv_bfloat16> host_data(tensor.storage->elems);
  CHECK_CUDA(cudaMemcpy(host_data.data(), tensor.storage->data,
                        tensor.storage->elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  return host_data;
}

TEST(TensorTest, LoadSafetensors) {
  std::string filepath = std::string(TEST_DATA_DIR) + "/load_test.safetensors";
  auto tensors = load_from_safetensors(filepath);

  // Check for tensor existence
  ASSERT_TRUE(tensors.contains("contiguous_2x3x5x7"));
  ASSERT_TRUE(tensors.contains("zeros_2x3x5x7"));

  // --- Validate contiguous_2x3x5x7 ---
  const auto &contiguous_tensor = tensors.at("contiguous_2x3x5x7");

  // Check dimensions and shape
  EXPECT_EQ(contiguous_tensor.dimensions, 4);
  EXPECT_EQ(contiguous_tensor.shape[0], 2);
  EXPECT_EQ(contiguous_tensor.shape[1], 3);
  EXPECT_EQ(contiguous_tensor.shape[2], 5);
  EXPECT_EQ(contiguous_tensor.shape[3], 7);

  // Check content
  auto contiguous_data = get_tensor_data(contiguous_tensor);
  ASSERT_EQ(contiguous_data.size(), 2 * 3 * 5 * 7);
  std::vector<float> contiguous_float_data(contiguous_data.size());
  for (size_t i = 0; i < contiguous_data.size(); ++i)
    contiguous_float_data[i] = __bfloat162float(contiguous_data[i]);

  std::vector<float> expected_contiguous_data(2 * 3 * 5 * 7);
  std::iota(expected_contiguous_data.begin(), expected_contiguous_data.end(),
            0.0f);

  for (size_t i = 0; i < contiguous_float_data.size(); ++i)
    EXPECT_FLOAT_EQ(contiguous_float_data[i], expected_contiguous_data[i]);

  // --- Validate zeros_2x3x5x7 ---
  const auto &zeros_tensor = tensors.at("zeros_2x3x5x7");

  // Check dimensions and shape
  EXPECT_EQ(zeros_tensor.dimensions, 4);
  EXPECT_EQ(zeros_tensor.shape[0], 2);
  EXPECT_EQ(zeros_tensor.shape[1], 3);
  EXPECT_EQ(zeros_tensor.shape[2], 5);
  EXPECT_EQ(zeros_tensor.shape[3], 7);

  // Check content
  auto zeros_data = get_tensor_data(zeros_tensor);
  ASSERT_EQ(zeros_data.size(), 2 * 3 * 5 * 7);
  std::vector<float> zeros_float_data(zeros_data.size());
  for (size_t i = 0; i < zeros_data.size(); ++i)
    zeros_float_data[i] = __bfloat162float(zeros_data[i]);

  for (size_t i = 0; i < zeros_float_data.size(); ++i)
    EXPECT_FLOAT_EQ(zeros_float_data[i], 0.0f);
}
