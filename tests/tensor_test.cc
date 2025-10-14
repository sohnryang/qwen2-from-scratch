#include "tensor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

#include <cuda_bf16.h>

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Pointwise;

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
  EXPECT_THAT(contiguous_tensor.shape, ElementsAre(2, 3, 5, 7));

  // Check content
  auto contiguous_data = contiguous_tensor.storage->to_host();
  ASSERT_EQ(contiguous_data.size(), 2 * 3 * 5 * 7);
  std::vector<float> contiguous_float_data(contiguous_data.size());
  for (size_t i = 0; i < contiguous_data.size(); ++i)
    contiguous_float_data[i] = __bfloat162float(contiguous_data[i]);

  std::vector<float> expected_contiguous_data(2 * 3 * 5 * 7);
  std::iota(expected_contiguous_data.begin(), expected_contiguous_data.end(),
            0.0f);

  EXPECT_THAT(contiguous_float_data,
              Pointwise(FloatEq(), expected_contiguous_data));

  // --- Validate zeros_2x3x5x7 ---
  const auto &zeros_tensor = tensors.at("zeros_2x3x5x7");

  // Check dimensions and shape
  EXPECT_EQ(zeros_tensor.dimensions, 4);
  EXPECT_THAT(zeros_tensor.shape, ElementsAre(2, 3, 5, 7));

  // Check content
  auto zeros_data = zeros_tensor.storage->to_host();
  ASSERT_EQ(zeros_data.size(), 2 * 3 * 5 * 7);
  std::vector<float> zeros_float_data(zeros_data.size());
  for (size_t i = 0; i < zeros_data.size(); ++i)
    zeros_float_data[i] = __bfloat162float(zeros_data[i]);

  EXPECT_THAT(zeros_float_data, Each(FloatEq(0.0f)));
}

TEST(TensorTest, Reshape) {
  std::string filepath =
      std::string(TEST_DATA_DIR) + "/reshape_test.safetensors";
  auto tensors = load_from_safetensors(filepath);
  ASSERT_TRUE(tensors.contains("arange_2x3x4"));
  const auto &tensor = tensors.at("arange_2x3x4");

  // Check original tensor
  EXPECT_EQ(tensor.dimensions, 3);
  EXPECT_THAT(tensor.shape, ElementsAre(2, 3, 4, 0));
  ASSERT_EQ(tensor.storage->elems, 24);

  // --- Test various reshapes ---

  // Reshape to (6, 4)
  auto reshaped_6x4 = tensor.reshape({6, 4});
  EXPECT_EQ(reshaped_6x4.dimensions, 2);
  EXPECT_THAT(reshaped_6x4.shape, ElementsAre(6, 4, 0, 0));
  EXPECT_EQ(reshaped_6x4.storage, tensor.storage); // Should share storage

  // Reshape to (2, 12)
  auto reshaped_2x12 = tensor.reshape({2, 12});
  EXPECT_EQ(reshaped_2x12.dimensions, 2);
  EXPECT_THAT(reshaped_2x12.shape, ElementsAre(2, 12, 0, 0));
  EXPECT_EQ(reshaped_2x12.storage, tensor.storage);

  // Reshape to (24)
  auto reshaped_24 = tensor.reshape({24});
  EXPECT_EQ(reshaped_24.dimensions, 1);
  EXPECT_THAT(reshaped_24.shape, ElementsAre(24, 0, 0, 0));
  EXPECT_EQ(reshaped_24.storage, tensor.storage);

  // Reshape to (2, 3, 2, 2)
  auto reshaped_4d = tensor.reshape({2, 3, 2, 2});
  EXPECT_EQ(reshaped_4d.dimensions, 4);
  EXPECT_THAT(reshaped_4d.shape, ElementsAre(2, 3, 2, 2));
  EXPECT_EQ(reshaped_4d.storage, tensor.storage);

  // --- Test reshape with -1 ---

  // Reshape to (-1, 4)
  auto reshaped_neg1_4 = tensor.reshape({-1, 4});
  EXPECT_EQ(reshaped_neg1_4.dimensions, 2);
  EXPECT_THAT(reshaped_neg1_4.shape, ElementsAre(6, 4, 0, 0));
  EXPECT_EQ(reshaped_neg1_4.storage, tensor.storage);

  // Reshape to (6, -1)
  auto reshaped_6_neg1 = tensor.reshape({6, -1});
  EXPECT_EQ(reshaped_6_neg1.dimensions, 2);
  EXPECT_THAT(reshaped_6_neg1.shape, ElementsAre(6, 4, 0, 0));
  EXPECT_EQ(reshaped_6_neg1.storage, tensor.storage);

  // Reshape to (-1)
  auto reshaped_neg1 = tensor.reshape({-1});
  EXPECT_EQ(reshaped_neg1.dimensions, 1);
  EXPECT_THAT(reshaped_neg1.shape, ElementsAre(24, 0, 0, 0));
  EXPECT_EQ(reshaped_neg1.storage, tensor.storage);

  // Check that data is still correct
  auto reshaped_data = reshaped_6x4.storage->to_host();
  ASSERT_EQ(reshaped_data.size(), 24);
  std::vector<float> float_data(reshaped_data.size());
  for (size_t i = 0; i < reshaped_data.size(); ++i) {
    float_data[i] = __bfloat162float(reshaped_data[i]);
  }

  std::vector<float> expected_data(24);
  std::iota(expected_data.begin(), expected_data.end(), 0.0f);

  EXPECT_THAT(float_data, Pointwise(FloatEq(), expected_data));
}
