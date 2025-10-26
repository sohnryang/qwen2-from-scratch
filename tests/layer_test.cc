#include "layer.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <vector>

#include <cuda_bf16.h>

static void assert_tensors_equal(const Tensor<__nv_bfloat16> &actual,
                                 const Tensor<__nv_bfloat16> &expected) {
  ASSERT_EQ(actual.dimensions, expected.dimensions);
  for (std::size_t i = 0; i < actual.dimensions; ++i) {
    ASSERT_EQ(actual.shape[i], expected.shape[i]);
  }
  auto actual_host_data = actual.storage->to_host();
  auto expected_host_data = expected.storage->to_host();

  ASSERT_EQ(actual_host_data.size(), expected_host_data.size());
  for (size_t i = 0; i < actual_host_data.size(); ++i)
    EXPECT_FLOAT_EQ(__bfloat162float(actual_host_data[i]),
                    __bfloat162float(expected_host_data[i]))
        << "at index " << i;
}

TEST(LayerTest, DenseNoActivation) {
  auto tensors = load_from_safetensors(std::string(TEST_DATA_DIR) +
                                       "/matmul_test.safetensors");
  const auto &input = tensors.at("in_a");
  const auto &weight = tensors.at("in_b_transposed");
  const auto &bias = tensors.at("bias");
  const auto &expected_out = tensors.at("out");

  Dense dense_layer = Dense::from_parameters(weight, bias, false);

  Tensor<__nv_bfloat16> actual_out = dense_layer(input);

  assert_tensors_equal(actual_out, expected_out);
}

TEST(LayerTest, DenseWithActivation) {
  auto tensors = load_from_safetensors(std::string(TEST_DATA_DIR) +
                                       "/dense_test.safetensors");
  const auto &input = tensors.at("x");
  const auto &weight = tensors.at("weight");
  const auto &bias = tensors.at("bias");
  const auto &expected_out = tensors.at("out");

  Dense dense_layer = Dense::from_parameters(weight, bias, true);

  Tensor<__nv_bfloat16> actual_out = dense_layer(input);

  assert_tensors_equal(actual_out, expected_out);
}

TEST(LayerTest, RMSNorm) {
  auto tensors = load_from_safetensors(std::string(TEST_DATA_DIR) +
                                       "/rmsnorm_test.safetensors");
  const auto &input = tensors.at("x");
  const auto &weight = tensors.at("weight");
  const auto &expected_out = tensors.at("out");
  const float epsilon = 1e-5;

  RMSNorm norm_layer = RMSNorm::from_parameter(weight, epsilon);

  Tensor<__nv_bfloat16> actual_out = norm_layer(input);

  assert_tensors_equal(actual_out, expected_out);
}

