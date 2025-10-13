#include "layer.h"
#include "tensor.h"

#include <gtest/gtest.h>
#include <vector>

static void assert_tensors_equal(const Tensor &actual, const Tensor &expected) {
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

  Tensor actual_out = dense_layer(input);

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

  Tensor actual_out = dense_layer(input);

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

  Tensor actual_out = norm_layer(input);

  assert_tensors_equal(actual_out, expected_out);
}

TEST(LayerTest, GroupedQueryAttention) {
  auto tensors = load_from_safetensors(std::string(TEST_DATA_DIR) +
                                       "/gqa_test.safetensors");
  const auto &q_weight = tensors.at("q_weight");
  const auto &k_weight = tensors.at("k_weight");
  const auto &v_weight = tensors.at("v_weight");
  const auto &o_weight = tensors.at("o_weight");
  const auto &expected_out = tensors.at("o_proj");

  const std::size_t num_kv_heads = 2;
  const std::size_t groups = 4;

  Dense q_layer = Dense::from_parameters(q_weight, false);
  Dense k_layer = Dense::from_parameters(k_weight, false);
  Dense v_layer = Dense::from_parameters(v_weight, false);
  Dense o_layer = Dense::from_parameters(o_weight, false);

  GroupedQueryAttention gqa_layer(num_kv_heads, groups, q_layer, k_layer,
                                  v_layer, o_layer);

  const auto &q_in = tensors.at("q_in");
  const auto &k_in = tensors.at("k_in");
  const auto &v_in = tensors.at("v_in");
  Tensor actual_out = gqa_layer(q_in, k_in, v_in);

  assert_tensors_equal(actual_out, expected_out);
}
