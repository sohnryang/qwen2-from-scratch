#include "layer.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
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

static void assert_tensors_near(const Tensor<__nv_bfloat16> &actual,
                                const Tensor<__nv_bfloat16> &expected,
                                float abs_error = 4e-2) {
  ASSERT_EQ(actual.dimensions, expected.dimensions);
  for (std::size_t i = 0; i < actual.dimensions; ++i) {
    ASSERT_EQ(actual.shape[i], expected.shape[i]);
  }
  auto actual_host_data = actual.storage->to_host();
  auto expected_host_data = expected.storage->to_host();

  ASSERT_EQ(actual_host_data.size(), expected_host_data.size());
  for (size_t i = 0; i < actual_host_data.size(); ++i)
    EXPECT_NEAR(__bfloat162float(actual_host_data[i]),
                __bfloat162float(expected_host_data[i]), abs_error)
        << "at index " << i;
}

TEST(LayerTest, DenseNoActivation) {
  auto tensors = load_from_safetensors(std::string(TEST_DATA_DIR) +
                                       "/matmul_test.safetensors");
  const auto &input = tensors.at("in_a");
  const auto &weight = tensors.at("in_b_transposed");
  const auto &bias = tensors.at("bias");
  const auto &expected_out = tensors.at("out");
  const auto max_sequence_length = input.elems() / weight.shape[1];

  Dense dense_layer =
      Dense::from_parameters(weight, bias, false, max_sequence_length);

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
  const auto max_sequence_length = input.elems() / weight.shape[1];

  Dense dense_layer =
      Dense::from_parameters(weight, bias, true, max_sequence_length);

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
  const auto max_sequence_length = input.elems() / weight.shape[0];

  RMSNorm norm_layer =
      RMSNorm::from_parameter(weight, epsilon, max_sequence_length);

  Tensor<__nv_bfloat16> actual_out = norm_layer(input);

  assert_tensors_equal(actual_out, expected_out);
}

TEST(LayerTest, Qwen2TransformerBlock) {
  auto tensors = load_from_safetensors(std::string(TEST_DATA_DIR) +
                                       "/transformer_block_test.safetensors");
  const auto &input = tensors.at("in");
  const auto &expected_out = tensors.at("out");

  const float rms_norm_eps = 1e-6;
  const auto input_sequence_length = input.elems() / input.shape[1];
  RMSNorm input_norm_layer = RMSNorm::from_parameter(
      tensors.at("input_norm_weight"), rms_norm_eps, input_sequence_length);
  RMSNorm post_attention_norm_layer = RMSNorm::from_parameter(
      tensors.at("post_norm_weight"), rms_norm_eps, input_sequence_length);

  const std::size_t num_kv_heads = 2;
  const std::size_t groups = 6; // 12 heads / 2 kv_heads
  const int encoding_base = 1000000;

  Dense q_layer =
      Dense::from_parameters(tensors.at("q_weight"), tensors.at("q_bias"),
                             false, input_sequence_length);
  Dense k_layer =
      Dense::from_parameters(tensors.at("k_weight"), tensors.at("k_bias"),
                             false, input_sequence_length);
  Dense v_layer =
      Dense::from_parameters(tensors.at("v_weight"), tensors.at("v_bias"),
                             false, input_sequence_length);
  Dense o_layer = Dense::from_parameters(tensors.at("o_weight"), false,
                                         input_sequence_length);
  GroupedQueryAttention attention_layer(num_kv_heads, groups,
                                        input_sequence_length, encoding_base,
                                        q_layer, k_layer, v_layer, o_layer);
  Dense gate_proj_layer = Dense::from_parameters(tensors.at("gate_proj_weight"),
                                                 true, input_sequence_length);
  Dense up_proj_layer = Dense::from_parameters(tensors.at("up_proj_weight"),
                                               false, input_sequence_length);
  Dense down_proj_layer = Dense::from_parameters(tensors.at("down_proj_weight"),
                                                 false, input_sequence_length);
  Qwen2TransformerBlock transformer_block(
      input_norm_layer, attention_layer, post_attention_norm_layer,
      gate_proj_layer, up_proj_layer, down_proj_layer);

  Tensor<__nv_bfloat16> actual_out = transformer_block(input, true);

  assert_tensors_near(actual_out, expected_out);
}

TEST(LayerTest, Embedding) {
  auto tensors = load_from_safetensors(std::string(TEST_DATA_DIR) +
                                       "/embedding_test.safetensors");
  const auto &embedding_table = tensors.at("embedding_table");
  const auto &expected_out = tensors.at("out");
  const auto &input_bf16 = tensors.at("input");

  auto input_bf16_host = input_bf16.storage->to_host();
  std::vector<int> input_int_host(input_bf16_host.size());
  for (size_t i = 0; i < input_bf16_host.size(); ++i)
    input_int_host[i] = static_cast<int>(__bfloat162float(input_bf16_host[i]));

  const auto input =
      Tensor{.shape = input_bf16.shape,
             .dimensions = input_bf16.dimensions,
             .storage = std::make_shared<Storage<int>>(input_int_host)};
  const auto max_sequence_length = input.elems();

  Embedding embedding_layer =
      Embedding::from_parameter(embedding_table, max_sequence_length);

  Tensor<__nv_bfloat16> actual_out = embedding_layer(input);

  assert_tensors_equal(actual_out, expected_out);
}
