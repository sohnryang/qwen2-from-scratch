#include "cuda_utils.h"
#include "layer.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <vector>

#include <cuda_bf16.h>

namespace fs = std::filesystem;

static void assert_tensors_equal(const Tensor<__nv_bfloat16> &actual,
                                 const Tensor<__nv_bfloat16> &expected) {
  ASSERT_EQ(actual.dimensions, expected.dimensions);
  for (std::size_t i = 0; i < actual.dimensions; ++i)
    ASSERT_EQ(actual.shape[i], expected.shape[i]);

  auto actual_host_data = actual.storage->to_host();
  auto expected_host_data = expected.storage->to_host();

  for (size_t i = 0; i < expected.elems(); ++i)
    EXPECT_FLOAT_EQ(__bfloat162float(actual_host_data[i]),
                    __bfloat162float(expected_host_data[i]))
        << "at index " << i;
}

static void assert_tensors_near(const Tensor<__nv_bfloat16> &actual,
                                const Tensor<__nv_bfloat16> &expected,
                                float abs_error = 4e-2) {
  ASSERT_EQ(actual.dimensions, expected.dimensions);
  for (std::size_t i = 0; i < actual.dimensions; ++i)
    ASSERT_EQ(actual.shape[i], expected.shape[i]);

  auto actual_host_data = actual.storage->to_host();
  auto expected_host_data = expected.storage->to_host();

  for (size_t i = 0; i < expected.elems(); ++i)
    EXPECT_NEAR(__bfloat162float(actual_host_data[i]),
                __bfloat162float(expected_host_data[i]), abs_error)
        << "at index " << i;
}

TEST(LayerTest, DenseNoActivation) {
  auto tensors = load_from_safetensors(
      (fs::path(TEST_DATA_DIR) / "matmul_test.safetensors").string());
  const auto &input = tensors.at("in_a");
  const auto &weight = tensors.at("in_b_transposed");
  const auto &bias = tensors.at("bias");
  const auto &expected_out = tensors.at("out");

  Dense dense_layer = Dense::from_parameters(weight, bias, false);

  LayerContext ctx(1);
  Tensor<__nv_bfloat16> actual_out = dense_layer(ctx, input);

  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  assert_tensors_equal(actual_out, expected_out);
}

TEST(LayerTest, DenseWithActivation) {
  auto tensors = load_from_safetensors(
      (fs::path(TEST_DATA_DIR) / "dense_test.safetensors").string());
  const auto &input = tensors.at("x");
  const auto &weight = tensors.at("weight");
  const auto &bias = tensors.at("bias");
  const auto &expected_out = tensors.at("out");

  Dense dense_layer = Dense::from_parameters(weight, bias, true);

  LayerContext ctx(1);
  Tensor<__nv_bfloat16> actual_out = dense_layer(ctx, input);

  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  assert_tensors_equal(actual_out, expected_out);
}

TEST(LayerTest, DenseCache) {
  auto tensors = load_from_safetensors(
      (fs::path(TEST_DATA_DIR) / "dense_cache_test.safetensors").string());
  const auto &weight = tensors.at("weight");
  const auto &bias = tensors.at("bias");
  const auto &input_a = tensors.at("input_a");
  const auto &input_b = tensors.at("input_b");
  const auto &expected_a = tensors.at("expected_a");
  const auto &expected_cached = tensors.at("expected_cached");

  Dense dense_layer = Dense::from_parameters(weight, bias, false, 4);

  LayerContext ctx(1);
  Tensor<__nv_bfloat16> actual_a = dense_layer(ctx, input_a);
  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  assert_tensors_equal(actual_a, expected_a);
  EXPECT_EQ(dense_layer.cached_batches(), expected_a.shape[0]);

  Tensor<__nv_bfloat16> actual_cached = dense_layer(ctx, input_b);
  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  assert_tensors_equal(actual_cached, expected_cached);
  EXPECT_EQ(dense_layer.cached_batches(), expected_cached.shape[0]);
}

TEST(LayerTest, RMSNorm) {
  auto tensors = load_from_safetensors(
      (fs::path(TEST_DATA_DIR) / "rmsnorm_test.safetensors").string());
  const auto &input = tensors.at("x");
  const auto &weight = tensors.at("weight");
  const auto &expected_out = tensors.at("out");
  const float epsilon = 1e-5;

  RMSNorm norm_layer = RMSNorm::from_parameter(weight, epsilon);

  LayerContext ctx(1);
  Tensor<__nv_bfloat16> actual_out = norm_layer(ctx, input);

  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  assert_tensors_equal(actual_out, expected_out);
}

TEST(LayerTest, Qwen2TransformerBlock) {
  auto tensors = load_from_safetensors(
      (fs::path(TEST_DATA_DIR) / "transformer_block_test.safetensors")
          .string());
  const auto &input = tensors.at("in");
  const auto &expected_out = tensors.at("out");
  const auto &input_next = tensors.at("in_next");
  const auto &expected_out_next = tensors.at("out_next");

  const float rms_norm_eps = 1e-6;
  const auto max_sequence_length =
      (input.elems() + input_next.elems()) / input.shape[1];
  RMSNorm input_norm_layer =
      RMSNorm::from_parameter(tensors.at("input_norm_weight"), rms_norm_eps);
  RMSNorm post_attention_norm_layer =
      RMSNorm::from_parameter(tensors.at("post_norm_weight"), rms_norm_eps);

  const std::size_t num_kv_heads = 2;
  const std::size_t groups = 6; // 12 heads / 2 kv_heads
  const int encoding_base = 1000000;

  LayerContext ctx(1);
  Dense q_layer = Dense::from_parameters(tensors.at("q_weight"),
                                         tensors.at("q_bias"), false);
  Dense k_layer = Dense::from_parameters(
      tensors.at("k_weight"), tensors.at("k_bias"), false, max_sequence_length);
  Dense v_layer = Dense::from_parameters(
      tensors.at("v_weight"), tensors.at("v_bias"), false, max_sequence_length);
  Dense o_layer = Dense::from_parameters(tensors.at("o_weight"), false);
  const auto head_dimension = k_layer.out_features() / num_kv_heads;
  const auto rope_basis = GroupedQueryAttention::make_rope_bases(
      max_sequence_length, head_dimension, encoding_base, ctx.stream());
  GroupedQueryAttention attention_layer(
      num_kv_heads, groups, max_sequence_length, encoding_base, q_layer,
      k_layer, v_layer, o_layer, rope_basis);
  Dense gate_proj_layer =
      Dense::from_parameters(tensors.at("gate_proj_weight"), true);
  Dense up_proj_layer =
      Dense::from_parameters(tensors.at("up_proj_weight"), false);
  Dense down_proj_layer =
      Dense::from_parameters(tensors.at("down_proj_weight"), false);
  Qwen2TransformerBlock transformer_block(
      input_norm_layer, attention_layer, post_attention_norm_layer,
      gate_proj_layer, up_proj_layer, down_proj_layer);

  Tensor<__nv_bfloat16> actual_out = transformer_block(ctx, input);
  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  assert_tensors_near(actual_out, expected_out);

  Tensor<__nv_bfloat16> actual_out_next = transformer_block(ctx, input_next);
  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  assert_tensors_near(actual_out_next, expected_out_next);
}

TEST(LayerTest, Embedding) {
  auto tensors = load_from_safetensors(
      (fs::path(TEST_DATA_DIR) / "embedding_test.safetensors").string());
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

  Embedding embedding_layer = Embedding::from_parameter(embedding_table);

  LayerContext ctx(1);
  Tensor<__nv_bfloat16> actual_out = embedding_layer(ctx, input);

  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  assert_tensors_equal(actual_out, expected_out);
}

TEST(LayerTest, Sampler) {
  auto tensors = load_from_safetensors(
      (fs::path(TEST_DATA_DIR) / "sampler_test.safetensors").string());
  const auto &logits = tensors.at("logits");
  const auto &expected_out = tensors.at("expected");

  if (logits.dimensions > 1)
    ASSERT_EQ(logits.shape[0], 1U);

  Sampler sampler(logits.shape[1], 1);

  LayerContext ctx(1);
  Tensor<int> actual_out = sampler(ctx, logits);

  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  ASSERT_EQ(actual_out.dimensions, expected_out.dimensions);
  ASSERT_EQ(actual_out.shape[0], expected_out.shape[0]);
  auto actual_host = actual_out.storage->to_host();
  auto expected_host = expected_out.storage->to_host();
  ASSERT_EQ(actual_host.size(), expected_host.size());
  for (size_t i = 0; i < actual_host.size(); ++i) {
    EXPECT_EQ(actual_host[i],
              static_cast<int>(__bfloat162float(expected_host[i])))
        << "at index " << i;
  }
}

TEST(LayerTest, LmHeadDense) {
  auto tensors = load_from_safetensors(
      (fs::path(TEST_DATA_DIR) / "lm_head_test.safetensors").string());
  const auto &hidden = tensors.at("hidden");
  const auto &weight = tensors.at("weight");
  const auto &expected_out = tensors.at("expected");

  LmHeadDense lm_head = LmHeadDense::from_parameters(weight);

  LayerContext ctx(1);
  Tensor<__nv_bfloat16> actual_out = lm_head(ctx, hidden);

  CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  assert_tensors_equal(actual_out, expected_out);
}
