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

  launch_gemm(actual_out, in_a, in_b, bias);

  auto actual_host_data = actual_out.storage->to_host();
  auto expected_host_data = expected_out.storage->to_host();

  ASSERT_EQ(actual_host_data.size(), expected_host_data.size());
  for (size_t i = 0; i < actual_host_data.size(); ++i) {
    EXPECT_FLOAT_EQ(__bfloat162float(actual_host_data[i]),
                    __bfloat162float(expected_host_data[i]));
  }
}
