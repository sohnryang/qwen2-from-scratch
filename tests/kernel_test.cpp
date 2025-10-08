#include "kernel.h"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

TEST(KernelTest, WriteToBuffer) {
  int *d_a;
  int *h_a;
  h_a = new int;
  cudaMalloc((void **)&d_a, sizeof(int));
  launch_kernel(d_a);
  cudaMemcpy(h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(*h_a, 42);
  delete h_a;
  cudaFree(d_a);
}
