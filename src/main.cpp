#include "kernel.h"

#include <cuda_runtime.h>

int main() {
  int *d_a;
  cudaMalloc((void **)&d_a, sizeof(int));
  launch_kernel(d_a);
  cudaFree(d_a);
  return 0;
}
