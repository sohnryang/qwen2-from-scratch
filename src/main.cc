#include <cuda_runtime.h>

int main() {
  int *d_a;
  cudaMalloc((void **)&d_a, sizeof(int));
  cudaFree(d_a);
  return 0;
}
