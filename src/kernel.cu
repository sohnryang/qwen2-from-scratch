#include "kernel.h"

#include <iostream>

__global__ void hello_kernel(int *a) { *a = 42; }

void launch_kernel(int *d_a) {
  std::cout << "Launching kernel..." << std::endl;
  hello_kernel<<<1, 1>>>(d_a);
  cudaDeviceSynchronize();
  std::cout << "Kernel finished." << std::endl;
}
