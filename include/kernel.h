#pragma once

#include <cstddef>
#include <cuda_bf16.h>

__global__ void gemm(__nv_bfloat16 *__restrict__ out,
                     const __nv_bfloat16 *__restrict__ in_a,
                     const __nv_bfloat16 *__restrict__ in_b,
                     const __nv_bfloat16 *__restrict__ bias, std::size_t m,
                     std::size_t n, std::size_t k);
