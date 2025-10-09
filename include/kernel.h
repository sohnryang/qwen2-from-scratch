#pragma once

#include "tensor.h"

#include <cstddef>
#include <cuda_bf16.h>

__global__ void gemm(__nv_bfloat16 *__restrict__ out,
                     const __nv_bfloat16 *__restrict__ in_a,
                     const __nv_bfloat16 *__restrict__ in_b,
                     const __nv_bfloat16 *__restrict__ bias,
                     __nv_bfloat16 scale, std::size_t m, std::size_t n,
                     std::size_t k);

void launch_gemm(Tensor &out, const Tensor &in_a, const Tensor &in_b,
                 const Tensor &bias, __nv_bfloat16 scale);
