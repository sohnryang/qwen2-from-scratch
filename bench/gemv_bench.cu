#include "cuda_utils.h"
#include "kernel.h"
#include "tensor.h"

#include <benchmark/benchmark.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

static void gemv_using_gemm(benchmark::State &state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  const std::size_t k = static_cast<std::size_t>(state.range(1));
  const std::size_t m = 1;

  std::vector<__nv_bfloat16> host_mat(n * k);
  std::vector<__nv_bfloat16> host_vec(k);
  for (std::size_t i = 0; i < n * k; ++i)
    host_mat[i] = __float2bfloat16(0.02f * static_cast<float>(i + 1));
  for (std::size_t i = 0; i < k; ++i)
    host_vec[i] = __float2bfloat16(0.01f * static_cast<float>(i + 1));

  Storage<__nv_bfloat16> mat_storage(host_mat);
  Storage<__nv_bfloat16> vec_storage(host_vec);
  Storage<__nv_bfloat16> out_storage(n);

  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&stop_event));

  const dim3 threads_per_block(16, 16);
  const dim3 num_blocks(ceil_div(m, threads_per_block.x),
                        ceil_div(n, threads_per_block.y));

  for (auto _ : state) {
    CHECK_CUDA(cudaEventRecord(start_event));
    gemm_transposed<<<num_blocks, threads_per_block, 0>>>(
        out_storage.data, vec_storage.data, mat_storage.data, nullptr,
        __float2bfloat16(1.0f), m, n, k);
    CHECK_CUDA(cudaEventRecord(stop_event));
    CHECK_CUDA(cudaEventSynchronize(stop_event));
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    state.SetIterationTime(elapsed_ms / 1000.0);
  }

  state.SetBytesProcessed(std::int64_t(state.iterations()) * (k * n + k + n) *
                          sizeof(__nv_bfloat16));

  CHECK_CUDA(cudaEventDestroy(start_event));
  CHECK_CUDA(cudaEventDestroy(stop_event));
}

BENCHMARK(gemv_using_gemm)
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({4096, 4096})
    ->Args({256, 1536})
    ->Args({1536, 1536})
    ->Args({1536, 8960})
    ->Args({8960, 1536})
    ->UseManualTime();

static void gemv_using_fastgemv(benchmark::State &state) {
  const std::size_t m = static_cast<std::size_t>(state.range(0));
  const std::size_t n = static_cast<std::size_t>(state.range(1));

  std::vector<__nv_bfloat16> host_mat(m * n);
  std::vector<__nv_bfloat16> host_vec(m);
  for (std::size_t i = 0; i < m * n; ++i)
    host_mat[i] = __float2bfloat16(0.02f * static_cast<float>(i + 1));
  for (std::size_t i = 0; i < m; ++i)
    host_vec[i] = __float2bfloat16(0.01f * static_cast<float>(i + 1));

  Storage<__nv_bfloat16> mat_storage(host_mat);
  Storage<__nv_bfloat16> vec_storage(host_vec);
  Storage<__nv_bfloat16> out_storage(n);

  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&stop_event));

  const dim3 threads_per_block(state.range(2), state.range(3));
  const dim3 num_blocks(1, n / threads_per_block.y);

  for (auto _ : state) {
    CHECK_CUDA(cudaEventRecord(start_event));
    gemv_transposed<<<num_blocks, threads_per_block,
                      sizeof(float) * threads_per_block.y * 32>>>(
        out_storage.data, mat_storage.data, vec_storage.data, nullptr, m, n);
    CHECK_CUDA(cudaEventRecord(stop_event));
    CHECK_CUDA(cudaEventSynchronize(stop_event));
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    state.SetIterationTime(elapsed_ms / 1000.0);
  }

  state.SetBytesProcessed(std::int64_t(state.iterations()) * (m * n + m + n) *
                          sizeof(__nv_bfloat16));

  CHECK_CUDA(cudaEventDestroy(start_event));
  CHECK_CUDA(cudaEventDestroy(stop_event));
}

BENCHMARK(gemv_using_fastgemv)
    ->Args({512, 512, 32, 4})
    ->Args({1024, 1024, 32, 4})
    ->Args({2048, 2048, 32, 4})
    ->Args({4096, 4096, 128, 8})
    ->Args({256, 1536, 32, 4})
    ->Args({1536, 1536, 32, 4})
    ->Args({1536, 8960, 32, 4})
    ->Args({8960, 1536, 32, 4})
    ->UseManualTime();

BENCHMARK_MAIN();
