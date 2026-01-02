#include "layer.h"
#include "tensor.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

#include <cuda_bf16.h>

ScratchPad::~ScratchPad() {
  if (_pool)
    CHECK_CUDA(cudaFree(_pool));
}

ScratchPad::ScratchPad(ScratchPad &&other) noexcept
    : _pool{std::exchange(other._pool, nullptr)}, _size{other._size},
      _used{other._used} {}

ScratchPad &ScratchPad::operator=(ScratchPad &&other) noexcept {
  using std::swap;
  swap(_pool, other._pool);
  swap(_size, other._size);
  swap(_used, other._used);
  return *this;
}

ScratchPad::ScratchPad(std::size_t size) : _size{size} {
  CHECK_CUDA(cudaMalloc(&_pool, size));
}

void ScratchPad::reset() { _used = 0; }

template <typename T>
std::shared_ptr<Storage<T>> ScratchPad::make_storage(std::size_t elems,
                                                     std::size_t align) {
  const auto aligned_usage = (_used + align - 1) & ~(align - 1);
  const auto alloc_size = elems * sizeof(T);
  if (aligned_usage + alloc_size > _size)
    return std::make_shared<Storage<T>>(elems);

  const auto res = std::make_shared<Storage<T>>(
      (T *)((std::uint8_t *)_pool + aligned_usage), elems);
  _used = aligned_usage + alloc_size;
  return res;
}
template std::shared_ptr<Storage<__nv_bfloat16>>
ScratchPad::make_storage(std::size_t elems, std::size_t align);
template std::shared_ptr<Storage<float>>
ScratchPad::make_storage(std::size_t elems, std::size_t align);
template std::shared_ptr<Storage<int>>
ScratchPad::make_storage(std::size_t elems, std::size_t align);

ScratchPadScope::~ScratchPadScope() { _scratchpad.reset(); }

ScratchPadScope::ScratchPadScope(ScratchPad &scratchpad)
    : _scratchpad{scratchpad} {}

LayerContext::~LayerContext() {
  if (_stream)
    CHECK_CUDA(cudaStreamDestroy(_stream));
  if (_last_token_index)
    CHECK_CUDA(cudaFree(_last_token_index));
}

LayerContext::LayerContext(std::size_t scratchpad_size)
    : _scratchpad{scratchpad_size} {
  CHECK_CUDA(cudaStreamCreate(&_stream));
  CHECK_CUDA(cudaMalloc(&_last_token_index, sizeof(int)));
}

LayerContext::LayerContext(LayerContext &&other) noexcept
    : _scratchpad{std::move(other._scratchpad)},
      _stream{std::exchange(other._stream, nullptr)},
      _last_token_index{std::exchange(other._last_token_index, nullptr)} {}

LayerContext &LayerContext::operator=(LayerContext &&other) noexcept {
  if (this == &other)
    return *this;
  if (_stream)
    CHECK_CUDA(cudaStreamDestroy(_stream));
  if (_last_token_index)
    CHECK_CUDA(cudaFree(_last_token_index));
  _scratchpad = std::move(other._scratchpad);
  _stream = std::exchange(other._stream, nullptr);
  _last_token_index = std::exchange(other._last_token_index, nullptr);
  return *this;
}

InOutBuffer::~InOutBuffer() {
  CHECK_CUDA(cudaFree(_inout[0]));
  CHECK_CUDA(cudaFree(_inout[1]));
}

InOutBuffer::InOutBuffer(InOutBuffer &&other) noexcept
    : _inout{std::exchange(other._inout, {})}, _size{other._size} {}

InOutBuffer &InOutBuffer::operator=(InOutBuffer &&other) noexcept {
  using std::swap;
  swap(_inout, other._inout);
  swap(_size, other._size);
  return *this;
}

InOutBuffer::InOutBuffer(std::size_t size) : _size{size} {
  CHECK_CUDA(cudaMalloc(&_inout[0], size));
  CHECK_CUDA(cudaMalloc(&_inout[1], size));
}

Dense::Dense(std::size_t in_features, std::size_t out_features,
             bool use_activation, std::size_t cache_size)
    : _in_features{in_features}, _out_features{out_features},
      _use_activation{use_activation},
      _weight{.shape = {_out_features, _in_features},
              .dimensions = 2,
              .storage = std::make_shared<Storage<__nv_bfloat16>>(
                  _in_features * _out_features)},
      _bias{}, _cache{std::make_shared<Storage<__nv_bfloat16>>(_out_features *
                                                               cache_size)} {}

Dense Dense::from_parameters(const Tensor<__nv_bfloat16> &weight,
                             bool use_activation, std::size_t cache_size) {
  assert(weight.dimensions == 2 && "invalid weight dimension");
  const auto in_features = weight.shape[1], out_features = weight.shape[0];
  Dense dense(in_features, out_features, use_activation, cache_size);
  dense._weight = weight;
  return dense;
}

Dense Dense::from_parameters(const Tensor<__nv_bfloat16> &weight,
                             const Tensor<__nv_bfloat16> &bias,
                             bool use_activation, std::size_t cache_size) {
  assert(weight.dimensions == 2 && "invalid weight dimension");
  assert(bias.dimensions == 1 && "invalid bias dimension");
  assert(weight.shape[0] == bias.shape[0] && "weight and bias shape mismatch");
  const auto in_features = weight.shape[1], out_features = weight.shape[0];

  Dense dense(in_features, out_features, use_activation, cache_size);
  dense._weight = weight;
  dense._bias = bias;
  return dense;
}

void Dense::rollback(std::size_t previous_cached_batches) {
  assert(_cache->data && "KV cache should be used");
  _cached_batches = previous_cached_batches;
}

RMSNorm::RMSNorm(std::size_t dimensions, float epsilon)
    : _dimensions{dimensions}, _epsilon{epsilon},
      _weight{.shape = {_dimensions},
              .dimensions = 1,
              .storage =
                  std::make_shared<Storage<__nv_bfloat16>>(_dimensions)} {}

RMSNorm RMSNorm::from_parameter(const Tensor<__nv_bfloat16> &weight,
                                float epsilon) {
  assert(weight.dimensions == 1 && "invalid weight dimension");
  RMSNorm rmsnorm(weight.shape[0], epsilon);
  rmsnorm._weight = weight;
  return rmsnorm;
}

void GroupedQueryAttention::rollback(std::size_t previous_cached_batches) {
  _k_layer.rollback(previous_cached_batches);
  _v_layer.rollback(previous_cached_batches);
}

Qwen2TransformerBlock::Qwen2TransformerBlock(
    const RMSNorm &input_norm_layer,
    const GroupedQueryAttention &attention_layer,
    const RMSNorm &post_attention_norm_layer, const Dense &gate_proj_layer,
    const Dense &up_proj_layer, const Dense &down_proj_layer)
    : _input_norm_layer(input_norm_layer), _attention_layer(attention_layer),
      _post_attention_norm_layer(post_attention_norm_layer),
      _gate_proj_layer(gate_proj_layer), _up_proj_layer(up_proj_layer),
      _down_proj_layer(down_proj_layer) {
  assert(_input_norm_layer.dimensions() ==
             _attention_layer.q_layer().in_features() &&
         _input_norm_layer.dimensions() ==
             _attention_layer.k_layer().in_features() &&
         _input_norm_layer.dimensions() ==
             _attention_layer.v_layer().in_features() &&
         "QKV layers and input RMSNorm dimensions should match");
  assert(_attention_layer.o_layer().out_features() ==
             _gate_proj_layer.in_features() &&
         _gate_proj_layer.out_features() == _up_proj_layer.out_features() &&
         _gate_proj_layer.out_features() == _down_proj_layer.in_features() &&
         _down_proj_layer.out_features() ==
             _attention_layer.o_layer().out_features() &&
         "attention layer and MLP layer dimensions should match");
}

void Qwen2TransformerBlock::rollback(std::size_t previous_cached_batches) {
  _attention_layer.rollback(previous_cached_batches);
}

Embedding::Embedding(std::size_t table_size, std::size_t dimension)
    : _table_size{table_size}, _dimension{dimension},
      _embedding_table{.shape = {table_size, _dimension},
                       .dimensions = 2,
                       .storage = std::make_shared<Storage<__nv_bfloat16>>(
                           _table_size * _dimension)} {}

Embedding
Embedding::from_parameter(const Tensor<__nv_bfloat16> &embedding_table) {
  assert(embedding_table.dimensions == 2 &&
         "invalid embedding table dimension");
  Embedding embedding(embedding_table.shape[0], embedding_table.shape[1]);
  embedding._embedding_table = embedding_table;
  return embedding;
}

Sampler::Sampler(std::size_t vocab_size)
    : _vocab_size{vocab_size}, _vals_storage{}, _vals_storage_next{},
      _indices_storage{}, _indices_storage_next{} {}

LmHeadDense::LmHeadDense(std::size_t in_features, std::size_t out_features)
    : _in_features{in_features}, _out_features{out_features},
      _weight{.shape = {_out_features, _in_features},
              .dimensions = 2,
              .storage = std::make_shared<Storage<__nv_bfloat16>>(
                  _out_features * _in_features)} {}

LmHeadDense LmHeadDense::from_parameters(const Tensor<__nv_bfloat16> &weight) {
  assert(weight.dimensions == 2 && "invalid weight dimension");
  const auto in_features = weight.shape[1], out_features = weight.shape[0];
  LmHeadDense dense(in_features, out_features);
  dense._weight = weight;
  return dense;
}
