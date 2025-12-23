#include "layer.h"
#include "tensor.h"

#include <cassert>
#include <cstddef>
#include <memory>

#include <cuda_bf16.h>

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
