#include "tensor.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <map>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <simdjson.h>

Storage::~Storage() { CHECK_CUDA(cudaFree(data)); }

Storage::Storage(const Storage &other) : elems(other.elems) {
  CHECK_CUDA(cudaMemcpy(data, other.data, sizeof(__nv_bfloat16) * other.elems,
                        cudaMemcpyDeviceToDevice));
}

Storage::Storage(Storage &&other) {
  std::swap(data, other.data);
  std::swap(elems, other.elems);
}

Storage::Storage(std::size_t elems_) : elems(elems_) {
  CHECK_CUDA(cudaMalloc((void **)&data, elems * sizeof(__nv_bfloat16)));
}

Storage Storage::load_from_offset(const std::uint8_t *buf, std::size_t begin,
                                  std::size_t end) {
  Storage loaded;
  const auto bytes = end - begin;
  loaded.elems = bytes / sizeof(__nv_bfloat16);
  CHECK_CUDA(cudaMalloc((void **)&loaded.data, bytes));
  CHECK_CUDA(
      cudaMemcpy(loaded.data, buf + begin, bytes, cudaMemcpyHostToDevice));
  return loaded;
}

std::vector<__nv_bfloat16> Storage::to_host() {
  std::vector<__nv_bfloat16> host_data(elems);
  CHECK_CUDA(cudaMemcpy(host_data.data(), data, elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  return host_data;
}

std::map<std::string, Tensor>
load_from_safetensors(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  const auto file_size = file.tellg();
  file.seekg(0);

  std::uint64_t header_size;
  if (!file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size)))
    throw std::runtime_error("failed to read header size");
  std::string header(header_size, '\0');
  if (!file.read(&header[0], header_size))
    throw std::runtime_error("failed to read JSON header");
  header.erase(header.find_last_not_of(" ") + 1);
  const auto buf_size =
      std::int64_t{file_size} - header_size - sizeof(header_size);
  std::vector<std::uint8_t> buf;
  buf.resize(buf_size);
  if (!file.read(reinterpret_cast<char *>(buf.data()), buf_size)) {
    throw std::runtime_error("failed to read byte buffer");
  }

  simdjson::dom::parser parser;
  simdjson::dom::element doc = parser.parse(simdjson::padded_string(header));
  simdjson::dom::object object = doc.get_object();
  std::map<std::string, Tensor> tensors;
  for (auto [name, specs] : object) {
    if (name == "__metadata__")
      continue;

    std::string_view dtype = specs["dtype"].get_string();
    if (dtype != "BF16")
      throw std::runtime_error("invalid dtype");

    simdjson::dom::array offsets = specs["data_offsets"].get_array();
    if (offsets.size() != 2)
      throw std::runtime_error("invalid offsets");
    std::uint64_t offsets_arr[2];
    for (auto [i, offset] : std::views::enumerate(offsets))
      offsets_arr[i] = offset.get_uint64();
    auto storage = std::make_shared<Storage>(std::move(
        Storage::load_from_offset(buf.data(), offsets_arr[0], offsets_arr[1])));

    simdjson::dom::array shape = specs["shape"].get_array();
    if (shape.size() > 4)
      throw std::runtime_error("dimension too large");
    Tensor tensor = {.dimensions = shape.size(), .storage = storage};
    for (auto [i, elem] : std::views::enumerate(shape))
      tensor.shape[i] = elem.get_uint64();
    tensors[std::string(name)] = tensor;
  }

  return tensors;
}
