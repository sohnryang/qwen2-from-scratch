#include "model.h"
#include "tensor.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <tokenizers_cpp.h>

#if __has_include(<unistd.h>)
#include <unistd.h>

static void print_usage() {
  std::cout << "Usage: app [options]\n"
            << "Options:\n"
            << "  -w <file>   Path to the weights file.\n"
            << "  -t <file>   Path to the tokenizer file.\n"
            << "  -i <file>   Path to the inputs file.\n"
            << "  -p <text>   System prompt text (default: \"You're a helpful "
               "assistant.\").\n"
            << "  -l <int>    Maximum sequence length (default: 4096).\n"
            << "  -m          Print timing stats per response.\n"
            << "  -h          Print this help message.\n";
}
#endif

static std::unique_ptr<tokenizers::Tokenizer>
load_tokenizer(const std::string &filename) {
  std::ifstream tokenizer_file(filename, std::ios::binary | std::ios::ate);
  const auto tokenizer_file_size = tokenizer_file.tellg();
  tokenizer_file.seekg(0);
  std::string tokenizer_blob;
  tokenizer_blob.resize(tokenizer_file_size);
  if (!tokenizer_file.read(reinterpret_cast<char *>(tokenizer_blob.data()),
                           tokenizer_file_size))
    throw std::runtime_error("failed to read tokenizer file");
  return tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);
}

int main(int argc, char **argv) {
  std::string weights_filename, tokenizer_filename, input_filename,
      system_prompt = "You're a helpful assistant.";
  std::size_t max_sequence_length = 4096;
  bool measure_timing = false;
#if __has_include(<unistd.h>)
  int next_option;
  do {
    next_option = getopt(argc, argv, "w:t:i:p:l:mh");
    switch (next_option) {
    case 'w':
      weights_filename = optarg;
      continue;
    case 't':
      tokenizer_filename = optarg;
      continue;
    case 'i':
      input_filename = optarg;
      continue;
    case 'p':
      system_prompt = optarg;
      continue;
    case 'l':
      max_sequence_length = std::stoi(optarg);
      continue;
    case 'm':
      measure_timing = true;
      continue;
    case '?':
    case 'h':
      print_usage();
      std::exit(0);
    }
  } while (next_option != -1);
#endif

  if (weights_filename.empty()) {
    std::cout << "Path to weights: ";
    std::cin >> weights_filename;
  }
  if (tokenizer_filename.empty()) {
    std::cout << "Path to tokenizer: ";
    std::cin >> tokenizer_filename;
  }

  const auto model_weights = load_from_safetensors(weights_filename);
  auto model = Qwen2Model::from_parameters(model_weights, max_sequence_length);

  const auto tokenizer = load_tokenizer(tokenizer_filename);

  const std::string BOS = "<|im_start|>", EOS = "<|im_end|>";
  std::string prompt = BOS + "system\n" + system_prompt + EOS + "\n";
  const auto tokenized_prompt = tokenizer->Encode(prompt);
  if (!model.prefill(tokenized_prompt)) {
    std::cerr << "System prompt too long." << std::endl;
    std::exit(1);
  }

  while (true) {
    std::cout << "User: ";
    std::string user_prompt;
    std::getline(std::cin, user_prompt);

    const auto tokenized_user_prompt = tokenizer->Encode(
        BOS + "user\n" + user_prompt + EOS + "\n" + BOS + "assistant\n");
    const auto start = std::chrono::steady_clock::now();
    auto generated_tokens = model.generate(tokenized_user_prompt);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    if (generated_tokens.empty()) {
      std::cout << "LLM: prompt too long, please try again." << std::endl;
      continue;
    }
    generated_tokens.pop_back();
    std::cout << "LLM: " << tokenizer->Decode(generated_tokens) << std::endl;
    if (measure_timing) {
      const auto seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(elapsed)
              .count();
      const auto token_count = generated_tokens.size() + 1;
      const auto tps =
          static_cast<double>(token_count) / std::max(seconds, 1e-9);
      std::cout << "[timing] elapsed=" << seconds
                << "s, tokens=" << generated_tokens.size()
                << ", tokens/s=" << tps << std::endl;
    }
  }
}
