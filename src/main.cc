#include "model.h"
#include "tensor.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

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
            << "  -b          Run benchmark mode on a preset prompt.\n"
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

static bool is_continuation(char c) { return (c & 0xC0) == 0x80; }

static std::pair<std::string, std::string>
split_partial_suffix(const std::string &input) {
  std::size_t continuation_count = 0;
  while (continuation_count < 3 && continuation_count < input.size() &&
         is_continuation(input[input.size() - 1 - continuation_count]))
    ++continuation_count;

  if (continuation_count == 0)
    return {input, ""};

  const std::size_t suffix_start = input.size() - continuation_count;
  if (suffix_start == 0)
    return {"", input};

  const auto lead = static_cast<unsigned char>(input[suffix_start - 1]);
  std::size_t expected_len = 0;
  if ((lead & 0x80) == 0x00)
    expected_len = 1;
  else if ((lead & 0xE0) == 0xC0)
    expected_len = 2;
  else if ((lead & 0xF0) == 0xE0)
    expected_len = 3;
  else if ((lead & 0xF8) == 0xF0)
    expected_len = 4;

  if (expected_len == continuation_count + 1)
    return {input, ""};

  return {input.substr(0, suffix_start), input.substr(suffix_start)};
}

int main(int argc, char **argv) {
  std::string weights_filename, tokenizer_filename, input_filename,
      system_prompt = "You're a helpful assistant.";
  std::size_t max_sequence_length = 4096;
  bool measure_timing = false;
  bool benchmark_mode = false;
  const std::string benchmark_prompt =
      "Explain the architecture of the Qwen2 language model.";
#if __has_include(<unistd.h>)
  int next_option;
  do {
    next_option = getopt(argc, argv, "w:t:i:p:l:mbh");
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
    case 'b':
      benchmark_mode = true;
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
  const int EOS_ID = 151645;
  std::string system_prompt_with_template =
      BOS + "system\n" + system_prompt + EOS + "\n";
  bool first_message = true;

  auto producer_thread = model.spawn_producer();
  auto make_prompt = [&](const std::string &user_message,
                         bool include_system) -> std::string {
    std::string prompt;
    if (include_system)
      prompt = system_prompt_with_template;
    prompt += BOS + "user\n" + user_message + EOS + "\n" + BOS + "assistant\n";
    return prompt;
  };

  auto run_prompt = [&](const std::string &user_prompt, bool print_output,
                        bool print_timing) -> bool {
    const auto tokenized_user_prompt = tokenizer->Encode(user_prompt);
    const auto start_time = std::chrono::steady_clock::now();
    std::optional<std::chrono::time_point<std::chrono::steady_clock>>
        first_token_timestamp;
    if (!model.append_prompt(tokenized_user_prompt)) {
      std::cout << "LLM: prompt too long, please try again." << std::endl;
      return false;
    }

    Qwen2Model::StreamResult result;
    std::string partial_chars;
    std::size_t generated_tokens = 0;
    if (print_output)
      std::cout << "LLM: ";
    do {
      result = model.stream_response();
      if (result.out_of_space) {
        std::cerr << "Out of context." << std::endl;
        std::abort();
      }
      if (result.tokens.empty())
        continue;
      generated_tokens += result.tokens.size();
      if (result.tokens.back() == EOS_ID)
        result.tokens.pop_back();
      if (!first_token_timestamp)
        first_token_timestamp =
            result.timestamp.value_or(std::chrono::steady_clock::now());
      if (print_output) {
        const auto decoded = partial_chars + tokenizer->Decode(result.tokens);
        const auto [complete, suffix] = split_partial_suffix(decoded);
        std::cout << complete << std::flush;
        partial_chars = suffix;
      }
    } while (!result.done);
    if (print_timing) {
      const auto end_time = std::chrono::steady_clock::now();
      const auto ttfb_ms = first_token_timestamp
                               ? std::chrono::duration_cast<
                                     std::chrono::duration<double, std::milli>>(
                                     *first_token_timestamp - start_time)
                                     .count()
                               : 0.0;
      const auto generation_s =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                    start_time)
              .count();
      const double tps =
          generation_s > 0.0
              ? static_cast<double>(generated_tokens) / generation_s
              : 0.0;
      if (print_output)
        std::cout << "\n";
      std::cout << "[Timing] TTFT: " << std::fixed << std::setprecision(2)
                << ttfb_ms << " ms, TPS: " << std::setprecision(2) << tps
                << " (" << generated_tokens << " tokens)";
    }
    std::cout << std::endl;
    return true;
  };

  if (benchmark_mode) {
    std::cout << "Benchmark prompt: " << benchmark_prompt << std::endl;
    const auto user_prompt = make_prompt(benchmark_prompt, true);
    const bool ok = run_prompt(user_prompt, false, true);
    model.quit();
    producer_thread.join();
    return ok ? 0 : 1;
  }

  while (true) {
    std::cout << "User: ";
    std::string user_message;
    std::getline(std::cin, user_message);

    const auto user_prompt = make_prompt(user_message, first_message);
    first_message = false;
    run_prompt(user_prompt, true, measure_timing);
  }
  producer_thread.join();
}
