#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>

#if __has_include(<unistd.h>)
#include <unistd.h>

static void print_usage() {
  std::cout << "Usage: app [options]\n"
            << "Options:\n"
            << "  -w <file>   Path to the weights file.\n"
            << "  -i <file>   Path to the inputs file.\n"
            << "  -h          Print this help message.\n";
}
#endif

int main(int argc, char **argv) {
  std::string weights_filename, input_filename;
#if __has_include(<unistd.h>)
  int next_option;
  do {
    next_option = getopt(argc, argv, "w:i:h");
    switch (next_option) {
    case 'w':
      weights_filename = optarg;
      continue;
    case 'i':
      input_filename = optarg;
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
  if (input_filename.empty()) {
    std::cout << "Path to input: ";
    std::cin >> input_filename;
  }
}
