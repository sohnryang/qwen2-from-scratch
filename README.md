# Qwen2 from Scratch

Creating a Qwen2 inference kernel from scratch

Inspired by, but not strictly following the [Tiny LLM tutorial](https://skyzh.github.io/tiny-llm/) by `skyzh`.

![Demo](./demo.png)

## Build and Run

Requirements:
- CUDA toolkit (CUDA C++20) with an Ampere+ GPU (default arch sm_89; override via CMake flag).
- CMake ≥ 3.20, Ninja (recommended), a C++23 compiler.

Configure (Debug, default arch sm_89):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```
Release is recommended for actual use:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```
Override GPU arch, e.g. sm_80:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
```

Build:
```bash
cmake --build build
```

Run unit tests:
```bash
ctest --test-dir build --output-on-failure
```

Run chat demo:
```bash
./build/bin/qwen2chat -w path/to/weights.safetensors -t path/to/tokenizer.json
```
Options:
- `-w <file>`: weights (safetensors)
- `-t <file>`: tokenizer
- `-p <text>`: system prompt (default: "You're a helpful assistant.")
- `-l <int>`: max sequence length (default: 128)
- `-m`: print timing stats (elapsed seconds, tokens/s) per response
- `-h`: help
