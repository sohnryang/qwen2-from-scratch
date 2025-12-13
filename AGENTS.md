# Repository Guidelines

## Project Structure & Module Organization
- `src/`: CUDA/C++23 implementation of tensors, kernels, and layers (`kernel.cu`, `layer.*`, `tensor.cc`); `main.cc` hosts the CLI chat driver.
- `include/`: Public headers consumed by both library and tests.
- `tests/`: GoogleTest cases (`*_test.cc`) plus fixtures in `tests/data`; paths resolved via `TEST_DATA_DIR`.
- `scripts/`: Maintenance utilities such as `create_test_data.py`.
- Build artifacts live under `build/` (or any CMake build dir); binaries land in `build/bin`, libraries in `build/lib`.

## Build, Test, and Development Commands
- Configure (adjust build type or CUDA arch as needed): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug`  
  For other GPUs, pass `-DCMAKE_CUDA_ARCHITECTURES=<sm>` to override the default 89.
- Build everything: `cmake --build build`
- Run unit tests: `ctest --test-dir build --output-on-failure`
- Run the chat demo: `./build/bin/qwen2chat <args>` after a successful build.
- Refresh test fixtures if needed: `python scripts/create_test_data.py`

## Coding Style & Naming Conventions
- C++23 with CUDA C++20; prefer modern features (ranges, aggregate initialization, `std::exchange`).
- Formatting uses `clang-format` (LLVM base). Run it before committing for any touched C++/CUDA sources.
- File and symbol names favor lower_snake_case; keep headers under `include/` and implementation in `src/`.
- Keep device/host boundaries explicit; guard CUDA calls with existing `CHECK_CUDA`/`assert` patterns.

## Testing Guidelines
- Tests live in `tests/*_test.cc` and use GoogleTest/GMock; mirror production filenames where possible.
- Add new fixtures to `tests/data` and reference via `TEST_DATA_DIR` to avoid hardcoded paths.
- Exercise edge shapes and dtype constraints for kernels; prefer deterministic inputs to enable bitwise comparisons.
- Run `ctest --test-dir build` before pushing; aim to keep runtime GPU-friendly (small tensors, short runs).

## Commit & Pull Request Guidelines
- Commit messages follow a conventional style seen in history (e.g., `feat(main): ...`, `test(layer): ...`, `build:`); keep them scoped and imperative.
- Keep commits focused (one logical change each) and include rationale in the body if behavior shifts.
- Pull requests should describe intent, list key changes, note test results (`ctest` output), and link issues when applicable. CLI log snippets beat screenshots for verification.
