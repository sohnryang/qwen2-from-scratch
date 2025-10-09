#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "numpy",
#     "packaging",
#     "safetensors",
#     "torch",
# ]
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
# ///
import argparse
import os

import torch
from safetensors.torch import save_file


def create_safetensors_test_file(data_dir: str):
    zeros_2x3x5x7 = torch.zeros((2, 3, 5, 7), dtype=torch.bfloat16)
    contiguous_2x3x5x7 = torch.arange(2 * 3 * 5 * 7, dtype=torch.bfloat16).reshape(
        (2, 3, 5, 7)
    )
    save_file(
        {"zeros_2x3x5x7": zeros_2x3x5x7, "contiguous_2x3x5x7": contiguous_2x3x5x7},
        os.path.join(data_dir, "load_test.safetensors"),
    )


def create_matmul_test_file(data_dir: str):
    in_a = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.bfloat16)
    in_b = torch.tensor([[6, 7, 8], [9, 10, 11]], dtype=torch.bfloat16)
    bias = torch.ones((3, 3), dtype=torch.bfloat16)
    out = torch.matmul(in_a, in_b) + bias
    save_file(
        {"in_a": in_a, "in_b": in_b, "bias": bias, "out": out},
        os.path.join(data_dir, "matmul_test.safetensors"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create_test_data")
    parser.add_argument("data_dir", help="Test data directory", type=str)
    parsed = parser.parse_args()

    create_safetensors_test_file(parsed.data_dir)
    create_matmul_test_file(parsed.data_dir)
