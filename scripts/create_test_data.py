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
from torch import nn


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
    in_b_transposed = in_b.transpose(0, 1).contiguous()
    bias = torch.ones((3,), dtype=torch.bfloat16)
    out = torch.matmul(in_a, in_b) + bias
    save_file(
        {
            "in_a": in_a,
            "in_b": in_b,
            "in_b_transposed": in_b_transposed,
            "bias": bias,
            "out": out,
        },
        os.path.join(data_dir, "matmul_test.safetensors"),
    )


def create_dense_test_file(data_dir: str):
    model = nn.Sequential(nn.Linear(2, 4, dtype=torch.bfloat16), nn.SiLU())
    weight = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.bfloat16)
    bias = torch.tensor([0, 1, 2, 3], dtype=torch.bfloat16)
    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.bfloat16)
    with torch.no_grad():
        model[0].weight = nn.Parameter(weight)
        model[0].bias = nn.Parameter(bias)
        out = model(x)
    save_file(
        {"weight": weight, "bias": bias, "x": x, "out": out},
        os.path.join(data_dir, "dense_test.safetensors"),
    )


def create_reshape_test_file(data_dir: str):
    arange_2x3x4 = torch.arange(2 * 3 * 4, dtype=torch.bfloat16).reshape((2, 3, 4))
    save_file(
        {"arange_2x3x4": arange_2x3x4},
        os.path.join(data_dir, "reshape_test.safetensors"),
    )


def create_rmsnorm_test_file(data_dir: str):
    dims = 128
    eps = 1e-5
    x = torch.linspace(-5, 5, 2 * dims, dtype=torch.bfloat16).reshape(2, dims)

    rmsnorm_layer = nn.RMSNorm(dims, eps=eps)
    rmsnorm_layer.to(torch.bfloat16)

    custom_weight = torch.linspace(0, 2, dims, dtype=torch.bfloat16)
    rmsnorm_layer.weight.data = custom_weight

    out = rmsnorm_layer(x)

    save_file(
        {"x": x, "weight": custom_weight, "out": out},
        os.path.join(data_dir, "rmsnorm_test.safetensors"),
    )


def create_softmax_test_file(data_dir: str):
    x = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0, -4.0],
            [1.2, -3.4, 5.6, -7.8],
        ],
        dtype=torch.bfloat16,
    )
    out = nn.functional.softmax(x, dim=-1)
    save_file(
        {"x": x, "out": out},
        os.path.join(data_dir, "softmax_test.safetensors"),
    )


def create_gqa_test_file(data_dir: str):
    batch_size = 1
    seq_len = 5
    hidden_dim = 32
    num_heads = 8
    num_kv_heads = 2
    groups = num_heads // num_kv_heads

    q_in = torch.linspace(
        -128, 128, batch_size * seq_len * hidden_dim, dtype=torch.bfloat16
    ).reshape((batch_size, seq_len, hidden_dim))
    q_weight = torch.eye(hidden_dim, dtype=torch.bfloat16)
    q_proj = torch.matmul(q_in, q_weight.T)
    q = q_proj.reshape((batch_size, seq_len, num_heads, -1)).transpose(1, 2)

    k_in = torch.linspace(
        64, -64, batch_size * seq_len * hidden_dim // groups, dtype=torch.bfloat16
    ).reshape((batch_size, seq_len, hidden_dim // groups))
    k_weight = torch.eye(hidden_dim // groups, dtype=torch.bfloat16)
    k_proj = torch.matmul(k_in, k_weight.T)
    k = k_proj.reshape((batch_size, seq_len, num_kv_heads, -1)).transpose(1, 2)

    v_in = torch.linspace(
        32, -32, batch_size * seq_len * hidden_dim // groups, dtype=torch.bfloat16
    ).reshape((batch_size, seq_len, hidden_dim // groups))
    v_weight = torch.eye(hidden_dim // groups, dtype=torch.bfloat16)
    v_proj = torch.matmul(v_in, v_weight.T)
    v = v_proj.reshape((batch_size, seq_len, num_kv_heads, -1)).transpose(1, 2)

    o_in = (
        nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        .transpose(1, 2)
        .contiguous()
        .reshape((batch_size, seq_len, hidden_dim))
    )
    o_weight = torch.eye(hidden_dim, dtype=torch.bfloat16)
    o_proj = torch.matmul(o_in, o_weight.T)
    save_file(
        {
            "q_weight": q_weight,
            "k_weight": k_weight,
            "v_weight": v_weight,
            "o_weight": o_weight,
            "q_in": q_in,
            "k_in": k_in,
            "v_in": v_in,
            "o_proj": o_proj,
        },
        os.path.join(data_dir, "gqa_test.safetensors"),
    )


def create_gqa_variable_len_test_file(data_dir: str):
    batch_size = 1
    q_seq_len = 1
    kv_seq_len = 5
    hidden_dim = 32
    num_heads = 8
    num_kv_heads = 2
    groups = num_heads // num_kv_heads

    q_in = torch.linspace(
        -128, 128, batch_size * q_seq_len * hidden_dim, dtype=torch.bfloat16
    ).reshape((batch_size, q_seq_len, hidden_dim))
    q_weight = torch.eye(hidden_dim, dtype=torch.bfloat16)
    q_proj = torch.matmul(q_in, q_weight.T)
    q = q_proj.reshape((batch_size, q_seq_len, num_heads, -1)).transpose(1, 2)

    k_in = torch.linspace(
        64, -64, batch_size * kv_seq_len * hidden_dim // groups, dtype=torch.bfloat16
    ).reshape((batch_size, kv_seq_len, hidden_dim // groups))
    k_weight = torch.eye(hidden_dim // groups, dtype=torch.bfloat16)
    k_proj = torch.matmul(k_in, k_weight.T)
    k = k_proj.reshape((batch_size, kv_seq_len, num_kv_heads, -1)).transpose(1, 2)

    v_in = torch.linspace(
        32, -32, batch_size * kv_seq_len * hidden_dim // groups, dtype=torch.bfloat16
    ).reshape((batch_size, kv_seq_len, hidden_dim // groups))
    v_weight = torch.eye(hidden_dim // groups, dtype=torch.bfloat16)
    v_proj = torch.matmul(v_in, v_weight.T)
    v = v_proj.reshape((batch_size, kv_seq_len, num_kv_heads, -1)).transpose(1, 2)

    o_in = (
        nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        .transpose(1, 2)
        .contiguous()
        .reshape((batch_size, q_seq_len, hidden_dim))
    )
    o_weight = torch.eye(hidden_dim, dtype=torch.bfloat16)
    o_proj = torch.matmul(o_in, o_weight.T)
    save_file(
        {
            "q_weight": q_weight,
            "k_weight": k_weight,
            "v_weight": v_weight,
            "o_weight": o_weight,
            "q_in": q_in,
            "k_in": k_in,
            "v_in": v_in,
            "o_proj": o_proj,
        },
        os.path.join(data_dir, "gqa_variable_len_test.safetensors"),
    )


def create_gqa_masked_test_file(data_dir: str):
    batch_size = 1
    seq_len = 5
    hidden_dim = 32
    num_heads = 8
    num_kv_heads = 2
    groups = num_heads // num_kv_heads

    q_in = torch.linspace(
        -128, 128, batch_size * seq_len * hidden_dim, dtype=torch.bfloat16
    ).reshape((batch_size, seq_len, hidden_dim))
    q_weight = torch.eye(hidden_dim, dtype=torch.bfloat16)
    q_proj = torch.matmul(q_in, q_weight.T)
    q = q_proj.reshape((batch_size, seq_len, num_heads, -1)).transpose(1, 2)

    k_in = torch.linspace(
        64, -64, batch_size * seq_len * hidden_dim // groups, dtype=torch.bfloat16
    ).reshape((batch_size, seq_len, hidden_dim // groups))
    k_weight = torch.eye(hidden_dim // groups, dtype=torch.bfloat16)
    k_proj = torch.matmul(k_in, k_weight.T)
    k = k_proj.reshape((batch_size, seq_len, num_kv_heads, -1)).transpose(1, 2)

    v_in = torch.linspace(
        32, -32, batch_size * seq_len * hidden_dim // groups, dtype=torch.bfloat16
    ).reshape((batch_size, seq_len, hidden_dim // groups))
    v_weight = torch.eye(hidden_dim // groups, dtype=torch.bfloat16)
    v_proj = torch.matmul(v_in, v_weight.T)
    v = v_proj.reshape((batch_size, seq_len, num_kv_heads, -1)).transpose(1, 2)

    bool_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attn_mask = (
        torch.zeros(seq_len, seq_len, dtype=torch.bfloat16)
        .unsqueeze(0)
        .repeat_interleave(num_heads, dim=0)
        .unsqueeze(0)
        .repeat_interleave(batch_size, dim=0)
    )
    attn_mask.masked_fill_(bool_mask, -float("inf"))

    o_in = (
        nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=True
        )
        .transpose(1, 2)
        .contiguous()
        .reshape((batch_size, seq_len, hidden_dim))
    )
    o_weight = torch.eye(hidden_dim, dtype=torch.bfloat16)
    o_proj = torch.matmul(o_in, o_weight.T)
    save_file(
        {
            "q_weight": q_weight,
            "k_weight": k_weight,
            "v_weight": v_weight,
            "o_weight": o_weight,
            "q_in": q_in,
            "k_in": k_in,
            "v_in": v_in,
            "attn_mask": attn_mask,
            "o_proj": o_proj,
        },
        os.path.join(data_dir, "gqa_masked_test.safetensors"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create_test_data")
    parser.add_argument("data_dir", help="Test data directory", type=str)
    parsed = parser.parse_args()

    create_safetensors_test_file(parsed.data_dir)
    create_matmul_test_file(parsed.data_dir)
    create_dense_test_file(parsed.data_dir)
    create_reshape_test_file(parsed.data_dir)
    create_rmsnorm_test_file(parsed.data_dir)
    create_softmax_test_file(parsed.data_dir)
    create_gqa_test_file(parsed.data_dir)
    create_gqa_variable_len_test_file(parsed.data_dir)
    create_gqa_masked_test_file(parsed.data_dir)
