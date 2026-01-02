#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "accelerate",
#     "numpy",
#     "packaging",
#     "safetensors",
#     "torch",
#     "transformers",
# ]
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
# ///
import argparse
import os

import torch
import transformers
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


def create_dense_cache_test_file(data_dir: str):
    weight = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16)
    bias = torch.tensor([1, -1], dtype=torch.bfloat16)
    input_a = torch.tensor([[1, 0, 2], [3, 1, 4]], dtype=torch.bfloat16)
    input_b = torch.tensor([[-1, 2, 0], [0, 1, -2]], dtype=torch.bfloat16)
    expected_a = input_a @ weight.T + bias
    expected_b = input_b @ weight.T + bias
    expected_cached = torch.cat([expected_a, expected_b], dim=0)
    save_file(
        {
            "weight": weight,
            "bias": bias,
            "input_a": input_a,
            "input_b": input_b,
            "expected_a": expected_a,
            "expected_cached": expected_cached,
        },
        os.path.join(data_dir, "dense_cache_test.safetensors"),
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


def create_transformer_block_test_file(data_dir: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct", dtype="auto", device_map="auto"
    )
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "How do I write inference kernels for Qwen2 model in CUDA, from scratch?",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    cached_tokens = model_inputs["input_ids"].numel()

    with torch.no_grad():
        model_outputs = model(
            **model_inputs, use_cache=False, output_hidden_states=True
        )
        generated_tokens = model.generate(**model_inputs)
    next_user_message_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Can you summarize the response above?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    next_user_message_tokenized = tokenizer(
        [next_user_message_text], return_tensors="pt"
    ).to(model.device)
    appended_input_ids = torch.cat((model_inputs["input_ids"], generated_tokens, next_user_message_tokenized["input_ids"]), dim=-1)  # type: ignore
    model_inputs_appended = {
        "input_ids": appended_input_ids,
        "attention_mask": torch.ones_like(appended_input_ids),
    }
    with torch.no_grad():
        model_outputs_appended = model(
            **model_inputs_appended, use_cache=False, output_hidden_states=True
        )

    weights = dict(model.named_parameters())
    save_file(
        {
            "input_norm_weight": weights["model.layers.0.input_layernorm.weight"],
            "q_weight": weights["model.layers.0.self_attn.q_proj.weight"],
            "q_bias": weights["model.layers.0.self_attn.q_proj.bias"],
            "k_weight": weights["model.layers.0.self_attn.k_proj.weight"],
            "k_bias": weights["model.layers.0.self_attn.k_proj.bias"],
            "v_weight": weights["model.layers.0.self_attn.v_proj.weight"],
            "v_bias": weights["model.layers.0.self_attn.v_proj.bias"],
            "o_weight": weights["model.layers.0.self_attn.o_proj.weight"],
            "post_norm_weight": weights[
                "model.layers.0.post_attention_layernorm.weight"
            ],
            "gate_proj_weight": weights["model.layers.0.mlp.gate_proj.weight"],
            "up_proj_weight": weights["model.layers.0.mlp.up_proj.weight"],
            "down_proj_weight": weights["model.layers.0.mlp.down_proj.weight"],
            "in": model_outputs[1][0].squeeze(0),
            "out": model_outputs[1][1].squeeze(0),
            "in_next": model_outputs_appended[1][0][:, cached_tokens:].squeeze(),
            "out_next": model_outputs_appended[1][1][:, cached_tokens:].squeeze(),
        },
        os.path.join(data_dir, "transformer_block_test.safetensors"),
    )


def create_embedding_test_file(data_dir: str):
    table_size = 10
    dimension = 4
    embedding_layer = nn.Embedding(table_size, dimension, dtype=torch.bfloat16)
    embedding_table = torch.arange(
        table_size * dimension, dtype=torch.bfloat16
    ).reshape(table_size, dimension)
    embedding_layer.weight.data = embedding_table
    input_indices = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.int)
    with torch.no_grad():
        out = embedding_layer(input_indices)
    save_file(
        {
            "embedding_table": embedding_table,
            "input": input_indices.to(torch.bfloat16),
            "out": out,
        },
        os.path.join(data_dir, "embedding_test.safetensors"),
    )


def create_sampler_test_file(data_dir: str):
    logits = torch.tensor([[0.1, 0.2, 0.9, -0.5, -1.0]], dtype=torch.bfloat16)
    expected = torch.argmax(logits, dim=-1).to(torch.bfloat16)
    save_file(
        {"logits": logits, "expected": expected},
        os.path.join(data_dir, "sampler_test.safetensors"),
    )


def create_lm_head_test_file(data_dir: str):
    hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16)
    weight = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ],
        dtype=torch.bfloat16,
    )
    expected = hidden[-1] @ weight.T
    save_file(
        {"hidden": hidden, "weight": weight, "expected": expected},
        os.path.join(data_dir, "lm_head_test.safetensors"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create_test_data")
    parser.add_argument("data_dir", help="Test data directory", type=str)
    parsed = parser.parse_args()

    create_safetensors_test_file(parsed.data_dir)
    create_matmul_test_file(parsed.data_dir)
    create_dense_test_file(parsed.data_dir)
    create_dense_cache_test_file(parsed.data_dir)
    create_reshape_test_file(parsed.data_dir)
    create_rmsnorm_test_file(parsed.data_dir)
    create_softmax_test_file(parsed.data_dir)
    create_transformer_block_test_file(parsed.data_dir)
    create_embedding_test_file(parsed.data_dir)
    create_sampler_test_file(parsed.data_dir)
    create_lm_head_test_file(parsed.data_dir)
