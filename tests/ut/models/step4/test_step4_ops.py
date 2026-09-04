# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Unit tests for the Step4 torch operator port.

The operators mirror the numerics of the standalone Step-4 minimal inference
release (step4-inference/inference/kernel.py) and the CUDA port's native
fallbacks. These tests run on CPU against reference implementations.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.step4.kernels import (
    Step4SparseConfig,
    checkpoint_has_step4_sparse_config,
    fused_indexer_norm_rope_forward_impl,
    fused_qknorm_rope_forward_impl,
    get_step4_sparse_config,
    router_bias_func,
)
from vllm_ascend.models.step4.layernorm import (
    OptimusLayerNorm,
    OptimusRMSNorm,
    _optimus_rms_norm_native,
)


def _build_rope_tables(
    max_position: int, rotary_span: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    half = rotary_span // 2
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_span, 2, dtype=torch.float32) / rotary_span
        )
    )
    freqs = torch.outer(
        torch.arange(max_position, dtype=torch.float32), inv_freq
    )
    assert freqs.shape[1] == half
    return freqs.cos(), freqs.sin()


def _reference_rms_norm_per_head(
    x: torch.Tensor, weight: torch.Tensor, head_dim: int, eps: float, bias: float
) -> torch.Tensor:
    out = torch.empty_like(x)
    num_heads = x.shape[-1] // head_dim
    for h in range(num_heads):
        chunk = x[..., h * head_dim : (h + 1) * head_dim].float()
        var = chunk.pow(2).mean(dim=-1, keepdim=True)
        normed = chunk * torch.rsqrt(var + eps) * (weight.float() + bias)
        out[..., h * head_dim : (h + 1) * head_dim] = normed.to(x.dtype)
    return out


def _reference_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    pairs: int,
) -> torch.Tensor:
    out = x.clone()
    num_heads = x.shape[-1] // head_dim
    for t, pos in enumerate(positions.tolist()):
        c = cos[pos].float()
        s = sin[pos].float()
        for h in range(num_heads):
            base = h * head_dim
            x1 = x[t, base : base + pairs].float()
            x2 = x[t, base + pairs : base + 2 * pairs].float()
            out[t, base : base + pairs] = (x1 * c - x2 * s).to(x.dtype)
            out[t, base + pairs : base + 2 * pairs] = (
                x2 * c + x1 * s
            ).to(x.dtype)
    return out


@pytest.mark.parametrize(
    "num_q_head,num_kv_head,head_dim,partial",
    [
        (4, 2, 128, 1.0),
        (8, 2, 128, 0.5),
        (4, 1, 192, 1.0),
        (2, 2, 64, 0.5),
    ],
)
def test_fused_qknorm_rope(num_q_head, num_kv_head, head_dim, partial):
    torch.manual_seed(0)
    tokens, max_pos = 7, 64
    rotary_span = int(head_dim * partial) // 2 * 2
    pairs = rotary_span // 2
    dtype = torch.bfloat16

    qkv = torch.randn(tokens, (num_q_head + 2 * num_kv_head) * head_dim, dtype=dtype)
    q_w = torch.randn(head_dim, dtype=torch.float32)
    k_w = torch.randn(head_dim, dtype=torch.float32)
    cos, sin = _build_rope_tables(max_pos, rotary_span, 10000.0)
    positions = torch.randint(0, max_pos, (tokens,))
    eps = 1e-5

    q, k, v = fused_qknorm_rope_forward_impl(
        qkv, q_w, k_w, cos, sin, positions,
        head_dim, num_q_head, num_kv_head, pairs, eps, norm_weight_bias=1.0,
    )

    q_size = num_q_head * head_dim
    kv_size = num_kv_head * head_dim
    q_ref, k_ref, v_ref = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q_ref = _reference_rms_norm_per_head(q_ref.float(), q_w, head_dim, eps, 1.0)
    k_ref = _reference_rms_norm_per_head(k_ref.float(), k_w, head_dim, eps, 1.0)
    q_ref = _reference_rope(q_ref, cos, sin, positions, head_dim, pairs)
    k_ref = _reference_rope(k_ref, cos, sin, positions, head_dim, pairs)

    torch.testing.assert_close(q.float(), q_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k.float(), k_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v, v_ref)

    # The rotated span ordering matches NeoX pairs, not interleaved pairs.
    tokens_one = torch.tensor([3])
    x = torch.zeros(1, head_dim)
    x[0, 0] = 1.0
    cos_one = torch.ones(1, pairs)
    sin_one = torch.zeros(1, pairs)
    sin_one[0, 1] = 1.0
    from vllm_ascend.models.step4.kernels import _apply_neox_partial_rope

    rotated = _apply_neox_partial_rope(
        x, cos_one, sin_one, tokens_one, 1, head_dim, pairs
    )
    assert rotated[0, 1].item() == pytest.approx(-1.0)
    assert rotated[0, pairs + 1].item() == pytest.approx(0.0)


def test_fused_indexer_norm_rope():
    torch.manual_seed(1)
    tokens, proxy_dim, rope_span = 5, 64, 16
    num_q_head, num_k_head = 4, 1
    pairs = rope_span // 2
    dtype = torch.bfloat16

    index_q = torch.randn(tokens, num_q_head * proxy_dim, dtype=dtype)
    index_k = torch.randn(tokens, num_k_head * proxy_dim, dtype=dtype)
    index_z = torch.randn(tokens, num_k_head * proxy_dim, dtype=dtype)
    q_w = torch.randn(proxy_dim, dtype=torch.float32)
    k_w = torch.randn(proxy_dim, dtype=torch.float32)
    k_b = torch.randn(proxy_dim, dtype=torch.float32)
    cos, sin = _build_rope_tables(32, rope_span, 10000.0)
    positions = torch.randint(0, 32, (tokens,))
    eps = 1e-6

    q, k, z = fused_indexer_norm_rope_forward_impl(
        index_q, index_k, index_z, q_w, k_w, k_b, cos, sin, positions,
        proxy_dim, num_q_head, num_k_head, pairs, eps, q_norm_weight_bias=1.0,
    )

    q_ref = _reference_rms_norm_per_head(index_q.float(), q_w, proxy_dim, eps, 1.0)
    q_ref = _reference_rope(q_ref, cos, sin, positions, proxy_dim, pairs)
    # LayerNorm reference with mean subtraction and affine bias.
    k_flat = index_k.float().reshape(tokens * num_k_head, proxy_dim)
    k_normed = torch.nn.functional.layer_norm(k_flat, (proxy_dim,), k_w, k_b, eps)
    k_ref = _reference_rope(
        k_normed.reshape(tokens, num_k_head * proxy_dim).to(dtype),
        cos, sin, positions, proxy_dim, pairs,
    )

    torch.testing.assert_close(q.float(), q_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k.float(), k_ref.float(), rtol=2e-2, atol=2e-2)
    assert z.data_ptr() != index_z.data_ptr()
    torch.testing.assert_close(z, index_z.contiguous())


@pytest.mark.parametrize("topk,scaling", [(4, 1.0), (8, 3.0)])
def test_router_bias_func_matches_iterative_reference(topk, scaling):
    torch.manual_seed(2)
    num_tokens, num_experts = 16, 32
    gating = torch.randn(num_tokens, num_experts, dtype=torch.float32) * 3
    bias = torch.randn(num_experts, dtype=torch.float32)

    weights, ids = router_bias_func(
        gating[:, :],
        gating,
        topk,
        True,
        router_bias=bias,
        routed_scaling_factor=scaling,
        indices_dtype=torch.int64,
    )

    gate_prob = torch.sigmoid(gating)
    scores = gate_prob + bias.unsqueeze(0)
    ref_weights = torch.zeros(num_tokens, topk)
    ref_ids = torch.zeros(num_tokens, topk, dtype=torch.int64)
    for t in range(num_tokens):
        row = scores[t].clone()
        total = 0.0
        for slot in range(topk):
            idx = int(torch.argmax(row).item())
            w = (row[idx] - bias[idx]).item()
            ref_weights[t, slot] = w
            ref_ids[t, slot] = idx
            total += w
            row[idx] = float("-inf")
        ref_weights[t] /= total + 1e-20
        ref_weights[t] *= scaling

    torch.testing.assert_close(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights, rtol=1e-5, atol=1e-6)


def test_router_bias_func_nan_row():
    torch.manual_seed(3)
    gating = torch.randn(3, 8)
    gating[1, 4] = float("nan")
    bias = torch.randn(8)
    weights, ids = router_bias_func(
        gating, gating, 2, True, router_bias=bias, nan_row_i_out=7
    )
    assert torch.all(weights[1] == 0).item()
    assert torch.all(ids[1] == 7).item()
    assert weights[0].abs().sum() > 0
    assert ids.dtype == torch.int32


def test_optimus_rms_norm_zero_centered():
    torch.manual_seed(4)
    x = torch.randn(6, 32, dtype=torch.bfloat16)
    weight = torch.randn(32, dtype=torch.float32) * 0.1

    out = _optimus_rms_norm_native(x, weight, 1e-5, True)
    ref = torch.nn.functional.rms_norm(
        x.float(), (32,), weight + 1.0, 1e-5
    ).to(x.dtype)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)

    module = OptimusRMSNorm(32, eps=1e-5, zero_centered=True, dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(weight)
    torch.testing.assert_close(module(x), out)

    module_nc = OptimusRMSNorm(32, eps=1e-5, zero_centered=False)
    out_nc, residual_nc = module_nc(x, x.clone())
    expected_residual = (x.float() + x.float()).to(x.dtype)
    expected_out = torch.nn.functional.rms_norm(
        expected_residual.float(), (32,), module_nc.weight.float(), 1e-5
    ).to(x.dtype)
    torch.testing.assert_close(residual_nc, expected_residual)
    torch.testing.assert_close(out_nc.float(), expected_out.float(), rtol=1e-2, atol=1e-2)


def test_optimus_layer_norm_per_head():
    torch.manual_seed(5)
    heads, head_dim, tokens = 3, 16, 4
    x = torch.randn(tokens, heads * head_dim)
    module = OptimusLayerNorm(head_dim, eps=1e-6)
    with torch.no_grad():
        module.weight.copy_(torch.randn(head_dim))
        module.bias.copy_(torch.randn(head_dim))
    out = module(x)
    for h in range(heads):
        ref = torch.nn.functional.layer_norm(
            x[:, h * head_dim : (h + 1) * head_dim],
            (head_dim,),
            module.weight,
            module.bias,
            1e-6,
        )
        torch.testing.assert_close(out[:, h * head_dim : (h + 1) * head_dim], ref)


def test_sparse_config_parsing():
    dense_config = SimpleNamespace(step4_sparse_config=None)
    assert checkpoint_has_step4_sparse_config(dense_config) is False
    assert get_step4_sparse_config(dense_config) is None

    section = {
        "enabled": True,
        "sparse_indexer_num_heads": 16,
        "index_tp_size": 4,
        "topk": 512,
    }
    sparse_checkpoint = SimpleNamespace(sparse_config=section)
    assert checkpoint_has_step4_sparse_config(sparse_checkpoint) is True
    cfg = get_step4_sparse_config(sparse_checkpoint)
    assert cfg is not None
    assert cfg.sparse_indexer_num_heads == 16
    assert cfg.index_tp_size == 4
    assert cfg.topk == 512
    assert cfg.apply_to_layer_types == ("full_attention",)

    disabled = SimpleNamespace(sparse_config={"enabled": False})
    assert get_step4_sparse_config(disabled) is None
    assert checkpoint_has_step4_sparse_config(disabled) is True

    bad = SimpleNamespace(sparse_config={"enabled": True, "proxy_dim": 128})
    with pytest.raises(ValueError, match="proxy_dim"):
        get_step4_sparse_config(bad)

    defaults = Step4SparseConfig()
    assert defaults.proxy_dim == 256
    assert defaults.region_block_size == 8
    assert math.isfinite(defaults.sparse_indexer_rope_dim)


def test_sparse_config_env_force_disable(monkeypatch):
    section = {"enabled": True}
    checkpoint = SimpleNamespace(sparse_config=section)
    monkeypatch.setenv("VLLM_STEP4_SPARSE", "0")
    assert get_step4_sparse_config(checkpoint) is None
