# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Unit tests for the Step4 DSA torch operators on Ascend.

References mirror the standalone Step-4 minimal inference kernels
(step4-inference/inference/kernel.py): scalar-shift softmax summaries with
fp32->bf16->e4m3 double rounding, weighted-ReLU indexer scoring with e4m3
activation rounding, history-only top-k candidate ranges, and sparse
attention over gathered regions. CPU-runnable.
"""

import math

import pytest
import torch

from vllm_ascend.models.step4.dsa import (
    REGION_BLOCK_SIZE,
    _gather_region_summaries,
    _quantize_e4m3,
    _select_topk_regions,
    csa_compress_regions,
    indexer_logits,
    sparse_attention,
)


def _reference_compress(k, z, start, count, region_size):
    """Per-region softmax(z).k with a scalar max shift, then e4m3 rounding."""
    d = k.shape[-1]
    out = torch.zeros(count.shape[0], d)
    for r in range(count.shape[0]):
        if int(count[r]) == 0:
            continue
        zk = z[start[r] : start[r] + int(count[r])].float()
        kk = k[start[r] : start[r] + int(count[r])].float()
        shift = zk.max()
        w = torch.exp(zk - shift)
        denom = w.sum(dim=0)
        numer = (w * kk).sum(dim=0)
        out[r] = torch.where(denom > 0, numer / denom.clamp_min(1e-20), 0.0)
    return out.to(torch.bfloat16).to(torch.float8_e4m3fn).float()


def test_csa_compress_regions_matches_reference():
    torch.manual_seed(0)
    tokens, d = 24, 32
    k = torch.randn(tokens, 1, d, dtype=torch.bfloat16)
    z = torch.randn(tokens, 1, d, dtype=torch.bfloat16)
    starts = torch.tensor([0, 8, 16], dtype=torch.int32)
    counts = torch.tensor([8, 8, 8], dtype=torch.int32)

    got = csa_compress_regions(k, z, starts, counts).float()
    ref = _reference_compress(
        k.reshape(tokens, d), z.reshape(tokens, d), starts, counts, REGION_BLOCK_SIZE
    )
    torch.testing.assert_close(got.reshape(3, d), ref, rtol=0.0, atol=0.0)


def test_csa_compress_regions_partial_region():
    torch.manual_seed(1)
    tokens, d = 12, 16
    k = torch.randn(tokens, 1, d, dtype=torch.bfloat16)
    z = torch.randn(tokens, 1, d, dtype=torch.bfloat16)
    starts = torch.tensor([0, 8], dtype=torch.int32)
    counts = torch.tensor([8, 4], dtype=torch.int32)

    got = csa_compress_regions(k, z, starts, counts).float()
    ref = _reference_compress(
        k.reshape(tokens, d), z.reshape(tokens, d), starts, counts, REGION_BLOCK_SIZE
    )
    torch.testing.assert_close(got.reshape(2, d), ref, rtol=0.0, atol=0.0)


def test_indexer_logits_weighted_relu():
    torch.manual_seed(2)
    tokens, heads, regions, d = 5, 4, 7, 64
    q = torch.randn(tokens, heads, d, dtype=torch.bfloat16)
    w = torch.randn(tokens, heads, dtype=torch.float32) * 0.5
    summaries = torch.randn(regions, 1, d, dtype=torch.bfloat16)

    got = indexer_logits(q, w, summaries)

    qq = _quantize_e4m3(q).float()
    kk = _quantize_e4m3(summaries.reshape(regions, d)).float()
    ref = torch.zeros(tokens, regions)
    for t in range(tokens):
        for r in range(regions):
            total = 0.0
            for h in range(heads):
                dot = float((qq[t, h] * kk[r]).sum())
                total += max(dot, 0.0) * float(w[t, h])
            ref[t, r] = total
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_select_topk_regions_history_range_and_padding():
    torch.manual_seed(3)
    rows, max_regions, topk = 4, 8, 4
    scores = torch.randn(rows, max_regions)
    counts = torch.tensor([0, 2, 5, 8], dtype=torch.long)

    picked = _select_topk_regions(scores, counts, topk)

    assert picked.shape == (rows, topk)
    # Row 0 has no history: all padding.
    assert torch.all(picked[0] == -1)
    for r, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        live = picked[r][picked[r] >= 0]
        # Candidates are strictly within [0, count), unique, ascending.
        assert torch.all(live < count)
        assert torch.all(live >= 0)
        assert torch.all(live[1:] > live[:-1]) or live.numel() <= 1
        # The chosen set equals the top-(min(count, topk)) scores.
        k = min(count, topk)
        ref = torch.topk(scores[r, :count], k).indices.sort().values
        torch.testing.assert_close(live, ref)


def test_sparse_attention_matches_dense_on_selected_regions():
    torch.manual_seed(4)
    tokens, heads, d = 3, 4, 32
    kv_heads, block_size, num_blocks = 1, 8, 6
    region_size = 4
    scale = d**-0.5

    kv = torch.randn(2, num_blocks, block_size, kv_heads, d, dtype=torch.bfloat16)
    flat_k = kv[0].reshape(-1, kv_heads, d)
    flat_v = kv[1].reshape(-1, kv_heads, d)
    q = torch.randn(tokens, heads, d, dtype=torch.bfloat16)

    # Single request owning all tokens; logical == physical layout.
    block_table = torch.arange(num_blocks).view(1, -1)
    request_ids = torch.zeros(tokens, dtype=torch.long)
    # Row 0 -> region 1 only; row 1 -> regions {0, 3}; row 2 -> region 2, 3 valid of 4.
    regions = torch.tensor(
        [
            [1, -1, -1],
            [0, 3, -1],
            [2, -1, -1],
        ],
        dtype=torch.long,
    )
    valid = torch.tensor(
        [
            [4, 0, 0],
            [4, 4, 0],
            [4, 0, 0],
        ],
        dtype=torch.long,
    )

    got = sparse_attention(
        q,
        kv,
        block_table,
        request_ids,
        regions,
        valid,
        scale=scale,
        region_size=region_size,
    )

    for t in range(tokens):
        keys, values = [], []
        for slot, region in enumerate(regions[t].tolist()):
            if region < 0:
                continue
            for j in range(int(valid[t, slot])):
                tok = region * region_size + j
                keys.append(flat_k[tok, 0])
                values.append(flat_v[tok, 0])
        keys = torch.stack(keys).float()
        values = torch.stack(values).float()
        scores = (q[t].float() @ keys.T) * scale
        probs = torch.softmax(scores, dim=-1)
        ref = (probs @ values).to(torch.bfloat16)
        torch.testing.assert_close(got[t], ref, rtol=1e-3, atol=1e-3)


def test_sparse_attention_empty_selection_is_zero():
    tokens, heads, d = 2, 2, 16
    kv = torch.randn(2, 2, 8, 1, d, dtype=torch.bfloat16)
    q = torch.randn(tokens, heads, d, dtype=torch.bfloat16)
    got = sparse_attention(
        q,
        kv,
        torch.arange(2).view(1, -1),
        torch.zeros(tokens, dtype=torch.long),
        torch.full((tokens, 2), -1, dtype=torch.long),
        torch.zeros(tokens, 2, dtype=torch.long),
        scale=0.1,
        region_size=4,
    )
    assert torch.all(got == 0)


def test_gather_region_summaries_follows_block_table():
    torch.manual_seed(5)
    regions_per_block, d = 4, 8
    # 3 physical blocks x 4 regions each; block 1 is the request's second page.
    cache = torch.randn(3 * regions_per_block, 1, d).to(torch.float8_e4m3fn)
    block_table = torch.tensor([[0, 2]])
    got = _gather_region_summaries(cache, block_table[0], 8, regions_per_block)
    expected = torch.cat(
        [cache[0:4], cache[2 * regions_per_block : 3 * regions_per_block]]
    ).squeeze(1)
    torch.testing.assert_close(got.float(), expected.float())


def test_e4m3_rounding_changes_values():
    torch.manual_seed(6)
    x = torch.randn(1024, dtype=torch.bfloat16)
    rounded = _quantize_e4m3(x)
    assert not torch.equal(rounded, x)
    assert torch.isfinite(rounded).all()
