"""Reproduce non-determinism of fused kernel under hardware resource contention.

Usage:
  # 1. checkout original kernel (with get_element/insert_slice)
  git checkout 5f3daa207 -- vllm_ascend/ops/triton/linearnorm/split_qkv_index_rmsnorm_rope.py

  # 2. run test
  python test_nondeterminism.py

  # 3. (optional) checkout fixed kernel and run again for comparison
  git checkout feat/m3-fused-qkv-index-rmsnorm-rope -- vllm_ascend/ops/triton/linearnorm/split_qkv_index_rmsnorm_rope.py
  python test_nondeterminism.py
"""

import argparse
import threading
import time

import numpy as np
import torch

import vllm_ascend.ops  # noqa: F401  registers torch.ops.vllm.qkv_index_rmsnorm_rope


def build_inputs(
    num_tokens=32,
    num_q_heads=64,
    num_kv_heads=4,
    head_dim=128,
    num_idx_heads=16,
    idx_head_dim=64,
    rope_dim=128,
    max_pos=262144,
    dtype=torch.bfloat16,
    device="npu:0",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    q_hidden = num_q_heads * head_dim       # 8192
    kv_hidden = num_kv_heads * head_dim     # 512
    index_q_size = num_idx_heads * idx_head_dim  # 1024
    total = q_hidden + 2 * kv_hidden + index_q_size + idx_head_dim

    qkv = torch.randn(num_tokens, total, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    iq_weight = torch.randn(idx_head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    ik_weight = torch.randn(idx_head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0

    cache = torch.from_numpy(
        np.random.uniform(0, 1, [max_pos, rope_dim])
    ).to(dtype).to(device).contiguous()

    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    return dict(
        input=qkv,
        cos_sin_cache=cache,
        positions=positions,
        q_weight=q_weight,
        k_weight=k_weight,
        index_q_weight=iq_weight,
        index_k_weight=ik_weight,
        q_hidden_size=q_hidden,
        kv_hidden_size=kv_hidden,
        index_q_size=index_q_size,
        head_dim=head_dim,
        idx_head_dim=idx_head_dim,
        eps=1e-6,
        attn_out_fp8=True,
        indexer_out_fp8=True,
        q_bias=None,
        k_bias=None,
    )


def background_workload(stop_flag, device, matrix_size=4096):
    """Run memory-intensive ops on a side stream to create NPU resource contention."""
    bg_stream = torch.npu.Stream(device=device)
    with torch.npu.stream(bg_stream):
        while not stop_flag.is_set():
            a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.bfloat16)
            b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.bfloat16)
            _ = torch.mm(a, b)
    bg_stream.synchronize()


def run_kernel(inputs, clone_inputs=True):
    if clone_inputs:
        kwargs = dict(inputs)
        kwargs["input"] = inputs["input"].clone()
    else:
        kwargs = inputs
    return torch.ops.vllm.qkv_index_rmsnorm_rope(**kwargs)


def compare_outputs(results, label=""):
    n = len(results)
    diffs = []
    for i in range(1, n):
        max_diff = 0.0
        for a, b in zip(results[0], results[i]):
            d = (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()
            max_diff = max(max_diff, d)
        diffs.append(max_diff)
    nondet = any(d > 0 for d in diffs)
    status = "NON-DETERMINISTIC" if nondet else "deterministic"
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Result: {status}")
    print(f"{'=' * 60}")
    for i, d in enumerate(diffs, 1):
        flag = " <-- DIFF" if d > 0 else ""
        print(f"  Run {i:2d} vs Run  0: max_diff = {d:.6e}{flag}")
    return nondet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=20, help="number of kernel runs")
    parser.add_argument("--num-tokens", type=int, default=32, help="batch size")
    parser.add_argument("--no-contention", action="store_true",
                        help="disable background stream contention (baseline)")
    parser.add_argument("--matrix-size", type=int, default=4096,
                        help="background mm matrix size (contention strength)")
    parser.add_argument("--bg-streams", type=int, default=1,
                        help="number of background workload threads")
    args = parser.parse_args()

    device = "npu:0"
    inputs = build_inputs(num_tokens=args.num_tokens, device=device)
    print(f"Config: tokens={args.num_tokens}, bg_streams={args.bg_streams}, "
          f"matrix={args.matrix_size}x{args.matrix_size}, runs={args.n_runs}")

    # start background contention
    stop_flags = []
    bg_threads = []
    if not args.no_contention:
        for _ in range(args.bg_streams):
            sf = threading.Event()
            t = threading.Thread(target=background_workload, args=(sf, device, args.matrix_size))
            t.daemon = True
            t.start()
            stop_flags.append(sf)
            bg_threads.append(t)
        print(f"Background contention: {args.bg_streams} stream(s) active")
        time.sleep(0.5)  # let background ramp up
    else:
        print("Background contention: OFF (baseline)")

    # run kernel N times
    results = []
    for i in range(args.n_runs):
        torch.npu.synchronize()
        out = run_kernel(inputs)
        torch.npu.synchronize()
        results.append([o.clone() for o in out])
        if (i + 1) % 5 == 0:
            print(f"  ...completed {i + 1}/{args.n_runs} runs")

    # stop background
    for sf in stop_flags:
        sf.set()
    for t in bg_threads:
        t.join(timeout=5)

    # compare
    label_parts = []
    if args.no_contention:
        label_parts.append("NO contention")
    else:
        label_parts.append(f"WITH contention ({args.bg_streams} stream)")
    label_parts.append(f"{args.num_tokens} tokens")
    label_parts.append(f"{args.n_runs} runs")
    compare_outputs(results, label=" | ".join(label_parts))


if __.name__ == "__main__":
    main()
