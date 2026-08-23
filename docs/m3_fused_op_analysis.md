# MiniMax-M3 融合算子开发与确定性问题分析

## 1. 项目概述

为 MiniMax-M3 模型的稀疏注意力准备阶段（`_sparse_prepare`）开发融合 Triton 算子，将原本分散的多个算子合并为单次 kernel 调用，减少 GM 往返开销。

### 1.1 原始流程（未融合）

```
qkv_proj → concat [q | k | v | index_q | index_k]
  → narrow/split × 5              (~9ms, 5个 aclnnInplaceCopy_Slice)
  → npu_rms_norm × 4              (~13ms, 4个 RmsNorm)
  → npu_rotary_embedding × 2      (~8ms, 2个 _triton_rope)
  → clamp ±448 × 4                (~6ms, 4个 ClipByValueV2)
  → cast →fp8 × 4                 (~6ms, 4个 Cast)
  总计: ~42ms
```

### 1.2 融合后流程

```
qkv_proj → concat
  → Python: cos_sin_cache[positions] (预取 cos/sin, ~1.5ms)
  → Triton kernel (单次调用):
      split (按 offset load)
      + Gemma RMSNorm (tl.sum)
      + NeoX RoPE (向量 cos/sin)
      + clamp ±448
      + cast →fp8
  总计: ~10ms
```

### 1.3 性能收益

| 指标 | 原始 | 融合后 | 收益 |
|---|---|---|---|
| 耗时 | ~42ms | ~10ms | ~76% |
| GM 往返 | 11+ 次 | 2 次 | 大幅减少 |
| 独立算子数 | 11+ | 1 (+1 Python 预取) | 大幅减少 |

---

## 2. 融合算子设计

### 2.1 算子名称

`torch.ops.vllm.qkv_index_rmsnorm_rope`（注册名 `qkv_index_rmsnorm_rope`）

### 2.2 输入/输出

**输入**:
- `input`: concat 张量 `[batch, q_size + 2*kv_size + index_q_size + idx_head_dim]`
- `cos_sin_cache`: RoPE 缓存 `[max_pos, rope_dim]`
- `positions`: token 位置 `[batch]`
- `q_weight / k_weight / index_q_weight / index_k_weight`: Gemma RMSNorm 的 1+w

**输出**: 5 个张量 `q_out, k_out, v_out, index_q_out, index_k_out`

### 2.3 Kernel 结构

```
grid = (num_vectorcore,)
每个 Vector Core 处理一段 token:

段1: main QK
  ├── load concat [q|k] by offset
  ├── load pre-gathered cos/sin (向量化)
  ├── Gemma RMSNorm (tl.sum, HEAD_DIM 归约)
  ├── ×(1+w) 权重乘
  ├── NeoX RoPE: [x1*cos - x2*sin | x2*cos + x1*sin]
  ├── clamp ±448 (if ATTN_OUT_FP8)
  └── store (cast on store)

段2: V
  ├── load concat [v] by offset
  ├── clamp ±448 (if ATTN_OUT_FP8)
  └── store (cast on store)

段3: indexer QK
  ├── load concat [index_q|index_k] by offset
  ├── load pre-gathered cos/sin (向量化)
  ├── Gemma RMSNorm (tl.sum, IDX_HEAD_DIM 归约)
  ├── ×(1+w) 权重乘
  ├── NeoX RoPE (IDX_HALF 半宽)
  ├── clamp ±448 (if INDEX_OUT_FP8)
  └── store (cast on store)
```

### 2.4 关键设计决策

1. **FP8 输出 + attn_out 强制 bf16**: `attn_out = torch.empty_like(q, dtype=torch.bfloat16)`，使 Q/K/V 输出 fp8（融合 clamp+cast），同时 attn_out 保持 bf16 喂给 o_proj 的 `npu_dynamic_mx_quant`（拒收 fp8）

2. **cos/sin 预取**: 在 Python 层用 `cos_sin_cache[positions]` 预取 cos/sin，传入 kernel，避免 kernel 内标量循环

3. **UB 感知 tiling**: 按 NPU 卡型（A2=192KB, A5=248KB）自动调整 tile 大小

4. **int64 GM 偏移**: 大 batch × hidden_size 超过 int32 范围时用 int64

---

## 3. 确定性问题

### 3.1 现象

bs1 贪婪采样下，使用融合算子时，相同输入多次运行输出长度不固定；去掉融合算子走 fallback 路径则输出长度固定。

### 3.2 根因

**Ascend Triton 的 `get_element` + `insert_slice` 标量循环导致指令调度不确定**

原 kernel 中 cos/sin 收集使用标量循环：

```python
for i in tl.range(batch_tile):
    pos = get_element(x, (i,))           # 标量提取 position
    cache_rows = insert_slice(            # 标量写回 UB 缓冲区
        cache_rows,
        tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None]),
        offsets=(i, 0), sizes=(1, ROPE_DIM), strides=(1, 1),
    )
```

Ascend Triton 编译器对 `get_element` + `insert_slice` 循环模式的指令调度在不同运行时不固定，导致：
- cos/sin 值出现 ULP 级差异
- RoPE 结果微小不同
- 贪婪采样时 top-1 logits 翻转
- 输出路径分叉 → 输出长度不同

### 3.3 消融实验定位过程

通过两组消融实验排除了 `tl.sum` 的嫌疑，定位到 RoPE 的 cos/sin 标量收集循环：

| 实验 | RMSNorm | RoPE | 确定性 | 结论 |
|---|---|---|---|---|
| 原融合 kernel | Triton `tl.sum` | Triton（含 get_element 循环） | ❌ 非确定 | 有问题 |
| 消融 B | Triton `tl.sum`（独立 kernel） | C++ `npu_rotary_embedding` | ✅ 确定 | `tl.sum` 本身不是问题 |
| 消融 A | C++ `npu_rms_norm` | Triton kernel（含 get_element 循环） | ❌ 非确定 | **RoPE 是问题** |

**消融 B** 证明 `tl.sum` 在独立简单 kernel 中是确定的；**消融 A** 证明只要 Triton kernel 里有 RoPE（含 get_element 循环），就非确定。

### 3.4 其他排除的方案

| 方案 | 结果 | 原因 |
|---|---|---|
| `tl.dot` 替代 `tl.sum` | ❌ 编译器崩溃 | hivmc-a5 断言失败 |
| `extract_slice` 逐列累加 | ❌ 输出乱码 | extract_slice 在循环中行为异常 |
| `sq[:, d]` 标量索引 | ❌ 编译失败 | `unsupported tensor index: int32[]` |
| `blk[:, 0]` constexpr 索引 | ❌ 编译失败 | `unsupported tensor index: constexpr[0]` |
| `num_stages=1` 禁用流水线 | ❌ 不支持 | Ascend Triton 无此参数 |
| `npu_rms_norm` + 独立 RoPE kernel | ❌ 各种问题 | slice 索引不支持 / worker 崩溃 / 精度问题 |

### 3.5 解决方法

**把 cos/sin 收集从 kernel 内的标量循环移到 Python 层的向量化索引**：

#### Python 层（impl）

```python
# 预取 cos/sin（确定，torch indexing = C++ op）
cos_sin_gathered = cos_sin_cache[positions]           # [batch, cache_dim]
cos_gathered = cos_sin_gathered[:, :max_half]         # [batch, max_half]
sin_gathered = cos_sin_gathered[:, cache_dim//2 : cache_dim//2 + max_half]
```

#### Kernel 层

```python
# 修改前（非确定）：标量循环
for i in tl.range(batch_tile):
    pos = get_element(x, (i,))
    cache_rows = insert_slice(cache_rows, tl.load(pos * ROPE_DIM + ...), ...)

# 修改后（确定）：向量化 load，无循环
cos = tl.load(cos_gm_ptr + row64[:, None] * ATTN_HALF + cos_qk_range[None, :],
              mask=mmask[:, None]).to(tl.float32)
sin = tl.load(sin_gm_ptr + row64[:, None] * ATTN_HALF + cos_qk_range[None, :],
              mask=mmask[:, None]).to(tl.float32)
```

#### 改动点

1. **kernel 签名**: `positions_gm_ptr` + `cos_sin_cache_gm_ptr` → `cos_gm_ptr` + `sin_gm_ptr`
2. **main QK**: 删掉 `get_element` + `insert_slice` 循环，换成向量化 `tl.load`
3. **indexer**: 同上
4. **impl**: Python 层预取 cos/sin，传给 kernel
5. **kernel launch**: 传 `cos_gathered` / `sin_gathered`

### 3.6 性能影响

| 操作 | 变化 | 耗时影响 |
|---|---|---|
| `cos_sin_cache[positions]` | 新增 | +1.5ms |
| `get_element` + `insert_slice` 循环 | 删除 | -1.5ms |
| 净影响 | — | **~0ms** |

cos/sin 数据量远小于 Q/K/V（`batch × max_half` vs `batch × q_size`），预取消耗可忽略。

---

## 4. FP8 优化

### 4.1 问题

原始流程中，Q/K/V 的 fp8 clamp+cast 分散在多处：
- `_insert_kv`: K/V 写入 KV cache 前的 clamp+cast（~6ms）
- `_to_fp8` (`msa_m3_npu_new.py`): Q 喂给 `npu_sparse_attention_score` 前的 clamp+cast（~5ms）

### 4.2 解决方法

在融合 kernel 内统一做 fp8 clamp+cast，输出即为 fp8：

```python
# kernel 内：
if ATTN_OUT_FP8:
    q_heads = tl.minimum(tl.maximum(q_heads, -448.0), 448.0)
tl.store(q_gm_ptr + ..., q_heads.to(q_gm_ptr.dtype.element_ty))  # cast on store
```

下游处理：
- `_insert_kv`: 检测到 K/V 已是 fp8，跳过 clamp+cast
- `_to_fp8`: 检测到 Q 已是 fp8，跳过 clamp+cast

### 4.3 dtype 兼容性

Q 的 fp8 输出不影响 `o_proj`（`npu_dynamic_mx_quant`），因为：
```python
attn_out = torch.empty_like(q, dtype=torch.bfloat16)  # 强制 bf16
```

---

## 5. 文件清单

| 文件 | 修改内容 |
|---|---|
| `vllm_ascend/ops/triton/linearnorm/split_qkv_index_rmsnorm_rope.py` | 融合算子 kernel + impl + 注册 |
| `vllm_ascend/ops/__init__.py` | 导入新算子模块 |
| `vllm_ascend/attention/msa_m3.py` | `_sparse_prepare` 接入融合算子；`_insert_kv` 跳过 fp8 clamp+cast；`empty_like` 强制 bf16 |
| `vllm_ascend/attention/msa_m3_npu_new.py` | `_to_fp8` 跳过已 fp8 输入 |
| `tests/.../test_split_qkv_index_rmsnorm_rope.py` | 参数化测试 |

---

## 6. Git 历史

分支: `feat/m3-fused-qkv-index-rmsnorm-rope`
仓库: `gxy9808/vllm-ascend-ShangHai-LLM`

关键 commit:
- `74b83d3cf` — 初始融合算子 + 测试
- `ec88306fb` — `empty_like(q, dtype=bf16)` + `attn_out_fp8=True`
- `b32dd5701` — `_to_fp8` 跳过 fp8 输入
- `9d11a5c03` — cos/sin 预取，消除非确定性
- `c4308e903` — 修复变量定义顺序
