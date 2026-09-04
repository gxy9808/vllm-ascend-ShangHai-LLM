# Step4 (StepFun) vllm-ascend 适配文档

> 分支：`step4-0.27.1`（基于 `releases/v0.27.1rc`）
> 源参考：vllm-gxy step4 分支 commit `940fcaed83`（vllm v0.27.0 patch）+ `1246cded78`（独立最小推理实现，数值参考）

## 更新记录

- **第二阶段（当前）**：DSA 稀疏注意力 Ascend 原生适配（torch 实现）。DSA checkpoint 不再被拒绝，full-attention 层原生执行稀疏注意力；`VLLM_STEP4_SPARSE=0` 仍可强制 dense fallback。见 §7。
- **第一阶段**：dense 路径完整移植 + Step4 专有算子 torch 化（§2–§6）。

---

## 1. 背景

Step4 是 StepFun 的 GQA + MoE + 稀疏注意力（DSA）混合架构模型。其 vLLM 适配（vllm-gxy）分为两部分：

| 组成 | 内容 | 位置 |
|---|---|---|
| **vllm core patch** | 对 vllm 主干 70+ 文件的修改（`valid_vocab_size`、KV cache sidecar 契约、flash-attn split-KV 后端、MTP proposer、envs 等） | `step4-vllm-v0.27.0-full.patch` |
| **step4 模型包** | `vllm/models/step4/`（model/mtp/kernels/layernorm + nvidia SM90 CuTeDSL 稀疏内核） | 独立目录，可整体移植 |
| **独立推理版** | `step4-inference/`：Triton + 纯 torch 的最小推理实现，数值对齐参考 | `step4-inference/inference/` |

vllm-ascend 是 pip 依赖 vllm v0.27.1 的硬件插件，**不能修改 vllm core**。因此适配策略为：模型包整体移植到插件内、core patch 的功能用 0.27.1 原生 API 等价实现、CUDA 专有算子按独立推理版的数值语义用 torch 重写。

---

## 2. 适配思路与总体方案

### 2.1 版本对齐（前置核对）

vllm-gxy 基于 v0.27.0 + patch，vllm-ascend 依赖 v0.27.1。适配前逐项核对了 patch 所依赖 API 在 0.27.1 的存在性与签名：

| patch 依赖 | 0.27.1 状态 | 适配处理 |
|---|---|---|
| `Attention.forward(kv_cache_dummy_dep=...)` | **已进上游** | 直接使用 |
| `Attention(attn_backend=...)` 参数 | **已进上游** | 直接使用（本适配未用到，DSA 后端才有） |
| `Step4FusedQKVIndexerLinear`（qkv+indexer 融合 GEMM） | 依赖 patch 语义 | 暂不移植（DSA 专用），见 §4.2 |
| `LogitsProcessor(valid_vocab_size=...)` | 不存在 | 用原生 `org_vocab_size`（`logits[..., :org]` 掩码，语义等价） |
| `SwigluStepAndMul(compile_native=...)` | 0.27.1 无该参数 | 去掉参数 |
| `FusedMoEFactory(custom_routing_function=..., router_logits_dtype=..., e_score_correction_bias=..., activation="swiglustep")` | 全部存在 | 直接使用 |
| `get_attention_context` / `unified_kv_cache_update` | 已进上游 | 不再引用（dense 路径不需要） |
| `envs.VLLM_STEP4_*` | 不存在 | 集中注册到 `vllm_ascend/envs.py`（AGENTS.md 要求），保留原名 |
| `ModelRegistry.register_model` | 存在 | 插件注册模式（`vllm_ascend/models/__init__.py`） |
| config 注册（`step4`/`step4_mtp` → Step4Config） | core patch 内容 | 运行时注入 `_CONFIG_REGISTRY`（见 §4.1） |

### 2.2 插件化落地（不改 vllm core）

```
vllm_ascend/models/step4/
├── __init__.py    # 懒加载入口
├── config.py      # Step4Config / Step4MTPConfig（逐字移植）+ registry 注入
├── envs.py        # VLLM_STEP4_* 的 typed 门面（实现在 vllm_ascend/envs.py）
├── kernels.py     # sparse 配置解析 + torch 算子（见 §3.2）
├── layernorm.py   # OptimusRMSNorm / OptimusLayerNorm 模块
├── model.py       # Step4ForCausalLM / Step4Model / Step4DecoderLayer /
│                  # Step4Attention / FusedMoEBlock / Step4MLP + 权重加载
└── mtp.py         # Step4MTP draft 模型
tests/ut/models/step4/test_step4_ops.py   # CPU 可运行单测
```

注册链路：`setup.py` entry-point `ascend_model = vllm_ascend:register_model` →
vllm `load_general_plugins()`（`EngineArgs.__post_init__`，早于 ModelConfig 构建）→
`register_step4_configs()` 注入 config 类 + `ModelRegistry.register_model("Step4ForCausalLM"/"Step4MTP")`。

### 2.3 分层适配策略

1. **配置层**：`Step4Config` 逐字移植（含 `layer_types_with_mtp` 与 transformers 5.x 校验的兼容处理），运行时注入而非改 core。
2. **模型层**：dense 路径完整移植（DecoderLayer/Attention/MLP/MoE/权重加载/8 线程 loader 原样保留）。
3. **算子层**：按下表分类处理——框架已有直接复用、缺失的用 torch 实现、CUDA 专有的明确拒绝。
4. **DSA 稀疏路径**：P0 明确不支持（SM90 CuTeDSL + sidecar KV cache 契约均无法在 NPU 落地），sparse checkpoint 启用时 fail-fast 报错并指引 `VLLM_STEP4_SPARSE=0` dense fallback。

---

## 3. 算子支持矩阵

### 3.1 框架层面支持（vllm 原生算子/层，vllm-ascend 已适配，直接复用）

| 算子 / 层 | vllm 侧定义 | vllm-ascend 对应实现 | Step4 中的用途 |
|---|---|---|---|
| `Attention` | `vllm.model_executor.layers.attention.Attention` | NPU attention backend（FA/acl 等，经插件选择） | 全部注意力层（dense 路径）；`per_layer_sliding_window` 承载 SWA 层；KV cache 写入/读取走标准 backend 路径 |
| `SiluAndMul` | `vllm...layers.activation` | `vllm_ascend/ops/activation.py` | dense MLP / 共享专家激活 |
| `SwigluStepAndMul`（非对称 clamp SwiGLU） | 同上 | 同上（已含 swiglustep_and_mul） | 带 `swiglu_limits` 的 MLP/MoE 层（gate 只 clamp 上界、up 双向 clamp，是原生语义，无需额外实现） |
| `fused_add_rms_norm`（标准 RMSNorm + residual） | `vllm._custom_ops` | NPU 自定义算子 | `zero_centered=False` 的 norm 路径 |
| `FusedMoE`（`FusedMoEFactory`） | `vllm...layers.fused_moe` | `vllm_ascend/ops/fused_moe/` + `patch/platform/patch_fused_moe.py` | 352 路由专家（EP 切分）、`custom_routing_function`、`swiglustep` 激活、EPLB 元数据 |
| 线性层（`QKVParallelLinear` / `MergedColumnParallelLinear` / `RowParallelLinear` / `ColumnParallelLinear` / `ReplicatedLinear`） | `vllm...layers.linear` | torch_npu matmul + HCCL | qkv/gate_up/down/o_proj/g_proj；TP 切分逻辑复用 |
| `VocabParallelEmbedding` / `ParallelLMHead` | `vllm...layers.vocab_parallel_embedding` | `vllm_ascend/ops/vocab_parallel_embedding.py` | embed / lm_head |
| `get_rope`（RotaryEmbedding cos_sin_cache 构建） | `vllm...layers.rotary_embedding` | 可用（表构建为纯 torch） | 慢路径 RoPE 与融合算子的 cos/sin 表 |
| `LogitsProcessor`（含 `org_vocab_size` padded-vocab 掩码） | `vllm...layers.logits_processor` | 可用 | padded 词表 logits 截断（替代 patch 的 valid_vocab_size） |
| TP/DP/EP 通信原语 | `vllm.distributed` | HCCL | all_reduce / all_gather / reduce-scatter |
| 权重加载工具（`AutoWeightsLoader` / `fused_moe_make_expert_params_mapping` / `maybe_remap_kv_scale_name` / `get_spec_layer_idx_from_weight_name`） | `vllm.model_executor.models.utils` 等 | 可用 | 权重名映射 / MoE 专家映射 / KV scale 重映射 |
| `support_torch_compile` / torch.compile | `vllm.compilation.decorators` | `vllm_ascend/compilation/`（getrock） | 图编译模式 |
| `direct_register_custom_op` | `vllm.utils.torch_utils` | 可用 | 本适配所有 torch 算子的 custom op 注册（编译图透明） |

### 3.2 Step4 专有算子 —— torch 实现（本次适配新增）

数值语义全部对齐独立推理版 `step4-inference/inference/kernel.py` 与 CUDA 版的原生 fallback。

| 算子 | 本实现（`step4/kernels.py` / `layernorm.py`） | vllm（CUDA step4 patch）上的对应实现 | 类型 |
|---|---|---|---|
| **zero-centered RMSNorm**（`optimus_rms_norm`，`x·rsqrt(mean x²+ε)·(w+1)`，fp32 计算） | `OptimusRMSNorm` + `torch.ops.vllm.optimus_rms_norm`（纯 torch fp32） | `torch.ops.Optimus.RMSNorm_forward`（闭源 step-optimus 扩展）→ 降级 `torch.ops._C.optimus_fused_add_rms_norm`（csrc CUDA kernel）→ 再降级同样的纯 torch fallback | custom op（torch） |
| **fused add RMSNorm**（residual + zero-centered norm，fp16 残差升 fp32） | `torch.ops.vllm.optimus_fused_add_rms_norm`（纯 torch） | 同上 Optimus/`_C` 链 | custom op（torch） |
| **per-head LayerNorm**（indexer k 归一化） | `OptimusLayerNorm`（`F.layer_norm` + unflatten） | `OptimusLayerNorm`（相同实现，平台无关） | 纯 torch 模块 |
| **fused QKNorm + NeoX 部分 RoPE**（主注意力 q/k：per-head RMSNorm → 落回 dtype → fp32 旋转前 `2·rotary_dim` 维 → 落回 dtype） | `torch.ops.vllm.fused_qknorm_rope_forward_impl`（纯 torch，`_rms_norm_per_head` + `_apply_neox_partial_rope`） | CuTeDSL `FusedQKNormRope` kernel（CUTLASS Python DSL，SM90 cp.async/wgmma）；含 cache 变体 `_C.optimus_fused_qknorm_rope_cache`（一次完成 norm+rope+KV 写入） | custom op（torch） |
| **fused indexer norm + RoPE**（RMSNorm(q)+LayerNorm(k)+部分 RoPE，z 透传） | `torch.ops.vllm.fused_indexer_norm_rope_forward_impl`（纯 torch） | CuTeDSL `indexer_norm_rope.py` kernel | custom op（torch） |
| **MoE 路由**（sigmoid 门控 + 选择 bias 的 top-k；权重减回 bias 后 renorm(ε=1e-20) × routed_scaling；NaN 行置零/`nan_row_i_out`；int32/int64 indices） | `router_bias_func`（torch：`sigmoid+bias → topk → gather 减 bias → renorm → scaling`） | Triton kernel `router_bias_topk_kernel`（`nvidia/ops/triton/router_bias.py`，CUDA driver launch，支持 DeepEP int64） | custom routing function（torch） |
| **FP32 门控 matmul**（`need_fp32_gate`） | `FP32ReplicatedLinear`：`F.linear(x.float(), w.float())` | `torch.ops.vllm.optimus_matmul_fp32`（闭源 OptimusMoe 扩展）/ batch-invariant 路径 | 线性层（torch fp32） |
| **head-wise attention gate**（`g_proj` → sigmoid 逐头缩放 attn_output） | 纯 torch 广播乘（`step4_materialize_gate_input` 保证编译下 buffer 稳定） | 相同（平台无关） | 纯 torch |
| **fp32 residual / norm dtype 切换** | `_cast_for_residual` / `_cast_for_param_op` 纯 torch | 相同（平台无关） | 纯 torch |

> KV cache 写入：CUDA 版的 `fused_qknorm_rope_cache` 将 norm+RoPE+KV 写入融合为单 kernel 并向编译器暴露副作用依赖（`kv_cache_dummy_dep`）。本适配**不做该融合**——norm+RoPE 走上述 torch 算子，KV 写入交还 `Attention.forward` 标准路径（vllm-ascend 已适配），功能等价、少一次融合。

### 3.3 原版（CUDA patch）中存在但 vllm-ascend 场景下不适用 / 已有等价物的算子

| CUDA patch 算子 | 说明 | Ascend 处理 |
|---|---|---|
| `torch.ops._C.optimus_fused_qknorm_rope_cache[_bitwise]`（csrc/libtorch_stable/step4/*.cu） | QKNorm+RoPE+KV cache 写入融合 CUDA kernel | 不移植；KV 写入走标准 backend |
| `torch.ops.Optimus.RMSNorm_forward` / `torch.ops.OptimusMoe.matmul_fp32`（闭源 step-optimus wheel） | 闭源扩展，CUDA only | 原版自带纯 torch fallback，本适配直接以 fallback 为主实现 |
| `Step4SplitKVFlashAttentionBackend` / `Step4DSAAttentionBackend` | DSA 专用 attention backend（flash-attn 改造 + sidecar 契约，core patch） | 不移植（DSA 未支持） |
| Optimus JIT vendor 机制（`OPTIMUS_JIT_VENDOR_MANIFEST.tsv` 等） | 闭源内核 JIT 分发 | 不适用 |
| `linear_fp32_batch_invariant`（batch 不变模式） | patch 对 batch_invariant 的扩展 | 省略（非功能性），FP32 gate 走 fp32 linear |

### 3.4 不支持（本次范围外，明确拒绝）

| 能力 | CUDA 侧实现 | 状态 |
|---|---|---|
| **DSA 稀疏注意力全套**：indexer topk 选择（`topk_selector_sm90_gqa` 等）、sparse GQA decode/prefill/union/splitkv、CSA summary cache + sidecar 状态（约 3 万行 SM90 CuTeDSL） | `vllm/models/step4/nvidia/ops/cute_dsl/` + `sparse_attention.py`（4195 行）+ `sparse_summary_cache.py`（3072 行） | **不支持**。sparse config 生效时 `Step4DecoderLayer.__init__` 直接 `NotImplementedError`；`VLLM_STEP4_SPARSE=0` 可跑 dense fallback（全层 full attention，indexer 权重静默跳过——注意数值与部署参考不同） |
| MTP spec-decode proposer 引擎接线（reprefill kernel、DP 协调等） | core patch `vllm/v1/spec_decode/step3p5.py` step4 路由 | `Step4MTP` 模型类已移植注册（可作为独立 draft checkpoint 加载），引擎侧 proposer 接线未做 |
| `valid_vocab_size` tokenizer 上界自动解析链 | core patch（`ModelConfig.resolve_valid_vocab_size` 等，跨 tokenizer/scheduler/spec-decode 的契约） | 以 `org_vocab_size` 掩码近似；显式指定需 `hf_overrides` 传 vocab 相关配置 |

### 3.5 后续优化候选（vllm-ascend 已有设施替换 torch 实现）

| torch 实现 | 可替换为 | 收益 |
|---|---|---|
| `_optimus_rms_norm_native` | `torch_npu` 融合 RMSNorm 算子 / `vllm_ascend/ops/layernorm.py`（`AscendRMSNorm`）思路扩展 zero-centered | 单 kernel 融合，减少 H2D 算子拆分 |
| `router_bias_func`（torch） | triton-ascend kernel（参照 `vllm_ascend/ops/triton/` 生态；CUDA 原版即 Triton） | topk 循环融合、少中间 tensor |
| `fused_qknorm_rope_forward_impl` | triton-ascend / ACLNN 组合（`npu_rms_norm` + rotary 融合） | norm+rope 单次 launch |
| （远期）DSA | 参考 `vllm_ascend/ops/dsa.py`（DeepSeek MLA 系稀疏）与 minimax-m3 lightning indexer 在 NPU 上的先例；独立版 kernel.py 已含 Triton 版 DSA 语义与 `_csa_compress_regions_kernel_ascend`（AIV 适配）雏形 | 稀疏注意力加速 |

---

## 4. 关键设计决策

### 4.1 config 注册（不改 core 的注入方式）

`register_step4_configs()` 在插件加载时执行：

```python
import vllm.transformers_utils.configs as vllm_configs
from vllm.transformers_utils.config import _CONFIG_REGISTRY

vllm_configs.Step4Config = Step4Config          # LazyConfigDict 按名字符串解析
vllm_configs.Step4MTPConfig = Step4MTPConfig
_CONFIG_REGISTRY.setdefault("step4", "Step4Config")
_CONFIG_REGISTRY.setdefault("step4_mtp", "Step4MTPConfig")
```

`_CONFIG_REGISTRY` 是 `LazyConfigDict`，`__getitem__` 经 `getattr(configs, value)` 惰性解析注入的类；vllm 的 `AutoConfig` 随后通过 `_register_config_class(model_type, cls)` 完成注册。因插件加载（`EngineArgs.__post_init__` → `load_general_plugins`）早于 `ModelConfig` 构建，step4/step4_mtp 的 config.json 可被正常解析。

### 4.2 DSA 拒绝与 dense fallback 语义

- 判定链：checkpoint 含 sparse section（`step4_sparse_config`/`step3p5_sparse_config`/`sparse_config`）且未设 `VLLM_STEP4_SPARSE=0` → `Step4DecoderLayer.__init__` 抛 `NotImplementedError`，错误信息指引 fallback 开关。
- dense fallback 下：稀疏层退化为全量 full attention；`qkv_indexer_proj` / `sparse_indexer_*` / `ssmax_s` 等 indexer 权重不构造，checkpoint 中对应张量在 `load_weights` 中因 `params_dict` 无匹配而静默跳过。
- 权重加载方面原样保留 mxfp4 / groupwise / per-expert 三种量化映射分支；实际可用性取决于 vllm-ascend 对应量化后端。

### 4.3 数值对齐要点（torch 实现必须复刻的语义）

1. **RoPE**：NeoX 配对（`(d, d+rotary_span/2)` 为一对）而非 interleave；部分旋转只转前 `2·rotary_dim` 维；norm 输出**先落回输入 dtype 再进 fp32 旋转**（独立版 kernel 明确依赖此舍入路径，float64 计表会改变 bf16 结果）。
2. **router**：选择按 `sigmoid(logits)+bias` 排序，权重为该分数**减回 bias**（算术上等于 `sigmoid(logits)` 但非位等价）；renorm 分母加 `1e-20`；`routed_scaling_factor` 在 renorm 之后乘。
3. **zero-centered RMSNorm**：`w+1` 仿射；fp32 计算后落回原 dtype。
4. **indexer**：q 是 RMSNorm（zero-centered 可配）、k 是 **LayerNorm（带 bias）**——两者只差一次均值消减，是易错点；z 纯透传。
5. **fp32 residual**：`fp32_residual_connection=True` 时残差流 fp32、GEMM 输入 bf16（`_cast_for_param_op`/`_cast_for_residual`）。

### 4.4 环境变量

按 vllm-ascend AGENTS.md 集中注册于 `vllm_ascend/envs.py`，保留 CUDA patch 原名（`VLLM_STEP4_SPARSE`、`VLLM_STEP4_ENABLE_QKVG_PROJ`、`VLLM_STEP_CC_LEVEL` 等）以统一运维口径；`step4/envs.py` 仅作 typed 门面。`VLLM_STEP4_SPARSE=0` 是 DSA checkpoint 跑 dense fallback 的唯一开关。

---

## 5. 验证

| 层级 | 内容 | 状态 |
|---|---|---|
| 静态 | 全部文件 py_compile；包内交叉引用；56 个 vllm API 在 v0.27.1 tag 的存在性逐一核对 | ✅ 本地通过 |
| 算法 | numpy 镜像对拍：NeoX RoPE（hd 64/128/192 × partial 0.5/1.0）、router 逐槽贪心参考（含 NaN 行）、zero-centered RMSNorm、indexer LayerNorm | ✅ 全部一致 |
| 单测 | `tests/ut/models/step4/test_step4_ops.py`：算子数值对拍参考实现、sparse config 解析（含 env 强制关闭、非法参数拒绝） | ✅ 已编写；**需 NPU/pytest 环境执行** |
| e2e | dense checkpoint 冒烟（`vllm serve` + greedy 对拍独立版 `generate.py` 输出） | ⬜ 待 NPU 环境 |

---

## 6. 已知限制与路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P0（本次）** | dense 推理路径：模型/config/权重加载完整移植，缺失算子 torch 化，DSA fail-fast | ✅ |
| **P1** | 性能：`torch_npu` 融合 norm、triton-ascend router/qknorm_rope、ACLGraph 下算子替换 | ⬜ |
| **P2** | DSA 稀疏注意力移植（indexer → topk → sparse GQA → summary cache），需同步设计 NPU 侧 KV sidecar 契约 | ⬜ |
| **P2** | MTP spec-decode proposer 引擎接线（draft 模型类已就绪） | ⬜ |
| **P2** | `valid_vocab_size` 完整解析链（tokenizer 上界推导 + spec-decode/结构化输出联动） | ⬜ |

**P0 dense 模式的预期差异**：稀疏层全量注意力使长上下文 prefill 显著变慢、且注意力数值与部署参考（稀疏选择）不同——适用于功能验证与 dense 变体模型，不适用于 DSA checkpoint 的精度评测。

---

## 7. DSA 稀疏注意力 Ascend 原生适配（第二阶段）

### 7.1 总体方案

DSA 不再拒绝。参照独立推理版 `step4-inference/inference/kernel.py` 的数值语义，用 **torch 实现**完整稀疏路径，并按 vllm-ascend 的模型专属注意力模式（MiniMax-M3 sparse 同款）接入 v1 引擎：

```
vllm_ascend/models/step4/dsa.py
├── torch DSA 算子
│   ├── csa_compress_regions      # region 压缩：softmax(z)·k，标量 shift，fp32→bf16→e4m3
│   ├── indexer_logits            # 加权 ReLU 打分：Σ_h relu(e4m3(q)·e4m3(s))·w，fp32
│   ├── _select_topk_regions      # torch.topk，候选域 [0, pos//region_size)，升序输出
│   ├── sparse_attention          # 选中 region 的 KV gather + 掩码 softmax（fp32）
│   └── _gather_region_summaries  # block_table 逻辑→物理 region 映射
├── Step4DSACore(nn.Module, AttentionLayerBase)   # indexer 模块 + sidecar + 选择 + 稀疏注意力
├── Step4DSABackend / Step4DSAMetadataBuilder / Step4DSAAttentionImpl
└── Step4IndexerLinear / Step4ReplicatedLinear    # checkpoint 权重名直接映射，provider-group 切分
```

集成方式：`Step4Attention` 对 full-attention 层构造 `Step4DSACore`（主 q/k/v 投影与 QKNorm+RoPE 留在父层，权重名与 checkpoint 一致）；DSA core 注册进 `static_forward_context`，由 runner 经 `AttentionLayerBase` 发现，KV cache 用标准 split 布局 `(2, num_blocks, block_size, kv_heads, head_dim)`，与 SWA 层同 spec。

### 7.2 关键设计

1. **summary sidecar 按物理 region 索引**（`block_id × regions_per_block + offset`，e4m3 存储）：summary 是 region token 的纯函数，天然正确支持 prefix-cache 块共享、请求重排、chunked prefill；无需请求级持久状态。
2. **尾部 region 用 per-physical-block pending 缓存**（活跃数 ≤ 并发请求数）：region 补满第 8 个 token 即压缩写入，之后释放。
3. **每个 step 的完成 region 集由调度元数据推导**（`[past//8, total//8)`），无累积计数器，天然幂等。
4. **选择语义与部署一致**：候选域为严格过去的完整 region，查询自身所在 region 无条件并入（valid tokens = `pos%8+1`）；e4m3 激活舍入参与选择（是语义不是优化）。
5. **DSA forward 经 `torch.compiler.disable` 豁免 torch.compile**：sidecar 更新与逐请求选择循环为 host 驱动，P0 以 eager 正确性优先（也就不在 ACL graph 捕获范围内）。性能优化（triton-ascend 融合内核、去 CPU 同步、批量打分）为 P1。

### 7.3 已知限制

| 项 | 说明 |
|---|---|
| eager 执行 | DSA 层不参与 torch.compile / ACL graph；长上下文性能显著弱于 CUDA 版，正确性优先 |
| topk 平局 | torch.topk 与独立版 radix 选择在精确零分平局上可能有罕见行差异 |
| 每步 CPU 同步 | sidecar 更新/选择需 3 次 metadata CPU 拷贝/层/步，P1 优化 |
| TP 拓扑 | 要求每 rank 恰好 1 个本地 KV/provider group（TP ≥ provider groups，覆盖 TP4/TP8） |
| MTP | draft 层的 DSA 同样可用；spec-decode 引擎接线仍属 P2 |
| 建议 | 首次运行建议 `--enforce-eager`；观察 sidecar 显存日志（约 32B/token/层） |
