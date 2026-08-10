# Model family: DeepSeek-V4 / Kimi-K3 (the 2026 trillion-scale MoE wave)

> ## ⛔ PHASE 0 VERDICT (2026-08-09) — THE ALIAS HYPOTHESIS IS REFUTED FOR BOTH FAMILIES
>
> Phase 0 was a gate, and **it caught what it was built to catch.** Every table below has been
> corrected against the real `config.json` + modeling files (fetched from HF; no weight downloads).
> The original doc's assumption — "V4/K3 are V3-shaped, so each is a near-free alias" — **does not
> survive contact with the configs.**
>
> | model | verdict | one-line reason |
> |---|---|---|
> | **DeepSeek-V4** (`-Pro`, `-Flash`) | **ALIAS REFUTED** | `kv_lora_rank` is **absent**; attention is `sparse_attn(...)` over a learned `Indexer` top-k with a sliding window + KV `Compressor`. Router has **no `n_group`/`topk_group`** and a third `scoring_func`. |
> | **Kimi-K3** | **PARTIAL** | MLA is genuinely ours (`kv_a_proj_with_mqa`/`kv_b_proj`, `kv_lora_rank: 512`) and the router is genuinely DeepSeekMoE — but MLA runs on only **24 of 93 layers**; the other **69 are `KimiDeltaAttention`** (gated linear attention), plus a latent-MoE wrapper and a new activation. |
>
> **DeepSeek-V4 — scope line rewritten: PRIMITIVE ADD, RE-ESTIMATE.** The primitives are (1) **DSA
> sparse attention** — a learned `Indexer` selecting `index_topk` positions, evaluated by
> `sparse_attn` over gathered indices; (2) **strided KV compression** (`Compressor`, per-layer
> `compress_ratios`); (3) **sliding-window rolling KV cache** with an **attention sink**; (4)
> **grouped low-rank output projection** (`o_lora_rank`/`o_groups`); (5) **hash routing** on the first
> `num_hash_layers` layers (expert ids looked up per *token id*); (6) **`sqrtsoftplus` scoring**; (7)
> **hyper-connections** (`hc_*`, Sinkhorn) in every block; (8) **clamped SwiGLU** (`swiglu_limit`).
> This is not an adapter on `deepseekArchitecture` — it is a new architecture that happens to share
> q-LoRA and a shared expert. Estimate accordingly; do not carry the "near-free" figure forward.
>
> **Kimi-K3 — scope line rewritten: PRIMITIVE ADD (hybrid), RE-ESTIMATE.** The MLA half is close to
> free. The other half is a **KDA (Kimi Delta Attention) layer type** on 69/93 layers. goinfer *has* a
> DeltaNet family (qwen3_5_moe), which makes this the cheapest of the two — but KDA is its own variant
> (short conv on q/k/v, low-rank forget gate + `dt_bias`, full-rank output gate) and the interleave is
> a hybrid-scheduling change, not a config value. Plus **latent-MoE** (`routed_expert_{up,down}_proj`),
> **`situ` activation**, **MLA-with-NoPE + output gate**, and **per-sublayer residual projections**.
>
> ### What the two "genuinely new work" items actually are
>
> **1. Router-capacity cap — ✅ DONE (2026-08-09), raised to 512.** "The cap" was never one number:
> it is **three shader constants plus one Go map**, and only one of them sat inside a frozen artifact.
> Itemized, with what changed:
>
> | where | was | now | why |
> |---|---|---|---|
> | `cuda/moe.cu` `MOE_MAX_E` | 256 | **512** | bounds `moe_route`'s per-thread `score[]`/`sel[]`. **Inside the audited 12.6.85 `moe.ptx`** — required regenerating it at the *pinned same* toolchain, with a byte-identical rebuild-unchanged control first (`cuda/testdata/REGEN.md`) |
> | `gpu/moe.go` `MAXE` / `array<f32,N>` | 256 | **512** | WGSL, compiled at runtime — no frozen artifact. **This is the one that actually unblocks K2** |
> | `metal/moe.go` `float score[256]` | 256 | **256 (unchanged)** | runtime-compiled MSL, but raising it is Mac-validation work — deferred |
> | `decoder/features.go` `residentBackendMoECap` | webgpu/cuda only | webgpu **512**, cuda **512**, **metal `{256,64}` newly DECLARED** | the map claimed "absent entry = no fixed-size router cap", which was **false for Metal** — its shader really is 256 and rejects above it |
>
> **512 not 896, deliberately:** it covers Kimi-K2's 384 (the shipped family that was declining) and
> DeepSeek-V4-Pro's 384, and stops short of K3's 896 — an unbuilt family should not set a validated
> limit. **V4-Flash's 256 already passed** (comparison is `>`).
>
> ⚠ **The headline claim in this doc needed correcting: raising the CUDA cap does NOT unblock K2.**
> `FeatMLA` is declared by **webgpu only** (`cuda: false`, `metal: false`), so on CUDA/Metal K2
> declines on *features*, not capacity — the cap was never its blocker there. WebGPU is the one
> backend where the cap was the sole obstacle, and after the bump **Kimi K2 flips `CPU → ✅ resident`
> in the generated hardware matrix**. The CUDA raise is still worth having (it admits non-MLA MoEs at
> 257–512 experts and removes one of two blockers ahead of any MLA-on-CUDA work), but it is not what
> delivered K2.
>
> **2. `streamExperts` — THE ASSUMPTION DOES NOT MATCH THESE CHECKPOINTS.** The doc claimed the
> gemma4 work "generalized `streamExperts` to *any* fused-stacked-expert MoE". That statement is true
> and **the antecedent is false**: `streamExperts` (`decoder/weights.go:1032`) takes **one fused
> `[nExpert, rows, cols]` tensor** and hard-validates `t.Elements() == nExpert*stride`. Both families
> ship **one tensor per expert** — checked in the safetensors index without downloading weights:
>
> - V4-Flash: `layers.N.ffn.experts.N.w{1,2,3}.weight` + `.scale` — 11 008 tensors = 43 layers × 256 experts.
> - Kimi-K3: `language_model.model.layers.N.block_sparse_moe.experts.N.w{1,2,3}.weight_packed` + `.weight_scale` — 82 432 = 92 MoE layers × 896 experts.
>
> So this is **not** "wire the loader onto that path" — it needs a per-expert-tensor entry point.
> That is arguably *easier* (no 3-D transient exists to avoid), but it is different code and the
> estimate must say so. **And a bigger blocker sits in front of it:** both checkpoints are
> **pre-quantized in formats we do not fully read** — V4 is `fp8` `e4m3` blockwise
> (`weight_block_size: [128,128]`, `scale_fmt: "ue8m0"`) with **no fp8 support in the tree**; K3 is
> `mxfp4-pack-quantized` (`weight_packed`/`weight_scale`), where `decoder/mxfp4.go` exists for gpt-oss
> but the compressed-tensors packing must be confirmed to match. **Reading the weights at all is a
> prerequisite to streaming them**, and that was not in the original estimate.
>
> ### Repo-naming delta (found while fetching)
> There is **no `deepseek-ai/DeepSeek-V4` repo.** The family is **`DeepSeek-V4-Pro`** (0.86 TB) and
> **`DeepSeek-V4-Flash`** (0.16 TB), plus `-Base`, `-0731` and `-DSpark` variants. `moonshotai/Kimi-K3`
> is 1.56 TB. The doc's "1.6T total / 49B active" and "2.8T / 16-of-896" figures are **not stated in
> the configs**; only K3's 16-of-896 is confirmed (`num_experts_per_token: 16`, `num_experts: 896`).
> Unverified parameter counts have been removed rather than repeated.

---

## Verified configuration — DeepSeek-V4 (`model_type: "deepseek_v4"`)

Source: `deepseek-ai/DeepSeek-V4-Pro/config.json`, `.../DeepSeek-V4-Flash/config.json`,
`inference/model.py` (both repos ship the same reference implementation).
`architectures: ["DeepseekV4ForCausalLM"]`.

### Attention — **NOT MLA as we implement it** ⚠ THE MAKE-OR-BREAK DELTA

| trait | doc expected | **CONFIRMED reality** | verdict |
|---|---|---|---|
| latent KV cache | `kv_lora_rank` present | **`kv_lora_rank` ABSENT.** `self.wkv = Linear(self.dim, self.head_dim)` — one 512-d vector per position, `num_key_value_heads: 1`. No `kv_a`/`kv_b` LoRA pair, no per-head expansion | ❌ **refuted** |
| attention op | dense softmax over full KV | **`o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)`** | ❌ **DSA line landed** |
| index/selector | — | **`class Indexer`**; `index_n_heads: 64`, `index_head_dim: 128`, `index_topk: 1024` (Pro) / `512` (Flash) | ❌ new primitive |
| KV compression | — | **`class Compressor`**; `compress_ratios` per-layer array (values `128`/`4`/`0`), `compress_rope_theta: 160000`. Indexer only attached where `compress_ratio == 4` | ❌ new primitive |
| window | — | **`sliding_window: 128`**; rolling cache sized `window_size + max_seq_len // compress_ratio` | ❌ new |
| attention sink | — | **`self.attn_sink = nn.Parameter(...)`** per head | ❌ new |
| output projection | plain `o_proj` | **grouped low-rank**: `o_lora_rank: 1024`, `o_groups: 16` (Pro) / `8` (Flash); `einsum("bsgd,grd->bsgr")` then `wo_b` | ❌ new |
| output RoPE | — | **inverse RoPE on the attention output**: `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` | ❌ new |
| q low-rank | `q_lora_rank` present | ✅ `q_lora_rank: 1536` (Pro) / `1024` (Flash), `wq_a → q_norm → wq_b` | ✅ **only shared trait** |

### Routing — **NOT group-limited DeepSeekMoE**

| trait | doc expected | **CONFIRMED reality** | verdict |
|---|---|---|---|
| `n_group` / `topk_group` | present | **BOTH ABSENT from config.** `Gate.forward` does plain global `indices = scores.topk(self.topk, dim=-1)[1]` | ❌ **refuted** |
| `scoring_func` | sigmoid or softmax | **`"sqrtsoftplus"`** → `scores = F.softplus(scores).sqrt()` — a **third** value the loader has never seen | ❌ new |
| correction bias | `e_score_correction_bias` | ✅ equivalent: `self.bias` added for selection only, not to weights (non-hash layers) | ✅ |
| `topk_method` | `noaux_tc` | ✅ `"noaux_tc"` in config — **but the grouped path it names is not implemented** | ⚠ misleading key |
| hash routing | — | **first `num_hash_layers: 3` layers use `self.tid2eid[input_ids]`** — expert ids per **token id**, requiring `input_ids` plumbed into the MoE | ❌ new primitive |
| shared expert | ungated, count from config | ✅ `n_shared_experts: 1`, `assert args.n_shared_experts == 1`, ungated add | ✅ |
| `first_k_dense_replace` | dense prefix | **ABSENT — every layer is MoE** (`self.ffn = MoE(layer_id, args)` unconditionally) | ⚠ delta |
| expert FFN | SwiGLU | ⚠ **clamped** SwiGLU: `swiglu_limit: 10.0` clamps `up` to ±10 and `gate` to ≤10 | ❌ new |
| block residual | plain residual | **hyper-connections**: `hc_mult: 4`, `hc_sinkhorn_iters: 20`, `hc_eps: 1e-06`; params `hc_ffn_fn`/`hc_ffn_base`/`hc_ffn_scale`, applied via `hc_pre` | ❌ new |

### Numbers

| | V4-Pro | V4-Flash |
|---|---|---|
| `num_hidden_layers` | 61 | 43 |
| `hidden_size` | 7168 | 4096 |
| `n_routed_experts` / `num_experts_per_tok` | **384** / 6 | **256** / 6 |
| `n_shared_experts` | 1 | 1 |
| `moe_intermediate_size` | 3072 | 2048 |
| `num_attention_heads` / `head_dim` / `num_key_value_heads` | 128 / 512 / 1 | 64 / 512 / 1 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | 1536 / 1024 / 16 | 1024 / 1024 / 8 |
| `index_topk` | 1024 | 512 |
| `max_position_embeddings` | 1 048 576 | 1 048 576 |
| RoPE | `rope_theta: 10000`, YaRN `factor: 16`, `original_max_position_embeddings: 65536`, `beta_fast: 32`, `beta_slow: 1`; separate `compress_rope_theta: 160000` | same |
| `vocab_size` / `tie_word_embeddings` | 129 280 / **false** | 129 280 / **false** |
| MTP | `num_nextn_predict_layers: 1` | 1 |
| checkpoint dtype | `expert_dtype: "fp4"`; `quant_method: "fp8"`, `fmt: "e4m3"`, `weight_block_size: [128,128]`, `scale_fmt: "ue8m0"` | same |
| repo size | 0.86 TB | 0.16 TB |

**`kv_lora_rank`: does not exist in either config.** (The doc asked for this number; the honest answer
is that the field the question presupposes is absent.)

---

## PHASE 0b (2026-08-09) — K3's two cost questions, priced

Read-only follow-up to Phase 0: modeling files, configs, the safetensors index, and goinfer's own
source. No weights, no code.

| question | verdict |
|---|---|
| **KDA vs shipped DeltaNet** | **SMALL VARIANT KERNEL** — same delta rule, per-channel decay instead of per-head scalar. *One open question named below.* |
| **MXFP4 packing** | **LAYOUT SHIM** — identical numeric format, different container (separate planes vs interleaved blocks). Confined to routed experts only. |
| **RoPE-less MLA** | **NEEDS A BRANCH** — the shipped MLA forward ropes unconditionally and `mlaParams` has no NoPE flag. |

### 1. KDA vs the shipped Gated DeltaNet — op by op

Ours: `decoder/deltanet.go` (qwen3_5_moe). Theirs: `KimiDeltaAttention`, `modeling_kimi_linear.py:477`.

| op | goinfer DeltaNet | K3 KDA | delta |
|---|---|---|---|
| q/k/v projection | **fused** `inProjQKV → [q;k;v]` | **separate** `q_proj`/`k_proj`/`v_proj` | ⚪ load-time concat |
| conv front-end | depthwise causal conv `convW[convDim,K]` + **SiLU** | `q_conv1d`/`k_conv1d`/`v_conv1d` = `ShortConvolution(kernel_size=4, activation='silu')` | ⚪ **same**, `short_conv_kernel_size: 4` |
| q/k L2 norm | `l2normScaled(...)` on q and k | `use_qk_l2norm_in_kernel=True` | ⚪ **same** |
| write gate β | `beta = sigmoid(b_t[headV])` — **per head** | `beta = b_proj(hidden)` → `[num_heads]`, `use_beta_sigmoid_in_kernel=True` | ⚪ **same** |
| **decay gate g** | `g = negExpA[headV] · softplus(a_t[headV] + dtBias[headV])` — **PER-HEAD SCALAR**; state decays uniformly `S[i] *= gt` | `g = f_b_proj(f_a_proj(h))` → `rearrange('... (h d) -> ... h d')` — **PER-CHANNEL**, with `dt_bias` of width `num_heads·head_dim` and `A_log` per head | 🔴 **THE DELTA** |
| gate floor | none | `safe_gate` + `gate_lower_bound: -5.0` | 🔴 new clamp |
| state update | `kv = Σ S[kd·hv+vd]·k[kd]`; `delta = (v[vd] − kv)·β`; `S += k⊗delta` | `chunk_kda` / `fused_recurrent_kda` (from **`fla`**) | ⚪ same delta rule |
| output norm | gated RMSNorm (`normW`) | `FusedRMSNormGated(head_dim, activation='sigmoid')` | ⚪ **same** |
| output gate | `inProjZ → z` | `g_proj` (`use_full_rank_gate: True`; a low-rank `g_a`/`g_b` variant also exists) | ⚪ same role |

**Why "small variant kernel" and not "new scan primitive".** The recurrence is the *same gated delta
rule* — L2-normed q/k, sigmoid β, `S ← decay·S + k⊗(v − S·k)β`. Every surrounding op (conv+SiLU,
L2 norm, gated RMSNorm, output gate, A_log/dt_bias/softplus decay) is already implemented and was
validated op-for-op for qwen3_5_moe. **The one structural change is that the decay stops being a
scalar per head and becomes a vector along one axis of the state matrix** — our `S[i] *= gt` becomes
a per-column multiply. That is a broadcast-shape change to an existing inner loop plus a clamp, not a
different fold.

> 🚩 **OPEN QUESTION, and it must be answered before anyone writes the kernel.** `g` has shape
> `(h, head_dim)` and KDA sets `head_k_dim == head_dim == head_v_dim`, so **the config alone does not
> say which axis of the `[head_k_dim, head_v_dim]` state the decay indexes** — key-dim or value-dim.
> The scan itself lives in **`fla` (flash-linear-attention), an external Triton library the modeling
> file only calls**; it is not in the checkpoint repo. Resolving this needs `fla`'s source, not more
> config reading. Getting the axis wrong is a silent-wrong class (plausible output, incorrect decay),
> so it is a hard prerequisite, not an implementation detail.
>
> **Bit-identity note:** the reference fold is `chunk_kda` — a *chunkwise* scan, whose accumulation
> order differs from a strict sequential recurrence. goinfer's DeltaNet decodes one token at a time
> (sequential by construction), so decode parity is against the sequential order and is unaffected;
> any future chunked *prefill* would inherit the same reassociation question the SSM work already met.

### 2. MXFP4 — the gpt-oss unpacker's format, a different container

`quantization_config` (text_config): `format: "mxfp4-pack-quantized"`, `quant_method:
"compressed-tensors"`, `group_size: 32`, `num_bits: 4`, `type: "float"`, `symmetric: true`,
`scale_dtype: "torch.uint8"`, `strategy: "group"`.

Ours (`decoder/mxfp4.go`): *"A block is 32 elements: one e8m0 8-bit power-of-two scale byte, then 16
bytes each packing two e2m1 4-bit values (4.25 bits/weight). 17 bytes/block."*

**The numbers match exactly** — 32-element groups, 4-bit e2m1 values, an 8-bit (e8m0/uint8)
power-of-two scale per group, symmetric. **The container does not:** ours reads one *interleaved*
17-byte-per-block stream (GGML type 39); K3 ships **two separate planes**, `…experts.N.w{1,2,3}.weight_packed`
and `…weight_scale`. So the value table and scale semantics are reusable verbatim; what is new is a
reader that zips two planes into the existing per-block decode. **Layout shim, not a new dequant.**

> ⚠ **One bit-level unknown that config cannot settle: nibble order within a byte** (low-then-high vs
> high-then-low) may differ between GGML MXFP4 and compressed-tensors packing. Our unpacker's order
> was *transcribed from the reference and verified bit-for-bit against a real gpt-oss checkpoint* —
> the same standard must be met here, which needs one real tensor. Until then, "shim" is the estimate,
> not a guarantee.

**What is quantized is narrow, and that is good news.** The `ignore` list excludes
`re:.*self_attn.*`, `re:.*shared_experts.*`, `re:.*mlp\.(gate|up|gate_up|down)_proj.*`,
`re:.*lm_head.*`, `re:.*vision_tower.*`, `re:.*mm_projector.*`. So **only the routed experts are
4-bit**; MLA, KDA, norms, router, shared experts and the head are all bf16. The MXFP4 work is
confined to the expert loader and **never touches the KDA or MLA parity work**. Per the gpt-oss
precedent this lands CPU-first with the GPU backends declining cleanly.

### 3. RoPE-less MLA — the shipped path assumes RoPE

Phase 0 found `mla_use_nope: True`, `self.rotary_emb = None`, and no `rope_theta`/`rope_scaling`
anywhere in `text_config`. Checked against our implementation:

- `decoder/arch.go:173-182` — `mlaParams` carries `QLoRARank`, `KVLoRARank`, `QKNopeHeadDim`,
  `QKRopeHeadDim`, `VHeadDim`. **There is no NoPE flag.**
- `decoder/forward_deepseek.go:89,108-111` — the forward computes `invFreq := arch.ropeInvFreq(layer)`
  and ropes the query's rope dims and the latent's rope key **unconditionally**; the file's own header
  says *"Decoupled RoPE rides on a separate qk_rope_head_dim slice"* with no conditional.

**Answer: K3's MLA layers need a no-rope config branch, and it does not exist today.** Cheap — a
descriptor flag plus skipping step 3, leaving the `qk_rope_head_dim` slice as extra nope dims carried
through unroped — but it is a real, currently-absent branch, and silently roping a NoPE model is
plausible-wrong output rather than an error.

### 4. `streamExperts` generalization — design note (shared with any future V4)

Current contract (`decoder/weights.go:1032`): one fused `[nExpert, rows, cols]` tensor, hard-validated
`t.Elements() == nExpert*stride`, sliced per expert via `SubF32` so the 3-D f32 is never materialized.
Both new families ship **one tensor per expert** (K3: 82 432 entries = 92 layers × 896 experts × 3).

Sketch — **not code**:

- Add a sibling entry point taking a **per-expert tensor list** (resolved by name template from the
  safetensors index) rather than one fused tensor.
- **Validate the count against config** (`len(list) == NumExperts`) exactly as the fused path validates
  element count — the check must not be quietly dropped, since a missing expert would otherwise route
  to a zero weight and produce plausible-wrong output.
- **Quantize at load, per expert**, reusing `quantizeWM` unchanged — each tensor *is* one expert, so
  the slicing step simply disappears. The bf16-transient win the fused path engineered is inherent
  here, not something to re-earn.
- **Retain the fused path** — gemma4 and the existing MoE families use it; this is an added shape, not
  a replacement.

> **Residency reality, recorded as the expected v1 shape:** K3 has **896 routed experts**, above the
> new **512** router cap (`0018114`), so K3's MoE **declines resident on cuda/webgpu and runs CPU**.
> That is correct and intended — the cap-bump leg deliberately stopped at 512 rather than let an
> unbuilt family set a validated limit. **Do not propose raising it here.**

### 5. Oracle plan sketch for the KDA layers

- **HF reference at tiny scale:** `text_config.model_type: "kimi_linear"` with `auto_map` →
  `modeling_kimi_linear.KimiLinearForCausalLM`, i.e. remote-code, runnable with `transformers` at a
  toy geometry. **Caveat: it calls `fla` for the scan**, so the oracle needs `flash-linear-attention`
  installed and (being Triton) likely a GPU — the same dependency that gates the §1 open question.
  If `fla` proves impractical, the fallback is a **NumPy transcription of the sequential recurrence**,
  which is the shape our DeltaNet was validated against anyway.
- **Tiny fixture:** buildable the way the DeltaNet/Mamba fixtures were (random weights at toy dims,
  pinned golden). ⚠ `testdata/kimi-tiny/` exists but is **config-only — no `model.safetensors`**
  (unlike `testdata/deepseek-tiny/`, which has weights); a K3/KDA fixture must be generated.
- **Harness reuse:** the qwen3_5_moe parity shape applies directly — same axis, same per-layer
  structure, and `deltanet` already has a validated op-for-op reference to diff against.

### 6. Revised bottom line for K3

| item | rides free | new work | class |
|---|---|---|---|
| MLA (24 layers) | shape is ours (`kv_lora_rank: 512`, `kv_a_proj_with_mqa`, `kv_b_proj`) | **NoPE branch**; MLA output gate | small |
| KDA (69 layers) | conv+SiLU, L2 norm, β, gated RMSNorm, output gate, A_log/dt_bias/softplus | **per-channel decay** + gate clamp; **`fla` axis question** | small variant kernel, **blocked on one external read** |
| Router | ours entirely (`e_score_correction_bias`, `noaux_tc`, sigmoid, `first_k_dense_replace: 1`) | config **key renames** only | config-delta |
| Latent-MoE | — | `routed_expert_{up,down}_proj` + norm | small |
| `situ` activation | — | new activation (`SituAndMul`, 2 betas) | trivial |
| Residual projections | — | per-sublayer `*_res_proj`/`*_res_norm`, `attn_res_block_size: 12` | small |
| MXFP4 experts | value table + e8m0 scale semantics | **plane-zip shim**; nibble order unverified | layout shim |
| Loader | `quantizeWM`, streaming discipline | **per-expert tensor entry point** | small |
| Hybrid scheduling | per-layer-kind dispatch exists (granite/nemotron precedent) | 24/69 interleave wiring | small |

**Effort: weeks, not days and not a campaign** — *conditional on the `fla` axis question resolving as
a broadcast change.* If it resolves as a different fold, the KDA row becomes **new scan primitive**
and the estimate moves to campaign class; that is exactly what this gate exists to catch, and it is
**not yet closed**.

**Queue: behind v1.0, and behind DeepSeek-V4's prerequisites** (fp8 blockwise dequant, per-expert
loader) which K3 partly shares. Two cheap unblocks are worth doing independently of K3 because they
serve other work: the **per-expert-tensor loader entry point** (§4) and the **MLA NoPE branch** (§3).

**Cross-reference:** `docs/task-mla-cuda-residency.md` lists K3's 24 MLA layers in its payoff table.
That doc's recommendation stands and is not duplicated here — note only that K3 would land **CPU-only
for its MoE regardless** (896 > 512 cap), so MLA-on-CUDA does not make K3 a GPU-resident model; the
two tasks are independent.

## Verified configuration — Kimi-K3 (`model_type: "kimi_k3"`)

Source: `moonshotai/Kimi-K3/config.json`, `modeling_kimi_linear.py`, `configuration_kimi_k3.py`.
Top level `architectures: ["KimiK3ForConditionalGeneration"]`; **text decoder is a nested
`text_config` with its own `model_type: "kimi_linear"`** and `architectures: ["KimiLinearForCausalLM"]`
(`auto_map` → `modeling_kimi_linear.KimiLinearForCausalLM`).

### Attention — **HYBRID: our MLA on 24 layers, KDA linear attention on 69**

| trait | doc expected | **CONFIRMED reality** | verdict |
|---|---|---|---|
| MLA shape | ours | ✅ **exactly ours**: `kv_a_proj_with_mqa(hidden → kv_lora_rank + qk_rope_head_dim)`, `kv_a_layernorm(512)`, `kv_b_proj(512 → nH*(qk_nope+v_head))`, `q_a_proj`/`q_b_proj` | ✅ |
| `kv_lora_rank` | present | ✅ **512**; `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`, `v_head_dim: 128`, `q_lora_rank: 1536` | ✅ |
| **layer coverage** | all layers | ❌ **`full_attn_layers: [4, 8, …, 92, 93]` — 24 of 93.** The other **69 are `KimiDeltaAttention`** (`kda_layers`) | ❌ **hybrid** |
| RoPE in MLA | YaRN | ❌ **NoPE**: `mla_use_nope: True`, `assert self.use_nope`, and **`self.rotary_emb = None`**. `text_config` has **no `rope_scaling` and no `rope_theta`** — position is carried by the KDA layers | ❌ delta |
| MLA output | plain `o_proj` | ⚠ **`mla_use_output_gate: True`** — extra gate (`g_proj` present on all 93 layers) | ❌ new |
| KDA mixer | — | `KimiDeltaAttention`: `ShortConvolution(kernel=4)` on q/k/v, low-rank forget gate `f_a_proj`/`f_b_proj` + `dt_bias` + `A_log`, `b_proj` (beta), full-rank `g_proj`, chunked delta rule. Tensors confirm on **exactly 69** layers | ❌ new variant |

### Routing — **IS group-limited DeepSeekMoE (with renamed keys)**

| trait | doc expected | **CONFIRMED reality** | verdict |
|---|---|---|---|
| `n_group` / `topk_group` | present | ✅ present but **renamed**: `num_expert_group: 1`, `topk_group: 1`. `modeling_kimi_linear.py` documents the mapping in-source: `num_expert_group -> n_group`, `moe_router_activation_func -> scoring_func` | ⚠ **key rename** |
| grouping active? | yes | ⚠ **degenerate**: guarded by `if self.num_expert_group > 1 and self.num_expert_group > self.topk_group` — with `1`/`1` the group-limit branch **never runs** | ⚠ note |
| `scoring_func` | sigmoid (the K2 gotcha) | ✅ **`moe_router_activation_func: "sigmoid"`** — checked explicitly, not assumed | ✅ |
| `e_score_correction_bias` | present | ✅ config + tensor `…block_sparse_moe.gate.e_score_correction_bias` on 92 layers | ✅ |
| `topk_method` | `noaux_tc` | ✅ `"noaux_tc"`, `use_grouped_topk: True`, `moe_renormalize: True` | ✅ |
| shared expert | ungated | ✅ `num_shared_experts: 2`; tensors `shared_experts.{gate,up,down}_proj` | ✅ |
| `first_k_dense_replace` | present | ✅ **1** — and exactly **1** layer carries plain `mlp.{gate,up,down}_proj` tensors | ✅ |
| latent MoE | — | ❌ **new**: `routed_expert_up_proj` / `routed_expert_down_proj` / `routed_expert_norm` per layer, `routed_expert_hidden_size: 3584`, `latent_moe_use_norm: True` — experts run in a projected latent space | ❌ new |
| activation | silu | ❌ **`hidden_act: "situ"`** (`class SituAndMul`), `activation_situ_beta: 4.0`, `activation_situ_linear_beta: 25.0` | ❌ new |
| block structure | pre-norm residual | ❌ **per-sublayer residual projections** on all 93 layers: `self_attention_res_proj`/`_res_norm`, `mlp_res_proj`/`_res_norm`, `attn_res_block_size: 12` | ❌ new |

### Numbers

| field | value |
|---|---|
| `num_hidden_layers` | **93** (24 full-attn / 69 KDA) |
| `hidden_size` / `num_attention_heads` | 7168 / 96 |
| `num_experts` / `num_experts_per_token` / `num_shared_experts` | **896** / 16 / 2 |
| `num_expert_group` / `topk_group` | 1 / 1 |
| `moe_intermediate_size` / `routed_expert_hidden_size` / `intermediate_size` | 3072 / 3584 / 33792 |
| `kv_lora_rank` / `q_lora_rank` | **512** / 1536 |
| `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | 128 / 64 / 128 |
| `max_position_embeddings` | 1 048 576 |
| RoPE | **none in `text_config`** (no `rope_scaling`, no `rope_theta`; MLA is NoPE) |
| `vocab_size` / `tie_word_embeddings` | 163 840 / **false** |
| `num_nextn_predict_layers` | 0 (no MTP) |
| checkpoint dtype | `mxfp4-pack-quantized` (`weight_packed` + `weight_scale`) |
| repo size | 1.56 TB |

### Vision — the text-only skip has a clean, named boundary ✅

The `qwen2_5_vl` precedent holds. Vision lives in a sibling `vision_config`, and the text decoder is a
self-contained `text_config` with its own `architectures` and `auto_map`. Tensor prefixes separate them
(`language_model.model.*` vs the tower), so a text-only load is a prefix filter, not a surgery.

Named keys to skip: `vt_num_hidden_layers: 27`, `vt_hidden_size: 1024`, `vt_intermediate_size: 4096`,
`vt_num_attention_heads: 12`, `patch_size: 14`, `mm_hidden_size: 1024`, `text_hidden_size: 7168`,
`mm_projector_type: "patchmergerv2"`, `merge_type: "sd2_tpool"`, `merge_kernel_size: [2,2]`,
`pos_emb_type: "divided_fixed"`, `init_pos_emb_{height,width,time}`, plus the top-level
`image_placeholder: "<|kimi_image_placeholder|>"` / `media_placeholder_token_id: 163605`.

---

## `model_type` strings, as they actually appear

| model | `model_type` | `architectures` | note |
|---|---|---|---|
| DeepSeek-V4-Pro / -Flash | **`deepseek_v4`** | `DeepseekV4ForCausalLM` | one string covers both sizes |
| Kimi-K3 (top level) | **`kimi_k3`** | `KimiK3ForConditionalGeneration` | multimodal wrapper |
| Kimi-K3 text decoder | **`kimi_linear`** | `KimiLinearForCausalLM` | ⚠ **the registry key that matters** — a `kimi_k3` alias alone would miss it, and `kimi_linear` is a *separate* Moonshot line that may ship standalone checkpoints |

---

## Taxonomy (updated — the doc's contingency is now the actual case)

The original doc's escape hatch applies: **"If V4/K3 introduce sparse attention, add a
`FeatSparseAttn` declared by no backend."** That is now the real path, and it needs to be wider:

- **`FeatSparseAttn`** (V4) — declared by no backend → CPU-correct day one, GPU auto-decline.
- **`FeatKDA`** or an extension of the existing DeltaNet feature (K3) — the hybrid interleave must be
  expressible, since a backend implementing MLA but not KDA must decline K3 rather than run 69 layers
  wrong. This is exactly the silent-drop class `decoder/features.go` exists to prevent.
- `FeatMLA` + `FeatMoE` remain necessary but are **no longer sufficient** for either family.

---

## Parity + gates — unchanged in shape, re-anchored on what is reachable

- **T1 tiny synthetic golden** stays the primary gate for both.
- **T3-real:** the doc named V4-Flash as "the tractable member" at 284B. It is **0.16 TB** on disk —
  still not tractable on this box, but the *smallest* real option by a wide margin (V4-Pro 0.86 TB,
  K3 1.56 TB). Treat all three as proxy-validated unless a box with the storage appears; label the
  cells "proxy" per the standing rule.
- **Break-it-first** targets change with the architecture: for V4, perturb the `Indexer` top-k
  selection and the `sqrtsoftplus` scoring; for K3, perturb the MLA↔KDA layer assignment (a hybrid
  that silently runs the wrong mixer on a layer is the failure this family invites).
- **Known side-effect** (unchanged): touching the shared MoE loader re-stales `uses:[core]`
  `deps_hash` — scripted goldens-gated refresh, not a re-validation.

---

## Phasing — re-scoped

- **Phase 0 — CONFIRM THE SHAPE (gate). ✅ DONE 2026-08-09. Outcome: REFUTED / PARTIAL.** No loader
  was written; this doc is the deliverable.
- **Phase 1 — decide whether to fund at all.** Both families are now primitive adds, not aliases, and
  neither is loadable today (fp8-block / mxfp4-packed weights, per-expert tensor layout). Per
  `docs/post-v1.0-models.md`, build work queues behind v1.0 regardless — so the only near-term
  question is whether the **prerequisites** are worth doing for their own sake.
- **Phase 2 — the prerequisites, which have value independent of these families.**
  (a) **fp8 `e4m3` blockwise dequant** — unlocks a growing share of frontier checkpoints.
  (b) **per-expert-tensor streaming entry point** beside `streamExperts` — every non-fused MoE needs it.
  (c) **router cap bump** — smallest item, and it independently unblocks the already-shipped
  **Kimi-K2 (384 experts)**, which is the one concrete win still available from this doc.
- **Phase 3 — Kimi-K3 text decoder** (the cheaper of the two: MLA is ours, the router is ours, the new
  work is KDA + latent-MoE + `situ` + residual projections).
- **Phase 4 — DeepSeek-V4** (the more expensive: eight new primitives).
- **MLA-on-CUDA/Metal residency** remains out of scope here and is now scoped in its own doc:
  `docs/task-mla-cuda-residency.md`. That scoping found the payoff is **thinner** than assumed
  (V2-Lite-class only; V3/K2 do not fit regardless, and V4 is not MLA at all) and the cost possibly
  **much lower** (CUDA's existing `attn_batched` may serve the latent geometry as-is). Queued after
  the v1.0 cut.

---

## Leverage — the original claim, corrected

The doc argued "MLA + DeepSeekMoE is the de-facto frontier standard, so each new frontier MoE is a
near-free alias; treat new MLA models as alias-first."

**Phase 0 falsified the premise on its first two test cases.** Both 2026 flagships moved *off* the
V3 shape within one generation — V4 replaced the KV latent with sparse attention + compression, K3
kept MLA but demoted it to a quarter of its layers behind a linear-attention mixer. The honest
generalization is the opposite of the original: **frontier MoE attention is diverging, not
converging, and "alias-first" is a hypothesis that must be config-verified every time — which is
precisely what this gate is for.** The one durable piece of leverage is the *process*: this
verification cost hours and no code, and it prevented an estimate built on a wrong shape.

What genuinely does carry over: the **router cap bump** (unblocks Kimi-K2 today) and any **streaming
loader** work, which serves every large MoE regardless of attention shape.

---

## Not in scope (recorded, so it isn't re-litigated)

- **MLA-on-CUDA/Metal residency** — separate kernel task, scoped in `docs/task-mla-cuda-residency.md`.
- **Kimi-K3 vision** — tower skipped; boundary named above.
- **DeepSeek-V4 sparse attention as a primitive** — its own task, now the *main* cost of V4 rather
  than a contingency.
- **`kimi_linear` as a standalone family** — worth checking separately whether Moonshot ships
  non-multimodal `kimi_linear` checkpoints; if so it is a smaller, more tractable entry point to the
  same primitives than K3 itself.
