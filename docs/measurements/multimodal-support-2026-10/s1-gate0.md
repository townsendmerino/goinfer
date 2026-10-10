# S1 Gate 0: Gemma 4 E2B/E4B decode on Metal. A design map

<!-- citations-at: 93f4ab0ebefc -->

Read-only research (a delegated desk read for S1 of `docs/tasks/task-multimodal-support-2026-10.md`), 2026-10-06, `main` @ 837b86f3. Every claim cites file:line at the commit above (S1.0, c12e2778, later fixed the two bugs this map found; the citations to the code it changed are prose now); `TF/` is the installed transformers package (5.12 where the map says so). Anything **INFERRED** was reasoned
from the code and not run. Nothing here was executed except reading files and the two configs.

## TL;DR

1. **Two pre-existing Metal Gemma 4 gaps that the E-model inherits**, found while mapping. Both are real in code today
   and are missed by the loose existing bars:
   - **(a) Metal never applies the per-layer output scalar on DENSE Gemma 4 layers.** It applies it only inside the
     g4moe join (`metal/gemma4_moe.go:730`). The CUDA fix (`0907f07c`, which also added `KVSrcAtResident`) wired dense
     layers on CUDA (`cuda/backend.go:1528`), and WebGPU wires them too (`gpu/residency.go:649`). Metal has no other
     `LayerScalar` reference outside `embeddinggemma2.go`/`gemma4_moe.go`. `encodeLayerResidualWith`'s dense tail
     ends at the down-proj and residual add (`metal/model.go:3009-3170`). Every E2B layer is dense, so this is
     mandatory for S1.
   - **(b) No resident backend applies the scale-less `v_norm` on a NON-K=V layer.** HF applies
     `value_states = self.v_norm(value_states)` on EVERY non-shared layer (transformers 5.12
     `TF/models/gemma4/modeling_gemma4.py:1247-1254`), and the CPU reference does too, unconditionally (`decoder/forward_gemma4.go:150`).
     Metal ran `v_norm` only `if g.kEqV` (in `metal/model.go`'s attention encode, before S1.0's fix in c12e2778). CUDA
     was the same (the `v_norm` launch in `segA`, as of 2026-10-06; WebGPU too; both fixed 2026-10-07, see the S1.0 block of the task doc). E2B has `attention_k_eq_v=False` (real `config.json`), so `kEqV` is false
     on every layer (`decoder/residency.go:955-963`), and **no E2B layer would get `v_norm`**. The same gap affects the
     already-shipped 12B/26B/31B **sliding** layers on all three GPU backends. INFERRED: the sandwich post-attention
     RMSNorm partly hides it, because a uniform V scale cancels exactly and a per-(pos, head) scale does not. That
     would explain why the existing gates pass: pos0 ≥ 0.97 in `metal/gemma4_dense_scaled_test.go` (raised to 0.99 by S1.0), the 0.90
     backstop in `metal/gemma4_twogeom_test.go` (0.995 since S1.0), and memory's "bar 0.88" for 26B. The 2026-08 scope doc's table
     lists local V as plain `v_proj` (`docs/completed/gemma4-resident-scope.md:121`), which is where the omission
     comes from.
2. **No tiny E-model fixture exists, and none can be loaded today.** Every `testdata/gemma4-*` config has
   `hidden_size_per_layer_input=0`, and only the gitignored, absent `gemma4-vl-tiny` has `num_kv_shared_layers=2`
   (`scripts/pin_gemma4_vl_tiny.py:52-54`). Safetensors PLE loading is refused outright, by the safetensors loader's gemma4 branch in `decoder/weights.go` (as of 2026-10-06, before S1.1 replaced the refusal),
   so a PLE fixture needs either that loader (Phase 4 work) or a GGUF writer. Today the only E-model on the Mac is the
   real `~/models/gemma-4-E2B_q4_0-it.gguf`, which is present.
3. **The bridge problem is the token id.** PLE's token-identity term needs the token id
   (the token-table row read inline in `runLayersGemma4FromEmbed` as of 2026-10-06; S1.2 moved it into
   `Model.gemma4PLEInputs`). Every resident entry point carries only `embedding []float32`
   (`decoder/residency.go:50-87`, `metal/backend.go:601-620`). Recommended: an "augmented embedding row" built in
   `embedResidentInto` (`decoder/residency.go:1630`), `[H ‖ L·P per-layer inputs]`, with the per-layer inputs
   computed host-side by the exact CPU code (bit-identical PLE inputs). Metal already has strict length checks to
   extend (`metal/backend.go:603,668`, `metal/model.go:3662`). Details in §3.1.
4. **The whole per-layer PLE branch can be built from kernels Metal already has:** `quant_vec`, `gemv_w4a8_sa`,
   `swiglu_quant` (whose `glu_act(g)*u` + int8 quant IS `gelu(gate)·ple` + quant), `rmsnorm_f32`, `residual`,
   `scale_vec`. No new MSL needed for step 1. `eg2_gelu_mul_ple` is semantically the same multiply but lives in a
   separate f32 library, and EG2's PLE has no token-identity term (§3.1).
5. **Shared KV on Metal is buffer aliasing.** Metal's KV is linear, not a ring (`metal/model.go:1659-1692`).
   Aliasing also brings 4 traps: kv_store must be skipped or it writes into the source, MC1 slot copies, UploadKV,
   and accounting (§5).
6. **Per-layer FFN width is a hard-coded model-level `r.I` on Metal**, and `I = FFNPerLayer[0]` (`decoder/gguf.go:376-377`).
   E2B is 6144 on layers 0-14 and 12288 on the 20 double-wide shared layers (`use_double_wide_mlp=True`, HF
   `TF/models/gemma4/modeling_gemma4.py:1068-1073`). `r.gu` was sized `2*I` from the one model-level width (in `buildResident`, as of 2026-10-06; S1.3 sizes it to the widest layer), so admitting E2B today would
   overflow scratch silently on unified memory.

---

## 1. The CPU reference (what Metal must match)

### 1.1 Shapes (real E2B, `~/models/gemma-4-E2B-unq/config.json`, read 2026-10-06)

| | value |
|---|---|
| hidden / layers / heads / KV heads | 1536 / 35 / 8 / 1 (global KV heads: `num_global_key_value_heads=None` → 1) |
| head_dim local / global | 256 / 512 |
| layer_types | 4 sliding : 1 full, period 5 (full at 4, 9, …, 34) |
| sliding_window | 512 |
| num_kv_shared_layers | 20 → layers 15..34 own no K/V |
| intermediate_size | 6144; `use_double_wide_mlp=True` → 12288 on the 20 shared layers |
| hidden_size_per_layer_input (P) | 256; PLE vocab = 262144 |
| attention_k_eq_v | False |
| final_logit_softcapping | 30; tied embeddings; pad_token_id 0 |

E4B: no checkpoint on either box (task doc). Shape is INFERRED to be the same family with different numbers.

### 1.2 Forward, one token: `runLayersGemma4` → `runLayersGemma4FromEmbed`

- **Embedding:** `h = Embed[id] × √hidden` (`decoder/forward_gemma4.go:29-40`). The resident twin is
  `embedResidentInto` (`decoder/residency.go:1630-1657`).
- **PLE inputs**, computed once per token from the INITIAL scaled embedding, before layer 0
  (inline in `runLayersGemma4FromEmbed` as of 2026-10-06, which the bare line numbers in this list refer to; S1.2
  moved the block verbatim into `Model.gemma4PLEInputs`):
  - `tok = PerLayerTokenEmbed.Row(pleTokenID) × √P`, shape `[L·P]` (:81-85).
  - `ctx = PerLayerModelProj · h` (`[L·P × H]`, :87), then `× 1/√H` (:88-91), then per layer an RMSNorm over its
    P-segment with the shared weight `PerLayerProjNorm` (:95). This is `normalize(arch, …)`. `arch.RMSAddOne` is
    false for gemma4 (`decoder/registry.go:368`).
  - `perLayer[l] = (tok[l] + ctx[l]) × 1/√2` (:92-99). Matches HF `project_per_layer_inputs`
    (`TF/models/gemma4/modeling_gemma4.py:1781-1811`).
  - `pleTokenID` is the token id for text. At an image/audio position it is **`arch.gemma4.PadTokenID`**: in the
    sequential path at `decoder/generate_gemma4_vl.go:51-64`, and per row in the batched path at
    `decoder/forward_gemma4_batched.go:103-107`. Doc and HF verification: `decoder/forward_gemma4.go:50-57`.
- **Shared KV source map** (`decoder/forward_gemma4.go:81-100`): with `firstShared = L - SharedKVLayers`, a layer
  `l ≥ firstShared` uses the LAST non-shared layer of the SAME type (sliding vs full). The standalone twin is
  `Architecture.gemma4KVSrcAt` (`decoder/arch.go:786-805`), exported as `Model.KVSrcAtResident`
  (`decoder/residency.go:984`), pinned by `decoder/gemma4_kvsrc_test.go:14`. For E2B, sliding layers 15+ read
  layer 13 and full layers read layer 14. Nothing outside decoder calls `KVSrcAtResident` yet (grep).
- **Per-layer attention** (`decoder/forward_gemma4.go:120-168`, the body of the layer loop):
  - pre-attn RMSNorm.
  - Q = QProj → per-head `q_norm` → RoPE, with the per-type table: local base 10k full-rotary, global base 1e6
    "proportional", where the first `GlobalRotaryDim/2` frequencies are real and the tail is zero (`gemma4InvFreq`,
    :255-263).
  - Owning layer (`l < firstShared`, :161-176):
    - K = KProj → `k_norm` → RoPE.
    - V = VProj, or a copy of the raw K on a K=V layer.
    - **`rmsNormNoWeight(v)` on EVERY owning layer** (:172).
    - Then `cache.Append(l,k,v)`.
  - Shared layer: no projection, no append.
  - Attention over `cache.Keys(kvSrc(l))` from `cache.WindowStart(pos, global)` (:180-186), where `global` is THIS
    layer's type, which equals the source's type by construction. Scale is `arch.AttnScale = 1.0`
    (`decoder/registry.go:374-376`).
  - o_proj → `PostAttnNorm` (sandwich) → residual add (:188-190).
- **Dense FFN**, at the **per-layer width** `ffn = arch.ffnAt(l)` (:146, `decoder/arch.go:611-616`):
  - pre-MLP norm.
  - gate/up at width `ffn`, then GeGLU (tanh).
  - down → `PostMLPNorm` → residual (:200-216).
- **PLE branch**, after the FFN residual and on the RAW (un-normed) residual `h` (:218-231):
  - `px = PLEGate·h` (`[P×H]`).
  - `px = geluTanh(px) × perLayer[l]`.
  - `PLEProj·px` (`[H×P]`) → `PostPLENorm` (RMS over H) → `h += …`.
  - Matches HF `TF/models/gemma4/modeling_gemma4.py:1435-1441`.
- **Layer scalar:** `h *= LayerScalar` when ≠ 0 (:233-238, HF :1444). The loader defaults it to 1 when the tensor is
  absent (`decoder/gguf.go:2703-2706`).
- The MoE variant (`lw.gemma4moe != nil`) replaces the FFN, PLE and scalar tail with `gemma4MoEFFN` (:193-198). It is
  PLE-free and not relevant to E2B.
- **After the layers:** `cache.Advance()` (:245). gemma4 uses `manualPos`, because the last layer is shared and never
  appends (`decoder/kvcache.go:94-98`). The final norm + LM head + softcap 30 run in the caller (:25-26).
- **No AltUp, no LAuReL.** `grep -i 'altup|laurel'` hits only `decoder/registry.go` (a family list). The Gemma-3n
  extras are absent from Gemma 4 and from goinfer's forward.

### 1.3 Batched / prefill variant: `runLayersGemma4FromEmbedN` (`decoder/forward_gemma4_batched.go:71-302`)

This is used only for bidirectional-vision checkpoints (26B/31B).
- Same PLE: per-row PAD substitution at image rows (:89-119).
- Same kvSrc (:126-143).
- Appends all K rows before any row attends (:200-209).
- Each row's attention range comes from `gemma4AttendRange`, which takes the union of the window and the image block
  (:33-45).
- PLE + scalar per row (:264-290).
- E2B/E4B use the **sequential** walk instead (`decoder/generate_gemma4_vl.go:10-73`), because
  `use_bidirectional_attention` is unset. HF's E2B attention over an image block is causal.

### 1.4 CPU KV cache for gemma4

- Append-forever, with no rings: gemma4 is listed as keeping append-forever (`decoder/kvcache.go:76-83`).
- A shared layer never appends, so `stride[l]=0` (`decoder/kvcache.go:68-73`).
- `LayerKV(l)` returns nothing for it, and `residentUploadPrefill` skips it via `len(k)==0`
  (`decoder/generate_vl_resident.go:20-34`).

### 1.5 Loading

- **GGUF** (`decoder/gguf.go:2626-2800`):
  - model-level PLE tables at :2632-2643 (`per_layer_token_embd` via `embMat`, int8/int4 per `embedInt4`,
    :1664-1675).
  - `l.KVShared = i >= firstShared`, after which K/V/KNorm are left empty (:2669-2686).
  - FFN at `arch.ffnAt(i)` (:2647, :2689-2697).
  - PLE gate/proj/post_norm (:2761-2771).
  - `FFNPerLayer` from the `gemma4.feed_forward_length` array, with `IntermediateDim = FFNPerLayer[0]`
    (:372-380).
  - `PadTokenID` is **never set from GGUF** (no reference in `decoder/gguf*.go`). It stays 0, which is correct for
    E2B only because E2B's `pad_token_id` is 0.
- **Safetensors:**
  - shared-KV skip (`decoder/weights.go:1004-1022`).
  - FFNPerLayer discovery, recorded only when it varies (:1160-1183).
  - **PLE refused**: the safetensors loader's gemma4 branch in `decoder/weights.go` (as of 2026-10-06, before S1.1 replaced the refusal).

---

## 2. The Metal resident path for Gemma 4 today

### 2.1 Files

| file | role |
|---|---|
| `metal/backend.go` | `BuildResident` (:78-131): feature decline at :89-91, ctx-cap refusal, memory guard (:426-462), `metalResident` adapter (:474-), Forward/ForwardMRoPE (:537-560), ForwardNoLogits (:590), PrefillLast (:768-890), ForwardN (:1021-1050), UploadKV (:1074-) |
| `metal/model.go` | `residLayer` (:121-160), `resident` (:175-512), `buildResident` (:923-1845), host entries (Forward(id) :1944, ForwardEmb :1965, forwardLogits :2020, finalizeLogits :2042), executor (:2098-2250), ForwardArgmax (:2399), encode (encodeTrunkWith :2680, encodeLayerResidualWith :2726-2825, encodeAttentionResidualWith :3097-3260), ForwardBatch (:3262-3345) |
| `metal/geom.go` | per-layer `attnGeom {hd,nKV,kvDim,half,kEqV}`, deduped by value (:25-60) |
| `metal/gemma4_moe.go` | 26B dense‖MoE: `scale_vec` / `zero_vec` MSL (:97-106), the join with the layer scalar (:710-719) |
| `metal/kernels.go` | `allKernels`, compiled as ONE library: `+ moeKernels + gemma4MoeKernels + …` (:2089), so `scale_vec` is available to every build. Also `rmsnorm_f32` (:91), `quant_vec` (:172), `gemv_w4a8_sa` (:425), `kv_store` (:951), `glu_act` with the clamped tanh (:1889-1898), `swiglu_quant` (:1907) |
| `metal/batch.go` | MC3 batched step. `batchIneligible` declines sandwich, kEqV and windows (:308-340), so all Gemma is excluded |
| `metal/greedy_chain.go` | on-device embedding gather. `greedyChainWhyNot` declines `embedScale>1` (:63-91), so Gemma is excluded |
| `metal/prefill.go` | f16 MMA batched prefill. Off for Gemma 4 via `!m.HasPerLayerGeometry()` (`metal/model.go:1315`, `decoder/features.go:529`) |
| `metal/embeddinggemma2.go` | separate f32 encoder (own MSL lib); its PLE block is at :405-410 |

### 2.2 Build (`buildResident`, `metal/model.go:1036-2112`)

- **Dims and scratch:**
  - `H, nL, nH, I, V` come from `m.Dims()` (:987). `I` is model-level.
  - Per-token scratch is sized once, to the max over layers for attention (`maxNHhd`, `maxKvDim`, `maxHd`), but to
    `guDim = I` (or the MoE max) for the FFN (:1532-1538): `r.gu = 2*guDim` (:1673), `r.dq = guDim` (:1706),
    `r.uI = I` (:1721).
- **Per layer** (:1243-1500):
  - Attention, ordinary branch (:1281-1314): `kEqV = m.VFromKResident(l)`. QKV fused as `[Q|K|V]`, or `[Q|K|K]` when
    K=V. `qNorm`/`kNorm` buffers are built via `NewBufferFloats`, which panics on an empty slice (:1356-1360 note).
  - FFN, default branch (:1336-1338): `fuse(Gate, Up)` and `mk(Down)`. The g4moe branch is at :1323-1324.
  - Sandwich norms (:1374-1387). `preNorm` and `postNorm` (PreMLPNorm) at :1345-1351.
  - Per-layer `invf` from `RopeInvFreqLayerResident`, which builds the gemma4 per-layer table
    (`decoder/residency.go:1008-1023`), plus `mscale` and `geom` (:1397-1409).
  - Window uniform `uWindow` from `LayerIsLocalResident` (:1438-1443).
  - QKV bias as zeros sized to Q|K|V (:1444-1454).
- **KV:** one linear buffer per layer, `paddedCtxCap*kvDim` bytes at f16, int8 + per-head scales when `kvI8`
  (:1455-1480). There is no ring, and the window is applied in the attention kernel. MC1 slots are either views of
  one allocation (`kvContig`) or separate copies (:1484-1504).
- **Gemma specifics:**
  - `r.sandwich` (:1034), `finalSoftcap` (host softcap, :1748, applied in `finalizeLogits` :2042-2060).
  - `embedScale` is applied by the decoder (`embedResidentInto`) or by `loadEmbedRow` (:1846-1855).
  - `kvF32` is forced false (:1054), so the KV is f16.
  - The `vNormUnit`/`uZero` plumbing is built for every model (:1735-1743).
  - Width `%8` checks via `bad8` (:1561+). Threadgroup-stage budget (:1590-1612). R18 rows per GEMV:
    `gemvRows.gu = gemvRowsFor(2*I,4)`, and down-staging only if `I <= MaxThreadgroupMemoryLength` (:1614-1626).
- **Batched prefill** is declined for Gemma 4 (`r.prefillOK`, :1178-1180; the message is at `metal/backend.go:1031-1033`).
  The decoder then runs the prompt token by token: `ForwardNoLogits` for all but the last, then `Forward`
  (`decoder/model.go:1615-1643`).

### 2.3 Decode step encode (`encodeLayerResidualWith`, `metal/model.go:3009-3170`; attention :3097-3260)

- **Attention:**
  - pre-norm: `rmsnorm_quant` gives int8 `aq` with a single per-vector scale.
  - fused QKV SA-GEMV with a zero bias.
  - `qk_norm` over Q and K heads (`r.qkNorm`).
  - **`v_norm` only `if g.kEqV`.**
  - `rope2` over Q and K together, sized `nH*half + nKV*half`.
  - `kv_store` into `r.kc[l]/r.vc[l]` at `uPos`.
  - `attention` (or `attention_fa`, which needs hd 64/128 and so never runs for Gemma 4's 256/512) with `L.uWindow`.
  - `quant_vec` of ctx.
  - sandwich: o-proj into `oO`, then `rmsnorm_f32(postAttnNorm)`, then `residual`.
- **Dense FFN** (`else` branch :2768-2824):
  - `rmsnorm_quant(postNorm)` → SA-GEMV gate|up with `rows=2*r.I`.
  - `swiglu_quant(gu, gu.At(r.I*4), dq, dSc, r.uI, act)`.
  - Sandwich: down `pGemv` with `K=r.uI` into `dO` → `rmsnorm_f32(postMLPNorm)` → `residual`.
  - **No layer scalar** (gap 1a).
- **g4moe FFN:** `encodeGemma4MoEFFN`, which ends in `encodeG4Join` with `scale_vec` (`metal/gemma4_moe.go:722-731`).

### 2.4 Entry points

- **Adapter:**
  - `Forward(emb,pos)` → `ForwardMRoPE` checks `len==hidden` (`metal/backend.go:601-620`), then goes to
    `ForwardEmbMRoPEPipe`, the encode-ahead executor. The executor copies `job.emb` into `r.x` at commit time
    (`metal/model.go:2477`).
  - `ForwardNoLogits` (:590) goes through the executor with `noHead`.
  - `ForwardN` → `ForwardBatch` (`metal/model.go:3635`): layer-major in one command buffer, with per-row `batchX`
    slices and uniforms, and a `len(emb) != r.H` check.
  - `ForwardArgmax(id,pos)` uses `loadEmbedRow` (`metal/model.go:2670-2673`).
  - Others: `ForwardSample` (gumbel), `HiddenLast`/`ResidualAll` (`metal/backend.go:1192,1142`), and `StepBatch` (MC3,
    `metal/batch.go:916`, which declines Gemma).
- **UploadKV** (`metal/backend.go:1346-`): writes host K/V rows at `base*kvDim`, f16 or int8.
- **GenerateGemma4VL uses resident decode only when `bidirectional`** (the gate in `GenerateGemma4VL`, `decoder/generate_gemma4_vl.go`, as of 2026-10-06; S1.8 admits E-models too).
  That covers 26B/31B: CPU prefill → `residentUploadPrefill` → `m.resident.Forward(m.embedResident(next), gpuPos)`
  (:169-194). E2B never touches the resident, and `TestGenerateGemma4VL_sequentialPathNeverTouchesResident`
  (`TestGenerateGemma4VL_sequentialPathNeverTouchesResident`, as of 2026-10-06) pinned that; S1.8 changed the contract and the test was replaced on 2026-10-07.

### 2.5 Feature declaration and admission

- **Taxonomy:**
  - `ResidentFeature` and `FeatGemma4EModel` are at `decoder/features.go:100`.
  - It is derived at :268 from `HiddenSizePerLayerInput>0 || SharedKVLayers>0 || len(FFNPerLayer)>0`.
  - Backend sets live in `residentBackendFeatures` (`decoder/features.go:561-844`); Metal's block is :798-831 and
    lacks `FeatGemma4EModel`.
- **Gates:**
  - `residentGateReasonAct` (:372-401) checks missing features, MoE cap, per-layer geometry
    (`residentPerLayerGeomBackends`, Metal true, :488) and Gemma 4 MoE (:504).
  - `decodeRunnerEligible`'s gemma4 case falls through, so gemma4 is admitted by shape
    (`decoder/residency.go:655-673`).
  - `Model.residentAdmission` (`decoder/residency.go:583-605`) runs at load.
  - Metal re-checks in `BuildResident` (`metal/backend.go:89-91`).
- **Tests pinning the E-model decline** (update these on declare):
  - `decoder/gemma4_admission_test.go:62-73`.
  - `decoder/gemma4_emodel_real_test.go:40-49`.
  - `decoder/features_test.go:555-557`, the derivation table, which stays valid.
  - The generated hardware matrix (`decoder/hardware_matrix_test.go`).

---

## 3. Where each E-model difference hooks into Metal

### 3.1 PLE

**Semantics comparison with EmbeddingGemma 2.**
- **EG2 PLE is projection-only:** `RMSNorm(W·emb·H^-½)` with **no token-identity table and no ×1/√2**
  (`embeddinggemma2/model.go:306-316`, Metal `metal/embeddinggemma2.go:371-372`).
- **The per-layer block is identical in structure:**
  - `gate(x)` on the raw residual → `gelu_tanh·ple[li]` (`eg2_gelu_mul_ple`, `metal/embeddinggemma2.go:46-54`) →
    proj → RMS(post) → `x = (x + pr)·scalar` (`eg2_add_scale`, :63-67; used at :405-410).
  - EG2 also applies `v_norm` on every layer (`a.rms(e, s.v, a.ones, …)`, :384). That is the same fact as gap 1b.
- **Reuse is limited:** EG2's kernels live in a separate f32 MSL library (`eg2MSL`) driving f32 GEMMs, not the
  W4A8 resident GEMVs. Its `eg2_gelu` uses an **unclamped** `precise::tanh` (:33-36), while the resident's `glu_act`
  clamps the tanh argument (`metal/kernels.go:2047-2056`, the Gemma overflow fix in memory). So reuse the semantics,
  not the kernel.

**Recommended split.**
- **Host side, in the decoder:** the whole per-token PLE input `perLayer[L·P]`, computed by the **same code** as the
  CPU (inline in `runLayersGemma4FromEmbed` as of 2026-10-06; S1.2 made it `Model.gemma4PLEInputs`). Refactor that block into one helper, e.g. `m.gemma4PLEInputs(h, pleTokenID, dst)`,
  called by runLayersGemma4FromEmbed, runLayersGemma4FromEmbedN (per row), and the resident path.
  - The PLE inputs are then bit-identical to the reference, which removes one parity variable.
  - It keeps the 262144×8960 token table (2.35 G entries: ~2.35 GB int8) **on the host**. It is mmap-aliased from a
    .giw. Uploading it would be a disaster on a 16 GB Mac.
  - Cost: one `PerLayerModelProj` matmul (8960×1536 ≈ 13.8 M MAC) plus a 35 KB row gather per token. INFERRED
    ~1-3 ms CPU, which is on the decode critical path. Measure it in step S1.5. If it matters, move only the
    context-aware GEMV on-device later: `quant_vec(r.x)` → `gemv_w4a8_sa` (rows L·P, K=H) → `scale_vec(1/√H)` → L ×
    `rmsnorm_f32` on segments of P (needs a `uP` uniform) → `residual(tok)` → `scale_vec(1/√2)`.
- **Bridge, the "augmented embedding" convention:** for an E-model, `embedResidentInto(id, dst)` returns
  `[H ‖ perLayer(L·P)]`.
  - All ~40 resident call sites pass through unchanged (`grep embedResident decoder/*.go`: generate, prefill seed,
    spec ngram/blockspec, selftest, VL decode).
  - Image positions: the decoder builds the tail with `PadTokenID`. In S1 they are CPU-prefilled, so this is only
    needed for a later GPU image prefill.
- **Why not a side-channel `SetPLETokens()` interface:** a forgotten call site would silently reuse stale PLE.
  With augmented rows, a site that drops the tail fails a **strict** length check.
- **Metal must assert `len == H + L·P` when PLE is on (and `== H` otherwise) at EVERY entry:** ForwardMRoPE
  (`metal/backend.go:603`; ForwardNoLogits :609 already checks), the executor job, ForwardBatch (`metal/model.go:3662`), PrefillLast/prefillByStep,
  HiddenLast/ResidualAll, ForwardNoLogits, ForwardSample, and StepBatch. Today `ForwardEmbMRoPE`/`forwardHiddenNoHead`
  do a bare `copy(r.x.Floats(), emb)` (:1973, :2000; the executor at :2209), which would **silently truncate** the tail.
  - `ForwardArgmax(id,pos)` and `Forward(id,pos)` take an id and use `loadEmbedRow` (:1846). They need the host PLE
    helper too, or must refuse when PLE is on. They are test-only paths (ForwardArgmax is unused in production per
    the N-03 note at `metal/model.go` softcapParallel comment).
- **Device buffers:**
  - model-level: `pleIn` `[L·P]` f32 shared (written at commit time like `r.x`, `metal/model.go:2477`); for
    ForwardBatch, a per-row `batchPLE [cap·L·P]` (grow in `ensureBatchCap`, :1878); `uP` (=P).
  - per layer: `pleGateW/S` (int4, rows P, K=H), `pleProjW/S` (rows H, K=P), `postPLENorm` (f32 [H]),
    `uLayerScalar` (1 float, for `scale_vec`).
  - scratch: `pleG [P]` f32, `pleQ [P]` int8 + 1 scale (reuse `r.dq/dSc` after the down-proj, or dedicate one).
- **Per-layer encode,** appended after the dense FFN's residual in `encodeLayerResidualWith` (after :2814):
  1. `quant_vec(x → mq, mSc, uH)`. This quantizes the RAW residual. See risk R4, the massive-activation sink. The
     alternative is `f32_to_f16(x)` + `gemv_w4f16_sa` (pipelines already built: `metal/model.go:1112-1120`).
  2. `gemv_w4a8_sa(pleGateW, mq, mSc → pleG)`, rows P.
  3. **`swiglu_quant(pleG, pleIn.At(l·P·4), pleQ, pleSc, uP, uAct)`.** This IS `gelu_tanh(gate)·perLayer[l]`
     followed by int8 quant, with the clamped tanh (`metal/kernels.go:2065-2106`). P=256 is divisible by 4.
  4. `gemv_w4a8_sa(pleProjW, pleQ, pleSc → dO)`, rows H, K=P (staged 2·P bytes, fine).
  5. `rmsnorm_f32(dO, postPLENorm, uH, uEps, uAddOne)`.
  6. `residual(x, dO)`.
  7. `scale_vec(x, uLayerScalar)`. This is gap 1a, needed on every dense gemma4 layer, PLE or not.
- **Decoder accessors needed** (pattern: `decoder/residency.go:965-1023`):
  - `PLEDimResident()`.
  - `Gemma4PLELayerResident(l) (gate, proj *WeightMat, postNorm []float32)`.
  - The augmented-embedding width.

### 3.2 Shared KV (`SharedKVLayers`)

- **Build,** for `l ≥ firstShared`: `src := m.KVSrcAtResident(l)`, which exists (`decoder/residency.go:984`).
  - **Assert** `geom(l) == geom(src)` and `window(l) == window(src)`; decline otherwise. Same-type sourcing makes this
    true by construction.
  - Alias the buffers: `r.kc[l], r.vc[l] (, r.ks[l], r.vs[l]) = r.kc[src], …`, and set
    `r.kvSlotBytes[l] = r.kvSlotBytes[src]`.
  - Q-only projection: `L.qkvW, L.qkvS = mk(&lw.QProj)`, with `qkvBias` sized `nH·hd`.
  - Only `qNorm`. `lw.KNorm` is empty and `NewBufferFloats(empty)` panics, so bind a dummy/unit `kNorm`.
  - Mark the layer: `L.kvShared = true`, `L.kvSrc = src`.
- **Geometry:** add `kvShared` to `geomFor`'s key (`metal/geom.go:39-40`) with `uKtotal = 0`, so `rope2` rotates Q
  only. `qk_norm` grid `nH*tgReduceAttn` (Q heads only).
- **Encode** (`encodeAttentionResidualWith`):
  - QKV SA rows = `nH·hd`.
  - Skip K-norm, `v_norm`, K-rope and **`kv_store`**.
  - `attention` reads `r.kc[l]` (the alias) with `uNKeys = pos+1` and the layer's own window. Inside one command
    buffer the source layer's `kv_store` for this `pos` was already encoded earlier (src < l), so ordering is correct.
- **MC1 slots** (`metal/model.go:1692-1719`):
  - In the `kvContig` path, `r.kc[l].At(s*kvSlotBytes[l])` coincides with the source's view when aliased.
    INFERRED OK.
  - The **non-contig path (int8 KV, `allocSlots=1`) allocates a fresh `byteBuf` for every layer with a non-zero
    `kc`**, so a shared layer gets its own empty buffer in slots ≥ 1. That is a silent garbage read. Alias in that
    loop explicitly, as `b.kc[l] = b.kc[src]`.
- **UploadKV** (`metal/backend.go:1346`): refuse (or no-op) a shared layer explicitly. Today `residentUploadPrefill`
  skips it only because the CPU cache is empty there (`decoder/generate_vl_resident.go:26-28`), and an upload through
  the alias would overwrite the source.
- **Bookkeeping:**
  - `kvBuffers()` (`metal/model.go:2551-2558`) will list aliased buffers twice. The ledger-based `ReleaseAll` is
    safe, but check any residency-set or teardown-consistency test that assumes uniqueness (`residencyBufs`, :262).
  - Host readers of `kc` (snapshots, `kvHostOff`) must not double-export.
- **Accounting:**
  - `ResidentKVBytes("metal")` (`decoder/residentneed.go:46-77`) loops over every layer and so prices the 20
    aliased layers too. E2B f16 KV per position is 43 KB priced against 18.4 KB allocated (2.33×, my arithmetic from
    the shapes in §1.1).
  - That doc says "exact to the allocation", so skip `l ≥ firstShared` there. The memory guard and `metalKVSlots`
    use it (`metal/backend.go:387-399, 375-405`).
  - The CPU fit guard `kvBytesForCtx` (`decoder/arch.go:583-608`) also prices shared layers (`kvDimAt` is non-zero),
    but it is conservative. CPU allocates nothing for them.

### 3.3 Per-layer FFN width (`FFNPerLayer`)

- **Model-level `r.I` and `r.uI` are wired through every dense FFN dispatch:**
  - gate|up rows `2*r.I` (`metal/model.go:3076-3080`).
  - the `swiglu_quant` up offset `r.gu.At(r.I*4)` and `r.uI` (:2802).
  - down `K=r.uI` (:2804, :2816-2818).
  - LoRA offsets (:2799).
  - `gemvRows.gu`/`down` (:1618-1621) and `bad8` (:1561+).
- **Needed:**
  - Per-layer `L.inter` and `L.uI`. Source either a new accessor `FFNWidthAtResident(l)` → `arch.ffnAt(l)`
    (`decoder/arch.go:611`), or `lw.GateProj.Rows()`. Assert they agree and decline if not.
  - `guDim = max_l inter` for `r.gu`/`r.dq`.
  - `gemvRows.gu = min_l gemvRowsFor(2*inter_l, 4)` (mirror the qkv loop at :1616-1620).
  - Down staging gated on `max_l inter`.
  - `bad8` on every layer's width.
  - Swap `r.I`/`r.uI` → `L.inter`/`L.uI` at the dispatch sites.
  - `batch.go`/`prefill.go` also read `r.I` (`metal/batch.go:338-339`), but both are declined for this family. Add
    explicit E-model guards anyway (§5, R7).
- Budget: E2B max 12288. Threadgroup staging for the dense down is not counted
  (`metal/model.go:758-760` comment). The R18 staged down needs `I ≤ MaxThreadgroupMemoryLength` (12288 ≤ 32768 OK).
- `use_double_wide_mlp` is the HF origin of the varying widths (`TF/models/gemma4/modeling_gemma4.py:1068-1073`). The GGUF carries
  the array.

### 3.4 Decoder bridge (`GenerateGemma4VL`, generate paths, features)

- **Text:**
  - Once Metal declares `FeatGemma4EModel`, `residentAdmission` admits E2B and the stateless Generate path uses
    `residentPrefillSeed` (`decoder/model.go:1574-1643`).
  - `PrefillLast` declines on per-layer geometry, so the prompt runs sequentially per token through the executor,
    and every embedding comes from `embedResident`, so it carries PLE.
  - Nothing else in the decoder needs to know.
- **Image:**
  - Change the gate in `GenerateGemma4VL` (as of 2026-10-06; done by S1.8) from `bidirectional && tryClaimResident()` to also admit
    the sequential class when the resident declared the E-model feature.
  - The sequential CPU prefill → `residentUploadPrefill` already skips shared layers.
  - Decode then uses `m.resident.Forward(m.embedResident(next), gpuPos)`, which carries the token's PLE tail.
  - Flip `TestGenerateGemma4VL_sequentialPathNeverTouchesResident` into a positive test (the fake resident must
    accept the augmented width).
- **CUDA/WebGPU** still decline `FeatGemma4EModel`, so E2B stays CPU there. No change.

---

## 4. Fixtures and tests that exist

| what | where | status |
|---|---|---|
| Real E2B GGUF CPU vs HF bf16, argmax + sample cosine ≥ 0.98 | `decoder/gemma4_parity_test.go:16-70`, golden `testdata/gemma4_forward_golden.json` ("The capital of France is", argmax 7001) | heavy; `~/models/gemma-4-E2B_q4_0-it.gguf` present on the Mac |
| Real E2B config/load | `decoder/gemma4_load_test.go:16,94` | heavy |
| Real E2B declines resident | `decoder/gemma4_emodel_real_test.go:15-50` | invert for metal on declare |
| Synthetic E-model admission | `decoder/gemma4_admission_test.go:62-73`; derivation `decoder/features_test.go:553-555` | update metal |
| kvSrc map | `decoder/gemma4_kvsrc_test.go:14-88` | reuse |
| Shared-KV-only tiny (PLE-free): `gemma4-vl-tiny`, 4 layers, `num_kv_shared_layers=2`, K=V globals | `scripts/pin_gemma4_vl_tiny.py:36-56`; tests `decoder/gemma4_vl_test.go:24,96` | gitignored (`.gitignore:213`), **absent on the Mac**; regen needs `~/.venv-vl` |
| E2B vision tower on Metal | `metal/gemma4_vision_e2b_real_test.go:21` | tower only |
| Metal resident-vs-CPU patterns | `metal/gemma4_twogeom_test.go:41-63,181-232` (direct `buildResident`, int4 both sides, argmax + 3% near-tie rule, cosine 0.90 backstop at the time, 0.995 since S1.0); `metal/gemma4_dense_scaled_test.go:24-111` (calibrated envelope: mean Metal-vs-CPUint4 ≥ mean CPUint4-vs-f32, pos0 ≥ 0.97); `metal/gemma1_resident_parity_test.go:18-56` (via `Load(Backend:"metal")` + `ResidentForwardForTest`, 16 positions, worst cosine ≥ 0.999); localize helpers `forwardTrunkForTest`/`forwardSubCaptureForTest` (`metal/model.go:2728,2728`) | mirror |
| **Tiny E-model with PLE + shared KV + varying FFN** | none | **must be built (S1.1)** |

Building the tiny E-model fixture requires either:
- **safetensors PLE loading.** That means implementing the tensors refused by the safetensors loader's gemma4 branch in `decoder/weights.go` (as of 2026-10-06, before S1.1 replaced the refusal); the HF
  names are at `TF/models/gemma4/modeling_gemma4.py:1371-1378, 1600-1613`.
- **or a tiny GGUF writer.**

The first also unlocks `~/models/gemma-4-E2B-unq` as an f32/bf16 reference for the real-checkpoint per-layer
differencing that CLAUDE.md recommends.

Pin the fixture like `pin_gemma4_vl_tiny.py`, with these settings:
- `hidden_size_per_layer_input=16`, `num_kv_shared_layers=2` of 5-6 layers (both types shared),
  `use_double_wide_mlp=True`, `attention_k_eq_v=False`.
- Strengthened (non-identity) norms and `layer_scalar≠1`, plus random PLE weights. CLAUDE.md requires the tiny-fixture
  degeneracy guard.
- A prompt of ≥ 9 tokens with `sliding_window=4`, so windows bind on both owning and shared layers. The ≥ 2-token
  lesson applies.

---

## 5. Risks and traps (the "zero looks legal" class first)

- **R1. kv_store on a shared layer writes INTO the source through the alias.** A zero-row K/V GEMV output (garbage
  scratch) lands at `pos` in layer `src`'s cache after src has attended. That corrupts every later token, silently.
  Skip it explicitly and plant it as a defect.
- **R2. Per-layer FFN overflow.** `r.gu` is sized `2·6144`, but layers 15-34 need `2·12288`. On unified memory the
  SA-GEMV writes past the buffer into adjacent MTLBuffers (the `metal/backend.go:585-591` narrative). With the
  dispatch grid left at `2*r.I`, the "up" half is read from the middle of gate and down reads a 6144 stride: finite,
  plausible, wrong.
- **R3. Silent PLE truncation** via `copy(r.x.Floats(), emb)` on any entry not given the strict length check (§3.1).
- **R4. Activation quantization of the raw residual for PLEGate.** Gemma's BOS sink has massive residual channels.
  Per-vector int8 (`quant_vec`, one scale per vector) would crush the rest. The o-proj and FFN inputs are normed
  first, but the PLE gate input is not. Measure on real E2B (PLE-gate output cosine vs CPU at pos 0) before choosing
  int8; the fallback is the W4F16 lane kernels. `ActivationQuantHazard` (`decoder/features.go:379`) is the existing
  taxonomy hook if it proves hostile.
- **R5. Gaps 1a/1b (layer scalar, v_norm)** must land before or with S1. The E-model gate cannot pass honestly
  without them. They are also live bugs in shipped dense/MoE Gemma 4 on Metal (and v_norm on CUDA/WebGPU), so they
  deserve their own commit and gate. Re-measure the dense-twogeom/dense-scaled/26B bars after the fix. INFERRED: the
  26B "bar 0.88, int4-hostile" may partly be this.
- **R6. MC1 int8-KV slot copies** allocate per layer, so a shared layer must alias in every slot (§3.2).
- **R7. Paths that are declined only incidentally:** MC3 step (sandwich, `metal/batch.go:322-323`), greedy chain
  (`embedScale>1`, `metal/greedy_chain.go:67-68`), f16 MMA prefill (`HasPerLayerGeometry`, `metal/model.go:1315`),
  W4F16 lane (sandwich, `metal/model.go:3202`). None of them runs PLE/shared-KV/per-layer FFN. Add explicit
  `pleDim>0 || sharedKV` declines to each, because a future uniform-geometry E-variant (or a step-kernel widening)
  would otherwise admit them. The doc's own lesson: "explicit, checked directly rather than assumed caught".
- **R8. Memory accounting:**
  - (a) KV overcount on shared layers (§3.2).
  - (b) `PerLayerTokenEmbed` (~2.35 GB int8 on E2B) is priced as device weight AND, for a direct GGUF, a second host
    copy (`ResidentHostCopyBytes`, `decoder/weightbytes.go:184-195`, counted in `residentWeightBytesSplit` :217).
    Only CUDA subtracts host-side tables (`residentHostSideBytes`, :150-158).
  - Under the recommended design the table never reaches the device, so add metal to the host-side subtraction
    (count it once). Without that, `fit` and the guard over-price E2B by ~2.35-4.7 GB. INFERRED rough budget:
    ~0.95 GB int4 layers + 0.4 GB int8 tied embed + the PLE table.
- **R9. PadTokenID is unset on the GGUF path** (`decoder/gguf.go:355-380`). It defaults to 0, which is correct for
  E2B by coincidence. A future E-checkpoint with pad ≠ 0 would mis-run image PLE silently. Read
  `tokenizer.ggml.padding_token_id`, or assert.
- **R10. Greedy-token bar.** "Identical 32 greedy tokens vs CPU" (task doc) compares two different quantization paths:
  - Metal: f16 scales and per-vector int8 activations, f16 KV.
  - CPU: f32 scales, its own activation grouping, f32 KV.

  This is not bit-identity, unlike F2b, which compared towers under one decoder. Pre-register either "identical, OR
  the first divergence is a near-tie (<3% gap in the CPU distribution, the twogeom rule)" or a teacher-forced
  per-position argmax agreement floor, with an ambiguous → parked band.
- **R11. Sliding × shared:** consistent by construction (same type → same window; Metal's KV is linear with no ring),
  and `LayerIsLocalResident(l)` is true for shared sliding layers. But the build should assert
  `window(l)==window(src)`. A future checkpoint sharing across types would break the CPU reference too.
- **R12. Test fixture absence.** `gemma4-vl-tiny` is gitignored and missing locally. Any gate that relies on it must
  regenerate it and pin the transformers version: memory notes 5.12 vs 5.15 differences, and the local venv has 5.12.

**CUDA, briefly (later):**
- Same model-level `r.inter` (`cuda/resident.go:3190, 3454-3493`).
- `v_norm` only on kEqV (as of 2026-10-06; every K/V-owning Gemma 4 layer since 2026-10-07, see the S1.0 block of the task doc).
- No PLE (as of 2026-10-06 the dense tail's comment said "no PLE branch yet"; the PLE branch landed 2026-10-07, see the S1-on-CUDA block of the task doc).
- Projections are built for every layer (`cuda/backend.go:384-387`), so a shared layer's empty K/V would fail or be
  zero.
- `KVSrcAtResident` is unused.
- The dense layer scalar IS wired (`cuda/backend.go:1528`).
- `residentHostSideBytes` already treats both PLE tables as host-side, while the recommended Metal design uploads
  `PerLayerModelProj` only if step S1.5b moves it on-device.

---

## 6. Proposed build order and gates

Every step is gated before the next. Steps S1.2-S1.4 can be tested by calling `buildResident` directly, the way
twogeom does (`metal/gemma4_twogeom_test.go:52`), before Metal declares the feature.

| step | change | gate (tier, size) |
|---|---|---|
| **S1.0** | Metal dense layer scalar (`scale_vec` after the dense FFN residual) + `v_norm` on every owning Gemma 4 layer (Metal; CUDA/WebGPU `v_norm` noted for their own pass) | dense-twogeom + dense-scaled resident parity before/after, recorded (quick, seconds; dense-scaled 471 MB local); planted: scalar removed / v_norm removed must turn a *tightened* bar red (calibrate the bar from the fixed run); 26B re-check on the night queue |
| **S1.1** | Safetensors PLE loader + `scripts/pin_gemma4_emodel_tiny.py` (PLE + shared KV + double-wide + strengthened) → tiny E fixture + HF f32 golden (every position) | CPU vs HF: argmax all positions, cosine ≥ 0.99999 (quick) |
| **S1.2** | Decoder: PLE-input helper shared by the CPU paths (refactor, bit-identical) + augmented `embedResidentInto` + accessors (`PLEDimResident`, `Gemma4PLELayerResident`, `FFNWidthAtResident`) | CPU goldens unchanged (gemma4 parity + tiny E golden bit-identical before/after the refactor); unit: augmented tail == CPU `perLayer` bytes |
| **S1.3** | Metal per-layer FFN width | tiny E (or a synthetic varying-width dense) resident vs CPU; planted "one width" → red |
| **S1.4** | Metal shared-KV aliasing (Q-only QKV/rope/norm, skip kv_store, slot aliasing, UploadKV refusal) | tiny E with PLE temporarily forced off is not possible on a real E fixture, so gate S1.3+S1.4+S1.5 together on G1; additionally a pure unit: after a forward, `kc[src]` rows bit-equal with and without the shared layers present (catches R1) |
| **S1.5** | Metal PLE branch (host inputs + 7-dispatch tail) and strict length checks at every entry | **G1** below |
| **S1.5b** (optional, speed) | context-aware PLE GEMV on device | G1 again + host-cost record |
| **S1.6** | Declare `FeatGemma4EModel` on metal; explicit E-model declines in batch / greedy chain / prefillOK / f16 lane; `ResidentKVBytes` skips shared layers; Metal host-side PLE table pricing; update admission tests + hardware matrix | `go vet`, `staticcheck`, tagged builds; admission tests; `ResidentKVBytes` == allocated bytes test on the tiny E |
| **S1.7** | real E2B text | **G3** |
| **S1.8** | `GenerateGemma4VL` resident bridge for the sequential class | **G4** + positive version of the sequential-path test |
| **S1.9** | speed | night queue: E2B decode tok/s, resident vs CPU, same-session interleaved (`night.py add`) |

**Gates (pre-register before measuring):**
- **G1. Tiny E resident vs CPU, every position, int4 both sides.**
  - Primary: argmax with the 3% near-tie rule (as twogeom).
  - Secondary: the calibrated envelope (as dense-scaled): mean Metal-vs-CPUint4 cosine ≥ mean CPUint4-vs-f32.
  - Localize helpers per layer for triage.
  - Cost: seconds, by day.
- **G2. Planted defects. Each must turn G1 red.**
  - (1) PLE branch skipped.
  - (2) token-identity term zeroed, or ×1/√2 dropped.
  - (3) shared-KV source off by one (`src-1`).
  - (4) one model-level FFN width.
  - (5) kv_store not skipped on a shared layer (R1).
  - (6) layer scalar dropped.
  - (7) v_norm dropped on non-K=V layers.
  - If one does not go red, the fixture is degenerate along that axis. Fix the fixture, not the bar.
  - Build-time test seams (`goinfer_testhooks`); seconds each.
- **G3. Real E2B text.**
  - Setup: ~8 fixed prompts × 32 greedy tokens, Metal resident vs CPU, both loaded from
    `~/models/gemma-4-E2B_q4_0-it.gguf` int4.
  - Bar: identical tokens or a near-tie first divergence (R10), plus teacher-forced argmax agreement over all
    positions ≥ a pre-registered floor, with an ambiguous band.
  - Heavy (`GOINFER_HEAVY_TESTS=1`).
  - INFERRED cost: about 2-4 min, dominated by CPU decode of ~500 tokens and two loads. Day-OK, one cell.
- **G4. Real E2B image chat.** The served reply through Metal resident decode matches the CPU decoder (F2b shape,
  `docs/multimodal.md:1044,1029`), under the same R10 bar caveat. Same size as G3.
- **Speed (night only):** E2B decode tok/s, resident vs CPU, same-session interleaved; record host PLE ms/token.

Size: M-L, consistent with the task doc. S1.0 is S; S1.1 is S-M; S1.3-S1.5 together are M; S1.6-S1.8 are S each. No
aikit API is needed: every kernel is in goinfer's own MSL.
