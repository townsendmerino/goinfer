# Task: Gemma 4 26B-A4B — MoE bring-up + weight streaming

One program, two parts. **Part A** makes goinfer able to load and run Google's
**Gemma 4 26B-A4B** (HF `google/gemma-4-26B-A4B-it`, `Gemma4ForConditionalGeneration`) on
the pure-Go CPU path, parity-gated. **Part B** closes the four streaming gaps against
`drumih/turbo-fieldfare`, which publishes **5.1–6.3 tok/s on an 8 GB M2 MacBook Air** and
**31–35 tok/s on a 24 GB M5 Pro** on this exact checkpoint.

**Why it's one task.** goinfer currently cannot load the checkpoint at all — `gemma4` is
registered dense-only (`decoder/registry.go:22`, "E2B/E4B + 12B dense") — so per
`docs/benchmarks.md`'s own methodology rule there is no comparable cell and cannot be one
until Part A lands. And Part B's best latency-hiding lever only exists *because* of the
architecture Part A adds (§B3). The phase list at the end sequences both.

The good news is that 26B-A4B is a **sparse-MoE variant of the Gemma 4 stack goinfer
already runs** — same `model_type`, same attention, same RoPE, same softcap. The delta is
the FFN sub-block.

Reference: HF `config.json` + `model.safetensors.index.json` on the model repo, and
`transformers/models/gemma4/modeling_gemma4.py`. **Phase 1a pins the math against a local
checkout** — items marked ⚠️ were read from the published config and tensor index, not
from a pinned `transformers` version.

---

# Phase 1a — RESOLVED (pinned against `transformers` 5.12.0)

Read `transformers/models/gemma4/modeling_gemma4.py` (identical in 5.10.2 ±1 line) and
`configuration_gemma4.py`. The MoE branch is fully present (the earlier fetched copy was the
one elided). Every ⚠️ above is resolved; line refs are into 5.12.0's `modeling_gemma4.py`.

**Decoder-layer FFN wiring — the inferred pseudocode in §A1 is CORRECT** (`Gemma4TextDecoderLayer.forward`, L1398–1455). Confirmed, with one subtlety that matters:

```
residual = h                                   # L1424 (post-attention hidden)
x1 = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(h) ) )   # L1425–1429
# MoE branch reads the RAW residual h, NOT pre_feedforward_layernorm(h):
_, w, idx = router( h.reshape(-1, D) )         # L1432–1433  (router has its OWN norm — see below)
x2 = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(h), idx, w ) )  # L1434–1437
h  = residual + post_feedforward_layernorm( x1 + x2 )    # L1440–1443  (single joint post-norm on the SUM)
# … PLE block skipped (hidden_size_per_layer_input=0) …
h *= layer_scalar                              # L1454  (VERY END, after the residual add)
```

- **⚠️ Three independent normalizations of the same `h`.** The dense branch consumes
  `pre_feedforward_layernorm(h)`; the expert branch consumes `pre_feedforward_layernorm_2(h)`
  (its **own** learned-weight RMSNorm, on the raw residual, **not** the dense pre-norm); the
  router consumes `h` through its **own weightless** RMSNorm + learned scale. **Do not share
  the dense pre-norm output with the MoE branch** — that's the exact subtle drift a
  cosine-only gate hides. `post_feedforward_layernorm` is applied **once** to `x1+x2`, then
  the residual is added.
- **⚠️ `layer_scalar` placement: END, after the residual add** (`h *= layer_scalar`, L1454) —
  a whole-layer multiply, NOT inside a sublayer. It's a `register_buffer("layer_scalar",
  torch.ones(1))` (L1381) — a **scalar buffer** (shape [1]), not a per-channel vector.
  `LayerWeights.LayerScalar` already exists; wire it into `runLayersGemma4`'s tail.
  (Dense E2B/E4B/12B run the same tail with `layer_scalar` — it's 1.0 in those checkpoints, so
  the existing dense parity is unaffected; the MoE checkpoint trains it away from 1.)

**Router (`Gemma4TextRouter`, L1333–1366) — all three §A2 deltas confirmed:**
- **`router.norm` is WEIGHTLESS.** `Gemma4RMSNorm(hidden, with_scale=False)` (L1341); `with_scale=False` ⇒ no weight parameter (L194/197/199/209). **No `router.norm.weight` tensor; no LayerWeights slot for it** — it's a pure RMS normalize.
- **`scalar_root_size = hidden_size**-0.5`** (L1338) — a constant (1/√2816 ≈ 0.018842 for 26B-A4B), not learned.
- **`router.scale` IS a learned `[hidden]` parameter** (`nn.Parameter(ones(hidden))`, L1343). Applied element-wise **before** the projection: `norm(h) * scale * scalar_root_size` (L1347–1348). This **is** a tensor to load (index name `router.scale`). So the router pre-projection input = `rmsnorm(h) ⊙ router.scale · (1/√hidden)` — distinct from `pre_feedforward_layernorm_2`.
- **`NormTopKProb` is UNCONDITIONALLY true.** `top_k_weights /= top_k_weights.sum(-1)` (L1361) with no config gate — set it `true` regardless of `config.json`.
- **Selection = softmax-over-ALL then top-k by probability then renorm** (L1350–1361): `softmax(proj(x))` over all 128 → `topk(probs, k)` → renorm. Mathematically the same shape `routeExperts(logits, sigmoid=false, norm=true)` already produces (the gpt-oss equivalence). So `routeExperts` is reusable; the router pre-norm/scale is caller-side, and `per_expert_scale` is a post-step.
- **`per_expert_scale` is a learned `[num_experts]` vector**, applied to the **renormalized** top-k weights, indexed by the selected experts: `w *= per_expert_scale[idx]` (L1364). New `MoEConfig`/`LayerWeights` field + an extra return-path arg to `routeExperts`. (`register`/`proj.weight` is f32 — keep full precision.)

**Experts (`Gemma4TextExperts`, L1293–1330) — §A3 split + activation confirmed:**
- **`gate_up` split is CONTIGUOUS.** `linear(x, gate_up_proj[e]).chunk(2, dim=-1)` (L1324) ⇒ `gate = out[:704]`, `up = out[704:]`. Not interleaved. `loadFusedExperts` (contiguous halves) is reusable.
- Shapes: `gate_up_proj` `[E, 2·inter, hidden]` = `[128, 1408, 2816]` (L1302, `linear` computes `x @ Wᵀ`, so it's `[out, in]`); `down_proj` `[E, hidden, inter]` = `[128, 2816, 704]` (L1303).
- **Expert activation is gelu-tanh GeGLU**: `act_fn(gate) * up` (L1325) with `act_fn = ACT2FN[hidden_activation]` (L1304); **`hidden_activation = "gelu_pytorch_tanh"`** (config L164). So the experts use goinfer's `ActGeluTanh`, **not SiLU** — `swiGLUExpert`/`moeMLP` (which hardcode/require SiLU) can't be reused as-is; gemma4's own forward must run a gelu-tanh expert. The **dense branch** (`Gemma4TextMLP`, gelu-tanh GeGLU, `down(act(gate(x))·up(x))`) uses `intermediate_size 2112` (`use_double_wide_mlp=false` ⇒ no doubling); experts use `moe_intermediate_size 704`.

**Residual naming caveat resolved:** the `experts.gate_up_proj` linear output dim is `2·704`; `top_k_weights` multiply happens INSIDE the expert loop (L1327), before `index_add_`, so the per-token weight (already `× per_expert_scale`) scales the expert's contribution — matches `moeMLP`'s weighted sum.

Nothing above needs a `transformers` newer than what's installed; the ⚠️ markers below are now settled by these refs.

---

# Part A — MoE bring-up

## Config (Gemma 4 26B-A4B, from `text_config`)

```
model_type gemma4  /  text_config.model_type gemma4_text
num_hidden_layers 30   hidden_size 2816   rms_norm_eps 1e-6   vocab_size 262144
tie_word_embeddings true   final_logit_softcapping 30.0

# attention (unchanged from the dense 12B bring-up):
num_attention_heads 16   head_dim 256   num_key_value_heads 8       # sliding layers
global_head_dim 512      num_global_key_value_heads 2               # full layers
attention_k_eq_v true    num_kv_shared_layers 0
layer_types: 5 sliding_attention : 1 full_attention  ×5   (layers 5,11,17,23,29 global)
sliding_window 1024      max_position_embeddings 262144
rope: sliding  {default,      theta 1e4}
      full     {proportional, theta 1e6, partial_rotary_factor 0.25 → rotary_dim 128}

# NEW — the FFN sub-block:
enable_moe_block true    use_double_wide_mlp false
intermediate_size 2112              # the always-on DENSE mlp width
num_experts 128   top_k_experts 8   moe_intermediate_size 704       # 2112 = 3 × 704
hidden_size_per_layer_input 0       # PLE-free, like the 12B dense
```

Note `top_k_experts`, **not** `num_experts_per_tok` — see §A4.

## What already exists (do not rebuild)

| need | where it already lives |
|---|---|
| `gemma4` model_type + descriptor | `decoder/registry.go:22`, `:165` |
| global wide head / single-KV / K=V / partial rotary | `gemma4Params` — `GlobalHeadDim`, `NumGlobalKVHeads`, `KVShared`, `GlobalRotaryDim` |
| proportional (p-)RoPE on global layers | `decoder/forward_gemma4.go:237–241` |
| 5:1 local:global interleave, dual-base RoPE | `layerIsGlobal` + `RoPELocalBase`/`RoPEGlobalBase` |
| final-logit softcap 30 | `FinalLogitSoftcap` (already wired for this family) |
| sandwich norms, GeGLU, √hidden embed scale, tied head | `NormSandwich4`, `ActGeluTanh`, `EmbedScale`, `TiedLMHead` |
| `text_config` flattening for nested configs | `decoder/config.go:673–684` |
| per-layer FFN width dispatch | `arch.ffnAt` (`decoder/arch.go:233`) |
| softmax top-k router + renorm | `routeExperts` (`decoder/mlp.go:159`) |
| routed-expert MLP driver | `moeMLP` (`decoder/mlp.go:81`) |
| **fused stacked expert loader** (`gate_up_proj` / `down_proj`) | `loadFusedExperts` (`decoder/weights.go:641`, gpt-oss path) |
| GGUF stacked-expert reader | `stackedExperts` (`decoder/gguf.go:1212`) |

That is most of the surface. The work below is genuinely five things.

## A1. A parallel dense-MLP + MoE FFN sub-block (the real work)

Every layer carries **both** a dense MLP and a 128-expert MoE block, run on separate
normed copies of the residual and summed. Tensor set per layer (verbatim from the
safetensors index):

```
input_layernorm                     post_attention_layernorm       # attention sandwich
pre_feedforward_layernorm           mlp.{gate,up,down}_proj        # dense branch
post_feedforward_layernorm_1
pre_feedforward_layernorm_2         experts.gate_up_proj           # MoE branch
post_feedforward_layernorm_2        experts.down_proj
                                    router.{proj.weight, per_expert_scale, scale}
post_feedforward_layernorm                                         # joint post-norm
layer_scalar
```

⚠️ Inferred wiring (pin in 1a — the fetched `modeling_gemma4.py` elided the second
branch):

```
r = h
x1 = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(h) ) )
x2 = post_feedforward_layernorm_2( moe( pre_feedforward_layernorm_2(h) ) )
h  = r + post_feedforward_layernorm( x1 + x2 )
h *= layer_scalar
```

This does **not** fit `NormSandwich4`, which assumes one pre-norm and one post-norm per
sublayer. `runLayersGemma4`'s MLP section (`forward_gemma4.go:174–188`) becomes a branch:
dense-only when `arch.MoE == nil`, the pair above when it is set. Seven norm slots and
`layer_scalar` are new `LayerWeights` fields.

`layer_scalar` is a scalar per layer applied **after** the residual add — a genuinely new
primitive in this codebase, and a likely source of a slow drift if placed wrong. Gate it
with a per-layer trace, not just an end-of-stack cosine.

**This branch is also the streaming program's latency-hiding opportunity — see §B3.**

## A2. Router: pre-norm, learned scale, per-expert output scale

`Gemma4TopKRouter` is not the Mixtral/Qwen router `routeExperts` implements:

```python
hidden_states = self.norm(hidden_states)                        # ⚠️ no router.norm.weight
hidden_states = hidden_states * self.scale * self.scalar_root_size   #    tensor exists →
expert_scores = self.proj(hidden_states)                        #    weightless RMSNorm?
router_probabilities = softmax(expert_scores, dim=-1)
top_k_weights, top_k_index = topk(router_probabilities, k=top_k_experts)
top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)         # ALWAYS renormalized
top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
```

Three deltas against `MoEConfig`:

- **`NormTopKProb` must be `true`** even though the key is absent from `config.json` — the
  renorm is unconditional in the class. This is the exact trap `docs/qwen3_5_moe.md`
  documents ("the difference between cosine 0.9985 and 1.0"). Do not infer it from config.
- **Router pre-norm + learned `scale`** (and a `scalar_root_size` constant) applied to the
  hidden state *before* the projection. `routeExperts` takes logits, so this belongs in the
  caller. ⚠️ There is no `router.norm.weight` in the index — confirm `self.norm` is
  weightless before deciding whether it needs a `LayerWeights` slot.
- **`per_expert_scale[idx]`** — a learned per-expert multiplier on the selected weights,
  distinct from `RoutedScale` (which is a single scalar). New `MoEConfig` field +
  `LayerWeights` vector, and a new argument to `routeExperts`.

`router.proj.weight` is f32 in the checkpoint; keep it full-precision as the existing MoE
paths do (selection stability).

## A3. Loader + tensor schema

- Expert prefix is **`experts.`**, a sibling of `mlp.`, not `mlp.experts.` as in gpt-oss —
  so `loadFusedExperts` is reusable but the names are new. ⚠️ Confirm the split: HF does
  `.chunk(2, dim=-1)` on the linear output, i.e. contiguous halves of the `2·704` output
  dim (`gate = [:704]`, `up = [704:]`). Verify `loadFusedExperts` splits contiguously and
  not interleaved before reusing it.
- Shapes: `experts.gate_up_proj` `[128, 1408, 2816]`, `experts.down_proj` `[128, 2816, 704]`.
- `registry.go:212` still returns `&gemma3TensorSchema` with the comment *"a dedicated
  gemma4 tensor schema lands with the loader work"* — **this is that work.** A
  `gemma4TensorSchema` is needed regardless, because the layer prefix is
  `model.language_model.layers.{i}.` (not `model.layers.{i}.`) and the vision tower must be
  skipped.
- **Text-only:** the checkpoint is `Gemma4ForConditionalGeneration` and carries a
  `gemma4_vision` tower (~550M, 27 layers, `pooling_kernel_size 3`, `standardize`,
  clipped linears — a *different* tower from the Gemma 3 SigLIP one already in the repo).
  The loader ignores `model.vision_tower.*` and `model.embed_vision.*`. Vision is out of
  scope; the capability matrix must say `text` for this variant, not inherit Gemma 3 VL.
- Layer 5 (and every global layer) has **no `v_proj.weight`** — `attention_k_eq_v` means
  K is reused as V. The descriptor already models this (`KVShared`); the schema must not
  require the tensor.
- GGUF: defer. `decoder/gguf.go:284–305` hard-codes the dense gemma4 metadata backfill
  (including `PartialRotaryFactor: 0.25`, "not in GGUF metadata"). A MoE GGUF convert for
  this checkpoint may not exist yet — check before committing to it. Safetensors →
  `cmd/prequant` → `.giw` is the path that matters for the benchmark anyway, and it is what
  Part B's pagers require (they only manage mapping-backed weights).

## A4. Config parse + validator

- **`top_k_experts` is not a field goinfer parses.** `Config.NumExpertsPerTok` is tagged
  `json:"num_experts_per_tok"` (`config.go:46`); `NumExperts` at `:45` is already correct.
  Add the alias, don't rename.
- `enable_moe_block` (bool) gates dense vs MoE within the same `model_type` — so
  `gemma4Architecture` must branch on it, and `MoE` stays `nil` for E2B/E4B/12B. **The
  existing three dense variants must stay bit-identical**; that is a regression gate, not
  a hope.
- `intermediate_size 2112` is the *dense* branch width and must not be confused with
  `moe_intermediate_size 704`. (`2112 = 3 × 704` — a coincidence worth not building on.)
- A `validateGemma4MoE` in the style of `config.go:341–441`: experts > 0, `1 ≤ top_k ≤ E`,
  `moe_intermediate_size > 0`, and `enable_moe_block` ⇒ router tensors present.

## A5. Residency: stays declined, honestly

`decoder/residency.go:130` declines residency for `a.gemma4 != nil` unconditionally (own
non-uniform forward), which is why `docs/hardware-matrix.md` shows Gemma 4 as `CPU` on
WebGPU, CUDA *and* Metal. **Nothing here changes that** — the MoE variant lands on the
pure-Go CPU path, and the hardware-matrix row must regenerate to `CPU` across the board
rather than silently inheriting a `✅ resident` from the MoE feature flag. Lifting it is
`docs/c8-softcap-residency.md` (gemma4 own-forward) + `docs/task-metal-moe.md` (Metal MoE),
a separate track.

---

# Part B — weight streaming

goinfer already ships bigger-than-RAM streaming (`serve --stream-weights` +
`--weight-cache`, ideas #2 and #4 in `docs/ideas-weight-memory.md`, 2026-06-13). Part B is
**not** "build streaming." It is the four levers fieldfare has that goinfer does not, found
by reading its design against ours, plus one hazard that lands on the next aikit bump.

**The headline finding is platform-specific and it is the reason this is on the critical
path.** On darwin, `mmap.Advise(span, false)` is a **deliberate no-op** — macOS will not
force a resident drop on a read-only file-backed mapping (`aikit/mmap/madvise_darwin.go`,
verified empirically on darwin/arm64: `MADV_DONTNEED` and `MADV_FREE` leave RSS unchanged,
`MADV_FREE_REUSABLE` returns `EPERM`, the `msync` variants are no-ops). So on the Mac,
`--weight-cache` is **bookkeeping, not a cap**: `SpanCache` counts bytes and picks victims,
but the actual replacement policy is macOS's Unified Buffer Cache under memory pressure.
The firm cap is Linux-only.

That is precisely what fieldfare avoided by not using mmap at all. It `pread`s into buffers
it owns and runs its own 16-slot LFU per layer, so on an 8 GB M2 Air it gets a *predictable*
5.1–6.3 tok/s. goinfer on the same machine is at the mercy of the UBC — and the benchmark
rig for Part A is an M1 Pro 16 GB.

## What already ships (the substrate — don't rebuild)

| | |
|---|---|
| `expertPager` — router top-k drives eviction over the read-only `.giw` mapping | `decoder/moepaging.go` |
| `layerPager` — dense windowed prefetch, `WILLNEED` L+1 while computing L | `decoder/layerpaging.go` |
| generic span residency + budget | `aikit/mmap.SpanCache`, `mmap.Advise`, `mmap.AutoBudget` |
| bit-exactness by read-only re-fault | `TestMadvise_dontneedRefaultsIntact`, `TestExpertPaging_bitExact`, `TestLayerPaging_bitExact` |
| the demand-signal seam | `moeMLP` top-k → `pager.touch` (`decoder/mlp.go:81`) |
| an access trace + cost model for the real 35B-A3B | `decoder/moepaging_spike_test.go` |

Validated 2026-06-13 on the real Qwen3.6-35B-A3B: 512 MB expert cache against ~16 GB of
experts, `hits=4706 misses=5534 evictions=5190`, decode **byte-identical** to fully
resident over 24 tokens.

## Reference: what fieldfare does

Resident: a **1.35 GB shared core** + FP16 KV. On disk: **14.3 GB** in a custom `.gturbo`
file. Per layer — Metal computes attention and the router from resident weights; the CPU
takes the top-8 expert IDs, plans against a **16-slot LFU cache for that layer**, and fills
misses with **bounded parallel `pread`** into Metal-visible buffers; Metal computes the
**resident shared-expert branch while those reads are in flight**, then combines. Prefill
runs in chunks of ≤128 tokens so one fetched expert serves multiple rows.

Two arithmetic checks say the designs converge and the target is realistic. Gemma 4
26B-A4B's non-expert weights are ~2.4B params ≈ **1.3 GB at int4** — fieldfare's "1.35 GB
core", independently derived. And one expert is `2112 × 2816` = 5.95M params ≈ **3.0 MB at
int4**, so top-8 × 30 layers = **~714 MB/token if every fetch is cold** — at ~3 GB/s that
is ~4 tok/s, right at their 8 GB M2 figure. Nothing exotic is happening; the gap is
engineering, not insight.

## B0. The aikit bump hazard (independent of everything else — do it first)

goinfer pins `github.com/townsendmerino/aikit v1.12.0` (`go.mod:6`). aikit `main` is past
v1.14.0 and contains **`6c0483f` "mmap: scan-resistant SpanCache eviction"**, which changes
`SpanCache.Touch` to evict `c.lru.Front()` — the **most**-recently-touched other member —
instead of `c.lru.Back()`. It is the only `mmap/` change since v1.12.0.

That change is correct for the workload it was measured on: aikit's own demand signal is
`FlatI8`'s paged query walking blocks 0,1,2,… every call, the textbook cyclic-scan
pathology where LRU hits **0% even at a 63/64 budget**. Evicting the most-recent pins a
stable prefix and recovers 10.9 / 45.0 / 88.6% at 8/32/63-of-64.

**It is the wrong policy for `expertPager`, whose demand signal is the opposite shape.**
The spike measured skewed *frequency*, not a cycle: the hottest 10% of experts absorb 72%
of accesses, the hottest 25% absorb 94%, and half the universe is never touched. Under
evict-most-recent, a hot expert becomes the victim as soon as anything else is touched
after it; the policy pins the *oldest* prefix rather than the hot set. Nobody will notice
at load time — it shows up as a hit-rate regression on the next bump.

Work:

1. Re-run `decoder/moepaging_spike_test.go`'s trace against both policies and record the
   hit-rate delta. Offline replay over a recorded trace — no I/O, no model load.
2. If the regression is real (expect it to be), the fix is **policy per demand signal**, not
   one global policy in the shared type: a `SpanCache` policy knob (or a
   `NewSpanCacheWithPolicy`) so the ANN scan keeps scan-resistance and the expert pager
   keeps a frequency-aware policy. That is an aikit change; land it there and bump.
3. Stale artifacts to fix in the same pass: `aikit/mmap/spancache.go`'s type doc still says
   Touch "releases the least-recently-touched members"; `decoder/moepaging.go:15–16`
   describes "the generic span-residency LRU … release the budget tail";
   `decoder/moepaging_test.go:11` cites `TestSpanCache_evictsLRUTailOverBudget`, renamed in
   aikit to `TestSpanCache_evictsMostRecentOverBudget`.

**Gate:** the bump does not land until the expert-pager hit rate on the replayed trace is
≥ its v1.12.0 value.

## B1. An owned-buffer `pread` miss path (the big one)

**Today.** A miss is `Advise(span, WILLNEED)` followed by a synchronous page fault when the
matmul reads the span. That is one fault at a time on the decode thread — queue depth 1 —
and the spike's cost model (`NVMe ≈ 20 µs seek + 3 GB/s`) assumed the *bandwidth*, which a
QD1 fault stream is unlikely to reach on any NVMe device. On darwin it is worse than a
throughput question: with eviction a no-op, there is no cap to enforce at all.

**Two steps, in order.**

**B1a — concurrency without changing the data path.** Keep the mapping and the zero-copy
`WeightMat` aliasing exactly as they are; add a small worker pool that faults the missing
spans *concurrently* ahead of the matmul (`Advise WILLNEED` plus an explicit touch-a-byte-
per-page read, since `WILLNEED` is only a hint). Every existing bit-exactness guarantee
holds unchanged — the bytes still come from the same read-only mapping. This is the cheap
half and it should be measured on its own: **if queue depth is the whole story, stop here.**

**B1b — owned buffers.** If B1a does not close the gap, or the darwin cap matters more than
zero-copy, move the expert path off mmap: `pread` into pooled, page-aligned buffers the
pager owns, bounded-parallel, with the pager's own residency accounting. This buys a firm
RAM cap on macOS — the thing `madvise_darwin.go` says cannot be had any other way — at the
cost of the `.giw` zero-copy aliasing for expert tensors and a real memcpy per fill.

B1b is a genuine architectural change, so gate it on B1a's measurement rather than
assuming. Bit-exactness stays trivially provable either way (the bytes are the same file
bytes), and `TestExpertPaging_bitExact` is the existing gate.

**Deliverable:** a tok/s-and-p99-latency-vs-budget curve on both darwin and linux, for
QD1-mmap / parallel-mmap / parallel-pread. The darwin column is what decides B1b.

## B2. Eviction policy matched to the demand signal

B0 forces this question; this is where it gets answered properly. fieldfare picked **LFU**;
goinfer has LRU (soon MRU-ish). The spike's own skew numbers — 10% of experts absorbing 72%
of accesses, stable across tokens — are an LFU distribution, not an LRU one.

The whole experiment is offline: `moepaging_spike_test.go` already records the access
trace, so LRU / scan-resistant / LFU / LFU-with-aging can be replayed over it and scored on
hit rate at 3, 8 and 16 GB budgets without touching a disk or loading a model. Do that
before writing any policy code, and publish the table the way the spike published its
original one (3 GB → 51.3% hit / +166.4 ms per token; 8 GB → 89.6% / +35.5 ms; 16 GB →
92.9% / +24.3 ms).

Note the interaction with B1: on darwin today the policy barely matters, because the real
replacement decision belongs to the UBC. Policy work only fully pays off once there is a
firm cap (B1b) — but the replay is cheap enough to do first regardless, and it is what
tells you whether B1b is worth it.

## B3. Overlap routed reads with the resident branch

`docs/ideas-weight-memory.md` states the cold-miss cost is "bandwidth-bound and
unhideable … the router selects just-in-time, so there's no prefetch lead." True for the
architectures goinfer had. fieldfare found the hiding place: run the **resident** branch
while the routed reads are in flight.

**Part A hands goinfer the same opportunity for free.** Gemma 4 26B-A4B's FFN sub-block is
a *parallel* dense-MLP + MoE pair (§A1): the dense branch is always resident, independent
of the routed branch, and its output is simply summed. So the sequence becomes — route →
issue the expert fills (B1's pool) → **compute the dense branch** → join → combine. On a
30-layer model that is 30 overlap windows per token.

This lever is therefore **sequenced after Phase 2** (the forward), and it is the one that
makes the "unhideable" line in `ideas-weight-memory.md` need an amendment. Generalizes to
any arch with an always-on shared expert (Qwen2-MoE, GLM, DeepSeek), where the shared
expert plays the dense branch's role.

**Measure the right thing:** the win is p99 token latency and the miss-cost coefficient,
not mean tok/s at a budget where everything hits anyway. Run it at a budget that forces a
real miss rate.

## B4. Expert-major prefill batching

`decoder/forwardn.go:14` states it plainly: batched prefill vectorizes attention but "the
MoE FFN itself stays per-row (router picks different experts per token)" — the per-row call
is `forwardn.go:228`. Under streaming that is the worst case: the same expert can be
fetched once per row.

fieldfare's fix is chunks of ≤128 tokens so one fetched expert serves multiple rows.
The goinfer version: within a prefill chunk, **group rows by selected expert** and run each
fetched expert once over all its rows (a gather → per-expert GEMM → scatter), so fetch
count falls from `rows × k` toward `distinct experts in the chunk`. This is a throughput
win even fully resident (it turns k GEMVs per row into one GEMM per expert), which is what
makes it worth doing independently of the streaming case — and it is the missing half of
the 2.4× batched-MoE-prefill result on Mellum2 (`CHANGELOG` v0.5.0, `08acc11`).

`ideas-weight-memory.md` #4 hedges that streamed mode may need to "cap the streamed mode to
modest prompts." This is the alternative to that cap.

---

# Rig, memory budget, and the number to expect

Target rig: **M1 Pro 16 GB** — same platform class as turbo-fieldfare. Secondary: the
**Ryzen 7 3700X / RTX 2070 SUPER** box for the linux firm-cap column.

| | |
|---|---|
| total params | 25.2B → int4 ≈ **~13.9 GB**, int8 ≈ 25 GB (**does not fit**) |
| active params/token | 3.8B → int4 ≈ **~2.1 GB read/token** |
| of which expert weights | top-8 × 3.0 MB × 30 layers ≈ **714 MB/token** (the streamable part) |
| non-expert resident floor | ~2.4B params ≈ **~1.3 GB at int4** |
| measured CPU effective bandwidth (M1 Pro) | **~34 GB/s** (`docs/completed/perf-campaign.md` roofline) |
| bandwidth-roofline decode | **~16 tok/s** — a ceiling, not a forecast |

Two things pull the real number below that ceiling: top-8-of-128 expert gather is random
access that defeats prefetch (unlike the dense streams the 34 GB/s was measured on), and
the LM head reads the full 262144 × 2816 tied matrix every token (~369 MB at int4, ~18% of
the per-token traffic on its own). **Expect single digits from Part A alone — call it
6–12 tok/s.**

**The honest framing of the comparison:** 13.9 GB of weights on a 16 GB machine means the
page cache is the binding constraint, so an M1 Pro 16 GB lands in turbo-fieldfare's *8 GB
M2 regime* (paging-bound, 5–6 tok/s), not its *24 GB M5 Pro regime* (31–35 tok/s, resident).
Beating the M5 Pro figure needs Metal residency, which this task explicitly does not
deliver. Part B is what makes the paging-bound regime *predictable and controlled* rather
than UBC-dependent; it is not a route to the resident number. Write the benchmark row that
way or it reads as a loss that isn't one.

`.giw` prequant matters more here than usual: `docs/mellum2-resident.md` measured
bf16-safetensors → int4 load at ~66 s for a 12B; a 25B will be worse, and `cmd/prequant`
already accepts a safetensors directory. It is also a hard requirement for Part B — both
pagers only manage weights that alias the mapping.

---

# Plan

Parity-first, matching the Gemma 4 dense and `qwen3_5_moe` bring-ups.

## Status (2026-07-31)

Phases 1a–2b landed and pushed; the whole-model forward matches the HF oracle
bit-for-bit on the tiny checkpoint (cosine 1.0, byte-identical greedy continuation).
K=V global layers and expert quantize-at-load followed. **The real 26B-A4B has NOT
been loaded end-to-end** (the Phase 4 headline gate) — the checkpoint is a ~50 GB
download, infeasible on the dev box; the cached dense gemma-4-12B config was used to
confirm the real structural facts instead.

| Phase | State | Where |
|---|---|---|
| 1a pin the math | ✅ | `cac4616` |
| 1b descriptor + config | ✅ | `6eea5fa` (dense E2B/E4B/12B bit-identical) |
| 1c tiny-random HF golden | ✅ | `d1b2189` |
| 2a FFN sub-block op-golden | ✅ | `f7515eb` (cosine 1.0, maxAbs 8e-7) |
| 2b wire into `runLayersGemma4` | ✅ | `51ea350` (whole-model cosine 1.0) |
| 3 loader + schema — **TINY ONLY** | ◐ | `51ea350` + `9c04ea3`; tiny checkpoint validated (per-layer attn widths, `layer_scalar`, streamed+quantized experts). **Real 26B not loaded**; the `model.language_model.*` prefix + vision skip are unwritten. |
| 4a K=V global layers | ✅ | `9e83043` (`attention_k_eq_v`/VFromK + `num_global_key_value_heads`, cosine 1.0) |
| 4 real-checkpoint parity | ☐ | blocked on the ~50 GB download |
| 5–8 (bench, streaming I/O, overlap, expert-major) | ☐ | not started |

**Follow-up review (post-`51ea350`), one commit each:**
- RMSAddOne consistency in `gemma4MoEFFN` — `97d2a2c`
- capability matrix reports the MoE variant honestly (`dense ‖ sparse`) — `a84fb76`
- experts quantize AT LOAD, streamed per-expert via aikit v1.15.0 `embed.Tensor.SubF32` (91 GB f32 → ~23 GB int4, no whole-tensor transient) — `9c04ea3`
- zero-alloc FFN — **DECLINED**: `runLayersGemma4` is parity-first / allocating by design (its own forward, not the generic `decodeScratch` zero-alloc path), so optimizing only the FFN sub-block is inconsistent and near-zero benefit; a whole-forward scratch pass is a separate perf task.

**Two findings the pin corrected — record them for the next Gemma variant:**

1. **Norm wiring (Phase 1a).** The MoE branch AND the router each read the RAW
   post-attention residual `h` through their OWN normalization — the router via its
   weightless norm, the experts via `pre_feedforward_layernorm_2` — NOT the dense
   branch's `pre_feedforward_layernorm`. Dense and MoE run in parallel on `h`, then
   the sum is joint-normed (`post_feedforward_layernorm`), residual-added, × `layer_scalar`.

2. **Global-layer proportional RoPE (found in Phase 2b, confirmed against the real
   12B config).** Gemma 4's global (full-attention) layers use `"proportional"` RoPE
   whose `partial_rotary_factor` (0.25) lives NESTED in
   `rope_parameters.full_attention` — NOT the top-level `partial_rotary_factor` (Phi's
   spelling, absent here). Reading the top-level value gave `GlobalRotaryDim=0` → the
   global layer ran as NoPE → a large mid-stack divergence the final RMSNorm nearly
   masked at the logits (whole-model cosine still ~0.99 while a per-layer trace showed
   the global layer's norm at 13.7 vs 8.3). `gemma4PartialRotary()` prefers the
   top-level value (GGUF injects it there, so the dense GGUF path stays byte-identical)
   and falls back to the nested one.

---

**Phase 0 — aikit bump hazard (B0).** Independent of the rest; do it now. Replay the spike
trace under both eviction policies, fix the policy split in aikit if it regresses, clear the
three stale doc references. Gate: expert-pager hit rate ≥ v1.12.0's before the bump lands.

**Phase 1a — pin the math.** Local `transformers` checkout at a pinned version: read
`Gemma4TopKRouter`, `Gemma4Experts`, `Gemma4MLP`, `Gemma4DecoderLayer.forward` in full.
Resolve every ⚠️ above: branch wiring and norm order, `layer_scalar` placement relative to
the residual, whether `router.norm` is weightless, `scalar_root_size`'s value, the
`gate_up` split order, and the activation on the expert branch. Update this doc with line
refs. **No code before this is done** — the norm-order and scalar-placement risks are
exactly the ones a cosine-only gate hides.

**Phase 1b — descriptor + config (A2, A4).** `top_k_experts` alias, `enable_moe_block`
branch in `gemma4Architecture`, `MoEConfig` extended with `PerExpertScale` + router
pre-norm/scale, `validateGemma4MoE`. Regression gate: `gemma4_test.go`,
`gemma4_parity_test.go`, `gemma4_12b_parity_test.go`, `gemma4_12btrace_test.go` all still
green and bit-identical.

**Phase 1c — tiny-random HF golden.** Extend `scripts/pin_gemma4_forward.py` (the dense
pinner already in the repo, alongside `pin_gemma4_12b.py`) with an `enable_moe_block`
variant, or fork it to `pin_gemma4_moe_forward.py`: a tiny `gemma4`-MoE checkpoint
(2 layers — one sliding, one global; 8 experts, top-2) →
`testdata/gemma4_moe_forward_golden.json`, plus a decode golden. Same shape as
`scripts/pin_qwen35_forward.py`.

**Phase 2 — forward (A1).** The parallel dense+MoE sub-block, the extended router,
`layer_scalar`. Gated to **argmax-exact + logit cosine** on the tiny golden, with a
**per-layer sub-layer trace** (`gemma_sublayer_trace_test.go` is the pattern) so a
norm-order or scalar-placement error surfaces at the layer it happens rather than as a soft
end-of-stack drift.

**Phase 3 — loader + schema (A3).** `gemma4TensorSchema` (the `model.language_model.layers.`
prefix, fused `experts.*`, no `v_proj` on global layers, vision skipped), reusing
`loadFusedExperts`. Gate: load the real 26B-A4B, assert shapes and tensor coverage (nothing
silently unconsumed). **PARTIAL (tiny-model only, `51ea350`+`9c04ea3`):** the gemma4
safetensors branch loads the tiny checkpoint — per-layer attention widths (global
`global_head_dim`), `layer_scalar`, K=V globals (`4a`), and experts streamed one at a time
via `embed.Tensor.SubF32` + quantized at load (int8/int4). Still unwritten for the real 26B:
the `model.language_model.*` prefix and vision-tower skip, plus the real-checkpoint gate
itself (Phase 4). The `experts.gate_up` split is contiguous `chunk(2)` per Phase 1a, so no
transpose is needed; `streamExperts` is the fused-expert reader (the `loadFusedExperts`
analogue that avoids the whole-tensor f32 materialization).

**Phase 4 — real-checkpoint parity.** Full-model int8/int4 parity vs the HF bf16 oracle on
the real checkpoint, per `docs/parity-coverage-policy.md`. Mellum2-style: argmax-exact on
the first N tokens + sampled logit cosine. **Slow** — a 25B CPU forward against a torch
oracle; budget for it (`TestMellum2_windowParity` took 534 s on a 12B).

**Phase 5 — measure and publish the CPU number.** `cmd/prequant -quant int4` → `.giw`, then
`scripts/bench_compare.sh` on the M1 Pro. Add the row to `docs/benchmarks.md` §A with commit
+ date + thermal note, and regenerate `docs/hardware-matrix.md` + `docs/capability-matrix.md`
(`go test ./decoder -run HardwareMatrix -update`). State the memory-regime caveat above in
the row, not just the tok/s. **This is the row that makes the fieldfare comparison legal.**

**Phase 6 — streaming I/O (B1, B2).** B1a first (concurrent faults, same data path);
measure; B1b (owned buffers, `pread`, firm darwin cap) only if B1a doesn't close it. B2's
policy replay can run in parallel with B1a since it needs no I/O.

**Phase 7 — overlap (B3).** Needs Phase 2. Route → issue fills → compute the dense branch →
join. Measure p99 token latency at a budget that forces real misses.

**Phase 8 — expert-major prefill (B4).** Group rows by expert within a prefill chunk. Worth
doing fully-resident too, so it can be justified on its own.

## Measurement and gates (both parts)

- **Rigs:** M1 Pro 16 GB (darwin, no firm cap — the fieldfare-comparable rig) *and* the
  Ryzen 7 3700X box (linux, firm cap). Report them separately; a single number across both
  hides the whole finding.
- **Bit-exactness is non-negotiable and already gated:** `TestExpertPaging_bitExact`,
  `TestLayerPaging_bitExact`, `TestMadvise_dontneedRefaultsIntact`. Any lever that cannot
  keep decode byte-identical to fully-resident is rejected, not caveated.
- **Publish curves, not points.** tok/s and p99 token latency vs `--weight-cache`, per
  lever, per platform — the shape the spike's original table had. Per `docs/benchmarks.md`
  rules: commit + date + machine + thermal note inline.
- **Peer figures are unpinned.** turbo-fieldfare's numbers come from its README with no
  version pinned. Re-read and pin before any published comparison row.

## Explicitly out of scope

- The `gemma4_vision` tower (text-only; see §A3).
- Metal / CUDA / WebGPU residency for Gemma 4 (`c8-softcap-residency.md`,
  `task-metal-moe.md`) — and therefore any route to fieldfare's M5 Pro number.
- Streaming into GPU-visible buffers (fieldfare `pread`s straight into Metal-visible
  memory). Gemma 4 declines residency on every goinfer backend (`residency.go:130`), so
  there is no GPU path to feed on this model.
- A MoE GGUF path, and a custom on-disk container. `.giw` + mmap stays the substrate; B1b
  changes how expert bytes are *read*, not the format.
- Cross-token expert prediction (prefetching next token's experts from this token's
  selection). Speculative, harmless when wrong, plausibly free once B1's pool exists — but
  a separate measurement, not a bundled assumption.

## Sources

HF `google/gemma-4-26B-A4B-it` — `config.json` (all config values above),
`model.safetensors.index.json` (tensor names, verbatim; layer 0 vs layer 5 key diff).
`transformers` `models/gemma4/modeling_gemma4.py` — router and experts code (partially
elided in the fetched copy; ⚠️ items are what Phase 1a must resolve).

In-repo: `decoder/registry.go`, `decoder/config.go`, `decoder/arch.go`, `decoder/mlp.go`,
`decoder/weights.go`, `decoder/gguf.go`, `decoder/residency.go`, `decoder/forward_gemma4.go`,
`decoder/forwardn.go`, `decoder/moepaging.go`, `decoder/layerpaging.go`,
`decoder/moepaging_spike_test.go`, `go.mod:6`, `docs/ideas-weight-memory.md` §2 and §4,
`docs/benchmarks.md`, `docs/hardware-matrix.md`, `docs/qwen3_5_moe.md` (plan template),
`docs/mellum2-resident.md`, `docs/completed/perf-campaign.md`, `docs/c8-softcap-residency.md`,
`docs/task-metal-moe.md`, `docs/parity-coverage-policy.md`.

aikit: `mmap/spancache.go`, `mmap/madvise_darwin.go` (the darwin no-op eviction),
`mmap/madvise_linux.go`, commit `6c0483f` + `docs/internal/perf-campaign-2026-07-28.md`
item 9 (the scan-resistance measurement), tags through v1.14.0.

Peer: `github.com/drumih/turbo-fieldfare` README — Gemma 4 26B-A4B, MLX affine 4-bit group
64, 8-bit router; 1.35 GB resident core, 14.3 GB `.gturbo` on disk, 16-slot LFU per layer,
bounded parallel `pread` into Metal-visible buffers, shared-branch/read overlap, ≤128-token
prefill chunks; 5.1–6.3 tok/s on an 8 GB M2 Air and 31–35 tok/s on a 24 GB M5 Pro.
**Version not pinned** — re-read and pin before publishing a comparison row, per
`docs/benchmarks.md`'s methodology rule.
