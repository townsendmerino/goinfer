# Task: Gemma 4 26B-A4B — MoE bring-up + weight streaming

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


> **⚠ Peer numbers below predate the Ollama v0.32.5 re-anchor (2026-08-04).** Competitive figures
> in this doc (e.g. Ollama-CUDA ~149, Ollama-Metal 83.3, llama.cpp-CUDA 72.8, and any "×Ollama"
> multiple) were measured against **Ollama 0.5.7 (2025-01) / Ollama-Metal 0.32.0 / llama.cpp as of
> v0.5.0** — historical working records, not current claims. Current same-box numbers vs Ollama
> **v0.32.5** are in `docs/benchmarks.md` §B2 (CUDA) / §B3 (Metal).


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

**RESOLVED (2026-07-31).** The hazard was real and, briefly, shipped: the aikit
v1.12→v1.15 bump (needed for `embed.Tensor.SubF32`) carried `6c0483f`, so the expert
pager silently ran the scan-resistant evict-most-recent policy. Measured on the real
35B-A3B router trace (`moepaging_spike_test.go`, now a two-policy comparison), it lost
up to **51 pp** of hit rate vs the LRU tail at interactive budgets (4 GB: 71.9% →
20.9%); the production gate at a 512 MB budget dropped from 45.9% to 11.1%.
Fix = the policy knob below (`mmap.NewSpanCacheWithPolicy` / `EvictPolicy`, aikit
v1.16.0): the ANN paths keep the scan-resistant default, `expertPager` passes
`EvictLeastRecent`, restoring the v1.12 baseline (production gate back to 45.9%,
`hits=4696` ≈ the doc's recorded `4706`). Correctness was never affected (paged decode
byte-identical throughout). The three stale doc references below are fixed. What
follows is the original problem statement.

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

Phases 1a–4 landed. The whole-model forward matches the HF oracle bit-for-bit on the
tiny checkpoint (cosine 1.0), and **the real gemma-4-26b-a4b-it now loads and generates
coherent English at int8** — the loader needed no changes (the unified
`model.language_model.*` prefix auto-detects, `text_config` flattens, vision skipped),
plus a nested-RoPE-base fix.

| Phase | State | Where |
|---|---|---|
| 1a pin the math | ✅ | `cac4616` |
| 1b descriptor + config | ✅ | `6eea5fa` (dense E2B/E4B/12B bit-identical) |
| 1c tiny-random HF golden | ✅ | `d1b2189` |
| 2a FFN sub-block op-golden | ✅ | `f7515eb` (cosine 1.0, maxAbs 8e-7) |
| 2b wire into `runLayersGemma4` | ✅ | `51ea350` (whole-model cosine 1.0) |
| 3 loader + schema | ✅ | `51ea350`+`9c04ea3`+`54d3096`; unified prefix auto-detect + `text_config` flatten + vision skip, proven on a synthetic unified checkpoint — real 26B loads with no loader change |
| 4a K=V global layers | ✅ | `9e83043` (`attention_k_eq_v`/VFromK + `num_global_key_value_heads`, cosine 1.0) |
| 4 real-checkpoint | ✅ | `681db0c`+`625303e`+`78478a8`: real 26B loads (30 layers, 128 experts top-8, K=V globals) + coherent English **at int4 AND int8** under the chat template (`TestGemma4_26B_gate`, both precisions, distinct-trigram 0.841/0.833). 625303e's "gate runs int8 because int4 is too aggressive" rationale is RETRACTED — see below. |
| 5 benchmark | 🟢 | **UNBLOCKED** — the "int4 incoherence" was a PROMPTING artifact, not a quantizer deficit. Real int4 (sym W4A8, ~13 GB) is fully coherent under the Gemma-4 chat template, greedy. **Caveat for the benchmark row:** ~13 GB on a 16 GB M1 Pro is still paging-bound (UBC-dependent), so the number is not a clean resident comparison against the peer's resident figures — say so explicitly, per `docs/benchmarks.md`. See the RESOLUTION at the top of the int4 finding. |
| 6–8 (streaming I/O, overlap, expert-major) | ☐ | not started |

> **RESOLUTION (2026-08-01) — it was the PROMPT, not the quantizer. Phase 5 unblocked.**
> `TestGemma4CoherenceProbe` (realckpt) ran {raw, templated} × {greedy, sampled} on the real 26B:
> - **Real shipping int4 (sym W4A8), raw prompt** → garbage (`"than than … 얓숌면"`), greedy AND
>   sampled. **Same int4, proper Gemma-4 chat template** (`chat.Gemma4().RenderSegments` +
>   `tk.EncodeSegments`) → **fully coherent, greedy**: *"The capital of France is **Paris**. …the
>   **Eiffel Tower**, the **Louvre Museum**, the **Notre-Dame Cathedral**, and the **Arc de
>   Triomphe**…"*. Sampling did NOT fix the raw prompt; the template did.
> - **int8 is ALSO degraded on the raw prompt** — `"water-water-water is 100°C … Earth-related crops
>   are Earth-related crops"` — repetitive English nonsense. It only ever *looked* coherent because
>   `TestGemma4_26B_gate`'s coherence check is printable-ASCII-majority: int8's off-distribution
>   garbage is English-ish (passes), int4's is CJK (fails). **Neither precision is coherent on the
>   raw completion prompt; both are fully coherent templated.**
> - **Consequence:** the entire matrix/probe arc below was diagnosing a non-problem. A raw completion
>   prompt on an **instruction-tuned** model is off-distribution; int8 had marginally more headroom to
>   emit ASCII, int4 fell to CJK — and a weak coherence gate turned that into a false "int4 is broken"
>   signal. No affine build, no `.giw` v5, no numerics hunt, no Mac trip needed. int4 at ~13 GB fits
>   the 16 GB target and is coherent for real (templated) use. **Lesson: judge quantization COHERENCE
>   only through the model's real chat template; the greedy+raw gate is for bit-exact NUMERICS.**
> - **RETRACTION of `625303e`.** That commit asserted, as a finding: *"int4 is too aggressive for
>   this model … the 128-expert top-8 MoE compounds int4's per-expert error (cosine ~0.988 on the
>   tiny) across 30 layers into garbage … so the real-model gate runs int8."* **This is false and is
>   the origin of the entire arc.** The "garbage" was the raw prompt, not int4; real int4 is fully
>   coherent templated (0.841 distinct-trigram, `78478a8`), and int8 is *equally* garbage on the raw
>   prompt. The per-expert-error-compounding mechanism never applied. (Same treatment `bcadd44`
>   received when its "deficit is the weights" note was corrected.)
> - **Follow-ups:** (a) ✅ DONE (`78478a8`) — `TestGemma4_26B_gate` now renders the chat template and
>   asserts int4 coherence via distinct-trigram, with a raw-prompt mutation control. (b) run Phase 5
>   on templated prompts; (c) the probe findings still stand as corroboration — routing healthy
>   (probe 1), peer naive-affine (probe 3), precision not the variable (probe 2) — all consistent with
>   "not a quantization problem."

**Decode perf — Step 0 profile (int4, real gemma4-26b .giw, Ryzen 7 3700X, GOMAXPROCS=8).** The
initial ~2.3 tok/s was NOT bandwidth-, GC-, or barrier-bound. Three instruments:
- **GOMAXPROCS sweep** 1/2/4/8 → 1.48 / 2.03 / 2.12 / 2.39 tok/s = **1.61× on 8 cores**, flat after 2.
- **pprof**: 87 % of samples in two dot kernels — `q8Span` **49 %** (the int8 LM head, the only int8
  matmul in an int4 model) + `dotW4A8FoldAVX2` **42 %** (all int4 matmuls) — but only **187 % CPU on
  8 cores** (threads asleep; barrier waits burn no CPU samples).
- **gctrace**: **1 GC cycle / 24 tokens** → zero-alloc revisit is moot (Step 3 dead).

**Root cause:** `linalg.MatmulBTW4A8Into` has a serial fast-path `if M*N*K < ws.thr()`, and aikit's
default `parThreshold = 1<<24 = 16.78M` MACs. At decode (M=1) every int4 matmul is smaller —
expert gate‖up 3.96M, down 1.98M, dense ~5.9M, attention ~11.5M — so **all ~600/token ran SERIAL**;
only the int8 head (738M) parallelized. The 1.61× is pure Amdahl (42 % serial int4 + 49 % parallel
head), **not a barrier** (no significant sync/futex frames — so A3, Step 1, is not the cause; defer).

**Fix (`199d4da`, surgical goinfer-side, NOT a global aikit change):** `matmul()`'s int4 branch runs
a per-call `Workspace` with `SetThreshold(int4ParThreshold = 1<<20 ≈ 1.05M)`, below the 1.98M
smallest decode matmul. **Measured 2.30 → 5.53 tok/s (2.4×), TTFT 7.3 → 2.66 s**, byte-identical
(same token ids serial vs parallel; `TestGemma4MoEFFN_parity` + `TestExpertPaging_bitExact` green).

**Step 4 (int4 head) — DONE, quality-safe.** The int8 LM head was 49 % of the *pre-threshold-fix*
compute (Q8 int8-weight × f32-activation is ~3–4× slower per element than W4A8). The `-embed-int4`
restriction was **incidental** (safetensors loaders used `.embedding()` not `.embeddingWith(embedInt4)`;
prequant exposed no flag) — now lifted: `embedInt4` threads through `buildWeightsFromSafetensors` and
`cmd/prequant -embed-int4`. **Verified on the real 26B (int4 head vs int8-pinned): trigram 0.911 vs
0.900, "Paris" survives, coherent prose** — the int4 head does NOT break coherence on this big-vocab
model. Decode +6 % here (4.84→5.13 tok/s; modest because after the threshold fix the head is a smaller
slice) but it **halves the head's 738 MB/token traffic**, the win that matters on the paging-bound
16 GB target. Default stays int8-pinned (opt-in). Step 1 (A3 barrier) is not the cause (deferred);
Step 3 (zero-alloc) is dead (no GC).

**Step 2 (batch int4 experts) — GO, justified.** Post-threshold-fix GOMAXPROCS sweep on the real
gemma4-26b-int4 .giw: 1/2/4/8 → 1.49 / 2.40 / 4.23 / 4.71 tok/s = **3.17× on 8 cores** (up from the
pre-fix 1.61×) — still clearly **sub-linear** (~40 % parallel efficiency), and the 4→8 step flattens
to +11 %. (Variance pin: the 8-core absolute jitters **±7 % back-to-back** — 4.8/5.1/5.4/4.8 over 4
reps of one load, so the earlier 5.53 and this 4.71 are both edges of one band, thermal/scheduler,
NOT a config difference — while 1-core is rock-stable at 1.51. Scaling stays **3.2–3.6× across the
whole band**, so the sub-linear verdict does not depend on the noisy absolute.) That flattening is the signature of the ~600 tiny int4 decode matmuls (each 2–12M MACs,
M=1) not filling 8 cores individually — spawn/join per matmul dominates past ~4 workers. Effective
bandwidth at 8 cores ≈ 10.5 GB/s, still ~⅓ of the ~34 GB/s this CPU path reaches on dense streams
(`perf-campaign.md`), so it is NOT approaching a memory ceiling. Batching the 8 experts' `gateUp`
(they share the activation `xe`) into one N=11264 op — and the dense `gate`+`up` (share `xd`) —
makes matmuls big enough to scale to 8 cores where the tiny ones don't. **Scope: a batched W4A8
kernel in aikit (`gemma4MoEFFN` + `moeMLP` expert loops are the call sites), with its own
bit-exactness gate; the `expertsDown` calls take distinct `mid` so they don't batch under a
shared-activation API — confirm rather than assume.**

**int4 quality finding — HISTORICAL (superseded by the RESOLUTION above; kept as the investigation
record). `scripts/gemma4_quant_recon.py` + `decoder/fakequant.go`.**
int8 (~26 GB) is coherent but doesn't fit the M1 Pro 16 GB target; **every 4-bit config tried is
incoherent** *[on the raw prompt — see RESOLUTION]*, so Phase 5 is blocked on output quality, not
throughput. This SUPERSEDES the earlier
"the deficit is the int4 WEIGHTS → implement affine int4" note (`bcadd44`), which reasoned from the
offline cosine table and never ran affine × f32-activations end-to-end.

**End-to-end matrix** (`fakeInt4WM`: reconstruct each 4-bit scheme, store int8, generate greedily
from "The capital of France is"; gemma-4-26b-a4b-it):

| weights ↓ / act → | **int8 act** (W4A8 — goinfer default) | **f32 act** (W4A16 — `MatmulBTQ4`) |
|---|---|---|
| **symmetric** (goinfer's int4) | garbage (`"than than … usual. usual."`) | garbage (CJK `"스트라이크 … 의심심한의"`) |
| **affine** (MLX's scheme) | garbage (`"de- fact- de- fact-"`) | **semi-coherent** (`"… the questioner questioner?"`) |

int8-everywhere is fully coherent (reference). **The fidelity gate:** the `sym` cell must reproduce
the real-int4 garbage, or the whole matrix is untrustworthy — asserted, not eyeballed, by
`TestFakeQuantSymMatchesRuntimeInt4` (`decoder/fakequant_test.go`), which proves the harness's `sym`
scheme is element-wise identical to aikit's runtime `QuantizeInt4`→dequant. Only then are the other
cells worth reading.

**Corrected conclusions:**
- **Neither lever alone recovers coherence** — it needs **affine weights AND f32 activations
  together**. sym+f32-act is CJK garbage; affine+int8-act is garbage. bcadd44 tested f32-act only
  with *sym* weights, so it missed the interaction.
- **Dropping affine int4 into the existing W4A8 (int8-activation) path would still be garbage.** The
  int4 dispatch must also move to the f32-activation `MatmulBTQ4` kernel — **but note `MatmulBTQ4` is
  documented as the *prefill* kernel (quantized weights × f32 activations); repurposing it for decode
  is correct but not its intended path, and the throughput cost lands squarely in Phase 5.**
- Even affine+f32-act (MLX's exact scheme) reaches only **semi-coherence** in goinfer (repetitive
  English: `"true or true or … or vice versa"`), **not** MLX's full coherence. A residual gap remains.

**Three probes before funding the affine + `.giw` v5 build — cheapest first.** The affine build is
multi-day (aikit affine quantizer + `.giw` v5 per-group zero-points + int4→Q4-f32act dispatch); do
NOT start it until these rule out that it's the wrong target:

1. **Expert-selection agreement** (hours, purely diagnostic — no new format). Per layer, measure
   top-8 overlap and selection entropy of the 4-bit run vs the int8 run. **Repetitive-English output
   is the signature of routing collapse, not of uniform weight noise.** If routing has degenerated,
   the residual gap is about router-*input* cleanliness, and more bits on the expert weights won't
   close it — which redirects the whole plan.
   **→ DONE — routing does NOT collapse; direction validated, not redirected.** Captured per-token/
   per-layer top-8 selections (`GOINFER_ROUTER_CAPTURE=1`, `TestGemma4RouterCapture`, 27-token fixed
   passage) for int8, real-int4 (shipping garbage), and affine+f32act (semi-coherent), vs int8:
     - **Selection entropy is preserved** — meanΔH(int8−q) ≈ **0.00 bits**; per-layer distinct-expert
       counts within ±2 (L0 53 vs 52–53, L29 58 vs 56–57). **No collapse to a few experts, any config.**
       So the repetition is NOT routing collapse — the "repetition ⇒ collapse" heuristic fails here.
     - Routing **drifts with depth but doesn't degenerate**: top-8 overlap with int8 falls 0.96 (L0) →
       ~0.68 (L29), mean **0.879 real-int4 / 0.894 affine+f32act** — uniform weight-noise nudging
       borderline picks, not collapse.
     - **Routing is not the discriminator**: garbage (0.879) and semi-coherent (0.894) route almost
       identically yet differ wildly in coherence. So the residual gap is in **expert *computation*
       fidelity (weight scheme × activation precision), not router-input cleanliness** — which
       **validates** the affine + f32-act lever rather than redirecting to a router fix.
2. **Re-test the mix under affine.** "int4mix won't help" was measured in the *symmetric* era and is
   **not transitive** — experts-4bit + everything-else-int8 under **affine + f32-act** is unmeasured.
   That config is ~14.5 GB (fits 16 GB), needs **no new format work**, and if all-4-bit affine+f32act
   is already semi-coherent the mix should be strictly better. This is the config that could actually
   unblock Phase 5.
3. **Check what fieldfare actually runs** (an afternoon; may explain everything). Does it quantize
   from bf16 itself with naive min/max affine g64, or consume a *pre-quantized* MLX checkpoint?
   mlx-lm supports mixed-bit recipes and per-module exclusions (it commonly skips modules whose dims
   aren't group-divisible and keeps some tensors higher). If fieldfare's weights are calibrated or
   mixed-bit, then "MLX affine 4-bit group 64" in its README is the **format, not the method**, and
   we've been comparing naive min/max against something calibrated — a residual gap **no format
   engineering closes**, and the concrete reason those peer figures needed pinning.
   **→ DONE — fieldfare is NAIVE affine, not calibrated. The comparison is fair; the target is reachable.**
   fieldfare (Swift/Metal) does NOT quantize — it repacks the pre-quantized HF checkpoint
   **`mlx-community/gemma-4-26b-a4b-it-4bit`** (found in its `THIRD_PARTY_NOTICES.md` /
   `docs/SYSTEM_DESIGN.md`) into its `.gturbo` format. That checkpoint's `config.json` quantization
   block: `mode: "affine"`, `bits: 4`, `group_size: 64` — MLX's **data-free min/max affine, NOT
   calibrated** (no AWQ/GPTQ/DWQ). Its *only* mixed-bit exception is **all 30 `layers.*.router.proj`
   at 8-bit**; experts, attention, shared expert, and **embeddings/LM-head are 4-bit** affine g64.
     - So the calibration worry is refuted — naive affine g64 + fp16 activations + 8-bit routers is
       *sufficient* for full coherence. goinfer already keeps the **router at f32** (better than
       fieldfare's int8) and **embed/head at int8** (better than fieldfare's 4-bit), so goinfer's
       target config strictly dominates fieldfare's on the tensors that matter most.
     - The remaining delta between goinfer's best measured cell (affine+f32act = *semi*-coherent) and
       fieldfare (fully coherent) is therefore NOT the weight scheme, activations, router, or
       calibration — the most probable culprit is **the harness's method caveat**: `fakeInt4WM` stores
       the affine reconstruction at **per-row int8**, which *might* re-quantize away the per-group-64
       affine zero-points before the matmul.
       **→ TESTED offline (`scripts/gemma4_int8_restore_probe.py`) and REFUTED.** On real expert
       tensors (down_proj / gate_up, layers 0/15/29), `cos(int8-restore, affine-recon) = 0.99996`
       (min 0.9997) at both g64 and g32 — the int8-per-row storage is **essentially lossless** vs the
       affine reconstruction. So the harness's affine+f32act cell **faithfully represents genuine
       affine-4bit weights**, and a real per-group affine Q4 kernel would land at the **same
       semi-coherence** — the semi-coherence is REAL, not a method artifact.

   **Net of the three probes — the affine build is NOT a confirmed path to coherence; precision does
   not explain the gap.** Routing is healthy (probe 1); the peer is naive-affine, not calibrated
   (probe 3); the int8-restorage is lossless (probe 2 offline). Yet goinfer's tested config already
   **strictly dominates fieldfare on precision** — f32 router (vs its int8), int8 embed/head (vs its
   4-bit), affine experts at g32 (finer than its g64), f32 activations — and still lands only
   *semi*-coherent, while the peer is reported fully coherent. Since a faithful affine representation
   is only semi-coherent, **building the real affine + `.giw` v5 quantizer would most likely also land
   semi-coherent** → do NOT fund it as a coherence fix on this evidence. The unexplained residual is a
   **goinfer-vs-MLX *execution* difference** (candidates: accumulation order / dtype across the 30-layer
   stack, softcap or per-layer-scalar application under 4-bit perturbation, RoPE/norm numerics), OR the
   peer's "coherent" figure is throughput-oriented and its 4-bit output is itself degraded — which this
   Linux+CUDA box **cannot** verify (MLX is Apple-silicon only). Cheapest genuine next steps: (a) run
   the `mlx-community/gemma-4-26b-a4b-it-4bit` checkpoint on a Mac and read its actual output quality
   before spending any more here; (b) if it *is* coherent, diff goinfer's gemma4 forward numerics
   against MLX under 4-bit weights (per-layer activation capture) to find the sensitive op — that, not
   a new quantizer, is where the gap lives. Absent (a), the honest status is: **int8 is the only
   coherent config, and it does not fit 16 GB — Phase 5 stays blocked.**

**Free reconstruction-table columns (per `docs/plan-cpubrrr-steal-and-bindings.md`).** goinfer already
has parity-verified **MXFP4** dequant (`decoder/mxfp4.go`, bit-verified vs `gguf/quants.py`; non-uniform
ladder `{0,1,2,3,4,6,8,12}`×E8M0, 32-elem blocks) and **Q4_K** dequant + a validated Q4_K×Q8_K matmul
(`linalg/kquant.go`). Add both as columns to the Step-3 reconstruction table. **Shape caveat:** Q4_K
needs `K % 256 == 0` (`QuantizeActQ8K` panics; `qkK=256`), and Gemma 4's `down_proj` has `K =
moe_intermediate_size = 704 = 2.75` superblocks — it **does not tile**, absent padding. MXFP4's 32-elem
blocks give `704/32 = 22` ✓ (same divisibility as today's group-32 int4). A **non-uniform ladder
(MXFP4) handles outliers differently from affine** and may behave differently at this coherence
threshold — worth a cell before committing to affine.

Mixed precision still can't substitute the experts away: they are ~22.8 B of 25.2 B params, so keeping
them ≥int8 exceeds 16 GB.

*Method caveat:* the harness stores each 4-bit reconstruction at int8 (near-lossless, 0.99995) and runs
it through the int8/Q8 matmul, so "f32-act" here is int8-weight × f32-act over the 4-bit-reconstructed
values — a faithful proxy for W4A16-affine, not a bit-exact stand-in for a real per-group affine Q4
kernel. Repro: `GOINFER_FAKEQUANT={sym,affine} [GOINFER_FAKEQUANT_ACT=f32] [GOINFER_FAKEQUANT_EXPERTS=1]
ZZBASE={int4,int8int8}` via `TestGemma4FakeQuant` (build tag `realckpt`). The env is read **once at
load**, default-off = bit-identical (`TestFakeQuantOffBitIdentical`).

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

---

## Phases 9a–9d — GPU residency (ordering corrected 2026-08-01 by the feature audit)

Everything above is the pure-Go CPU path. Gemma 4 is `CPU` in every backend column of
`docs/hardware-matrix.md` because `decodeRunnerEligible` declines `a.gemma4 != nil`
**unconditionally** (`decoder/residency.go:130`) — its own non-uniform forward. That
predicate is **arch-only and backend-agnostic**, so the bridge that lifts it is shared work
every backend inherits.

> **An earlier draft of this section ordered Metal → CUDA on the premise that Metal already
> had MoE and `cuda/` was dense-only, and proposed WebGPU as the local proving ground
> because it is "the richest runner." The feature audit (`93a32c0`) overturned both. The
> ordering below supersedes it.**

### The feature audit (`93a32c0`) — what each backend is actually missing

Arch-derived, from `ResidentBackendFeatures` in `decoder/features.go`. Gemma 4 26B-A4B needs
8 features:

| backend | missing for Gemma 4 | count |
|---|---|---|
| `decodeRunnerEligible` | own-forward gate — **blocks every backend** | — |
| **webgpu** | embed-scale, gated-gelu, logit-softcap, sandwich-norm | **4** |
| **cuda** | logit-softcap | **1** |
| **metal** | logit-softcap | **1** |

**The strategic finding:** WebGPU is the *worst*-equipped backend for this family despite
being the richest runner overall. It has the deep MoE / MLA / SSM set but lacks the Gemma
norm/activation/embed kernels CUDA and Metal already ship — and **none of those four
transfer** to another backend. Proving 9a on WebGPU costs the bridge plus four
WebGPU-only kernels; proving it on CUDA or Metal costs the bridge plus one.

### Phase 9a — the own-forward residency bridge (shared; the real prize)

Lift the `a.gemma4 != nil` decline so Gemma 4 is admissible to a resident runner at all.
`docs/c8-softcap-residency.md` is the analysis: Gemma-4-E2B trips **≥3 blockers**, and the
dominant one is *not* the softcap — it is `runLayersGemma4`'s own forward
(`forward_gemma4.go`), which the resident runners' uniform-layer assumption cannot express.
That doc's own conclusion: *"softcap is one of ≥3 blockers, and the dominant one is
Gemma-4's own forward."*

So fund the **own-forward residency bridge**, not a softcap patch. Spec it against the CPU
`runLayersGemma4` first — it is backend-agnostic, it is where the risk lives, and it is paid
once. Phases 9b–9d are comparatively cheap afterwards.

### Phase 9b — prove it on CUDA (local hardware, cheapest gap)

The RTX 2070 SUPER is the same card §B2's CUDA numbers were measured on, so CUDA is *also*
locally gateable — the earlier "WebGPU because it's the box's GPU" reasoning was simply
wrong. Bridge + 1 beats bridge + 4 on identical hardware.

**Resolve two things before writing a kernel:**

1. **A doc/code contradiction.** `docs/benchmarks.md` §B2 states the `cuda/` scope as
   *"dense architectures only. No MoE / MLA / SSM, no partial rotary"* (written 2026-07-16),
   but `features.go`'s `cuda` entry carries `FeatMoE: true` and `FeatPartialRotary: true` —
   which is what makes the audit read "1 missing." The features table is authoritative by
   its own charter (*"a backend adds an entry ONLY when it ships the kernel that implements
   it"*), so §B2's prose is probably stale. Confirm which, and fix the loser — the whole
   4-vs-1 asymmetry rests on it.
2. **`FeatLogitSoftcap` is probably over-broad, and the gap may be nearly free.** The flag
   is documented as *"attention / final logit softcap (Gemma 2/3)"* — one bit covering two
   different things. Attention softcap is per-layer and needs a real kernel. Final-logit
   softcap is `softcap · tanh(logits/softcap)`, applied **once** to the logit vector after
   the LM head. Gemma 4 needs only the second: `registry.go` says plainly *"Gemma 4 re-added
   Gemma 2's final-logit softcap (30 in the GGUF)… Attn softcap stays 0."*

   So Gemma 4 is being declined for a capability it does not use. If the resident path
   returns logits host-side anyway — as `FeatEmbedScale` already does (*"√hidden applied
   host-side in `embedResident`"*) — then **splitting the flag** into attention-softcap and
   final-logit-softcap reduces CUDA and Metal from "bridge + 1 kernel" to "bridge + a
   taxonomy split + a few host-side lines." Precedent for splitting rather than
   overclaiming: the `cuda` block already documents the gated-vs-ungated shared expert as
   *"one flag that cannot express the sub-shape."*

The CUDA payoff on its own terms is the strongest performance claim goinfer has —
**1.4–2.0× Ollama-CUDA at equal 4-bit quant, cgo-free, driver-only** (§B2, `7557723`).
Extending that lane to a 26B MoE is a product result. It is **not** the fieldfare
comparison and must not be written up as one.

### Phase 9c — Metal (the only backend that makes the peer row legal)

Unaffected by the audit's reordering: turbo-fieldfare is Swift + Metal on Apple silicon, so
a CUDA number can never be the head-to-head no matter how fast it is. If Phase 5's purpose
is still that comparison, Metal is the port that serves it. It inherits 9a and shares 9b's
softcap resolution as *design* work even though the kernel code does not transfer.

Gemma-4-specific work sits **on top of** `docs/task-metal-moe.md`, which scopes Mixtral /
Qwen2-MoE / Qwen3-MoE / GLM-MoE and explicitly excludes families with their own forwards —
easy to under-scope as "Metal already does MoE." See the delta below.

**Memory note:** `hardware-matrix.md` records that *"MoE residency is unified-memory-bound"*
on Metal. 26B-A4B at int4 is ~13 GB (~11 GB with the int4 head from `96269ca`) against a
16 GB M1 Pro — resident is plausible but tight, and it is the same paging-bound regime
Phase 5's row has to caveat. Resident does not imply comfortable.

### Phase 9d — WebGPU: deprioritized for this family

Four missing kernels, none of which transfer. Worth building only if WebGPU-Gemma is a goal
in its own right — not as a proving ground for the bridge.

### The Gemma-4 MoE delta (applies to whichever backend lands)

The resident MoE path assumes a single-branch MoE FFN. Gemma 4 does not fit it:

- **parallel dense-MLP ‖ MoE** sub-block with seven norms and `layer_scalar` (§A1) — the
  single-branch path cannot express it;
- **gelu-tanh GeGLU experts**, not SwiGLU — `task-metal-moe.md`'s `gemv_w4a8_moe` epilogue
  assumes SiLU;
- the **Gemma-4 router**: weightless pre-norm + learned `[hidden]` scale + `hidden^-0.5`,
  unconditional renorm, learned `per_expert_scale[nE]` (§A2). The `moe_route` kernel covers
  softmax/sigmoid top-k and none of these;
- **K=V global layers** (`global_head_dim` 512, 2 KV heads, `attention_k_eq_v`),
  **proportional/partial rotary** on the global layers, the **5:1 sliding:full** interleave,
  and **final-logit softcap 30**.

### Gates (every phase above)

- Resident decode held to the repo's **3% near-tie parity rule** against the CPU path on the
  real q4 checkpoint, the same bar §B2's CUDA numbers passed (9/10 exact argmax, 0 hard
  fails). Speed claims are only meaningful because the tokens match.
- Regenerate `docs/hardware-matrix.md` (`go test ./decoder -run HardwareMatrix -update`) and
  `docs/capability-matrix.md`. The Gemma 4 row moves off `CPU` **only** for the backend that
  actually landed — no inheriting a `✅ resident` from the MoE feature flag.
- **If `FeatLogitSoftcap` is split (9b.2), every backend's entry must be re-derived from
  what it actually ships**, not mechanically copied. `features_test.go`'s registry-driven
  admission gate exists precisely to catch an overclaim here; a split that quietly grants
  three backends a capability none of them implemented is the failure mode.
- Per `docs/benchmarks.md` methodology: same machine, same checkpoint, same quant, greedy,
  pinned peer version, date + commit + thermal note inline.

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
