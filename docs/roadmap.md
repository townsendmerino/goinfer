# goinfer roadmap (rolling; updated 2026-06-12, KV-memory program steps 1–2 shipped; int4 deferred)

> **Audience:** internal planning doc (docs/internal/ is gitignored). Started
> as the v0.2 gap analysis vs llama.cpp / Ollama / mistral.rs; v0.2.0 and
> v0.3.0 both shipped everything it scoped. This revision: fresh peer survey
> (2026-06-07) + the v0.4 sketch. Historical v0.2/v0.3 detail lives in
> CHANGELOG.md and the git log; the original survey conclusions are at the
> bottom.

## State of play

**v0.3.0 is tagged and released** (release notes:
`docs/internal/release-notes-v0.3.0.md`). Cumulative since v0.1.3, goinfer
gained: Gemma 4 (E2B/E4B/12B, parity-gated), sampler completeness (min-p,
penalties, logit bias, logprobs), the `chat` package (template detection +
byte-exact native renderers), JSON Schema constrained decoding
(`GrammarFromStruct` — the flagship), tool calling (per-family render +
parse + grammar-constrained calls), `cmd/serve` (OpenAI-compatible, pure
stdlib, warm KV sessions, `/v1/embeddings`), `decoder.Session`
prefix caching with `.giw-kv` snapshots, LoRA adapter loading, Q6_K/Q4_K_S
parity pins, and aikit v1.0.0 across both modules.

**Every item from the original gap table is closed** except the two
deliberate deferrals: continuous batching/PagedAttention and multimodal.
The catch-up phase is over; v0.4 should play offense.

> **✅ DONE — overnight GGUF fuzz soak PASSED (2026-06-10→11).** A clean full 8h
> `FuzzGGUFConfig` run on the Linux box: **~4.91B execs, no crasher** under
> `decoder/testdata/fuzz/`. This clears the release-notes claim "hostile model
> files error, never panic" (shipped in v0.5.0). (Run soaks ALONE — an earlier
> attempt died at ~4h52m with a non-reproducing false-hang crasher purely because
> concurrent `-race ./...` runs starved the box.) Optional future follow-up: a
> deeper target over `buildWeightsFromGGUF` (dequant/shape-check/requant with real
> tensor data) — still unfuzzed on goinfer's side (the K-quant kernels are aikit's,
> fuzzed in `6f416ca`).

## Positioning: attention-coverage axes (not model count)

The field has converged on **three** efficient-attention/sequence-mixing families,
and goinfer now runs two of them — the strategic frame for model work going forward
is **covering the axes, not counting models**:

| Axis | Mechanism | goinfer status |
|---|---|---|
| **gated-linear** | Gated DeltaNet (recurrent matrix state) | ✅ qwen3_5_moe |
| **state-space** | Mamba-2 selective scan | ✅ Granite-4.0-H |
| **latent-KV** | MLA (compressed low-rank KV) | ✅ DeepSeek-V2/V3 (deepseek_v2 / deepseek_v3) |

"Runs every modern attention variant in one cgo-free static binary" is a stronger,
more defensible claim than "supports N models," and one no other pure-Go runtime can
make — the per-layer-kind machinery already generalizes from mixer-kind to
attention-kind. **Family selection is axis-driven; popularity breaks ties, it
doesn't lead.** Full sequencing (Nemotron-H → MLA → Llama 4 / Phi momentum) and the
per-family definition-of-done are in `docs/completed/task-model-families-next.md` (built on
`docs/parity-coverage-policy.md`).

## Fresh peer survey (2026-06-07)

The field's competition moved while we closed the library gap. Five
battlegrounds now:

| Battleground | Who's pushing | goinfer position |
|---|---|---|
| **MTP speculative decoding** (models ship multi-token heads; no draft model; ~2x on dense) | llama.cpp (Qwen 3.6 MTP, PR #22673), Ollama 0.23.1 (Gemma 4 MTP via MLX, 2x on 31B), LM Studio 0.4.14 (stable), vLLM (Gemma 4 MTP; EAGLE 3.1 in v0.22) | Greedy draft-model spec exists, parked on CPU. **MTP doesn't change the math**: CPU decode is compute-bound, so an M=K verify pass costs ~K sequential decodes regardless of how cheap drafting is. Stays parked; revisit with a bandwidth-bound (GPU) backend. Caveat from the field: even llama.cpp sees *no* MoE single-stream win on consumer hardware. |
| **Model freshness** | Qwen 3.6, DeepSeek (MLA), GLM-4.5/4.6, Granite 4.0, Nemotron-H, Gemma 3n, Llama 4, Phi | The race the descriptor design lets us run cheaply — now **axis-driven** (see "Positioning: attention-coverage axes" above): GLM-4.5/4.6 + Granite-4.0-H (Mamba-2) + Nemotron-H + DeepSeek-V2/V3 (MLA, latent-KV) shipped — all three efficient-attention axes covered — plus Phi-3/Phi-4 (momentum). Plan: `docs/completed/task-model-families-next.md`. Gemma 4 already in, incl. PLE (which llama.cpp struggled with). |
| **Hardware acceleration depth** | CUDA kernel fusion (+24% on 4090), Vulkan now beating CUDA on some hardware, M5 Neural Accelerators (MLX-only, up to 4x TTFT) | Structural gap, accepted trade. Pure-Go CPU SIMD + opt-in WebGPU. The counter-position *is* portability: one static binary, every platform, no driver. Don't fight on kernels. |
| **Serving surface** | mistral.rs v0.7: dynamic model load/unload, **multi-model serving**, initial **OpenAI Responses API**, prefix caching; LM Studio: MCP host + OAuth; Ollama: `launch` app integrations | **Largely closed:** multi-model + dynamic admin load/unload and the OpenAI Responses API shipped v0.4.0; prefix caching at parity; the **Anthropic Messages API** (`/v1/messages`) adds a second chat surface in v0.5.0 (Claude Code points at goinfer). Remaining true gaps: MCP host / OAuth / `launch`-style app integrations. |
| **Quantization frontier** | Native FP8/FP4 (DeepSeek V4 in llama.cpp), TurboQuant (4.9x vs f16, tracked in llama.cpp #20969), NVFP4 on Apple | int8/int4 runtime + broad K-quant dequant coverage. FP8/FP4 is GPU-hardware-driven — not our fight on CPU. Watch TurboQuant: a CPU-implementable 3–4 bit quant with near-paper MSE could matter for the embed-demo size/quality curve. |

### Timely, high-fit openings — BOTH LANDED (detail in Track A)

1. ✅ **Mellum2 (JetBrains)** — 12B MoE / 2.5B active, code-focused. Fully
   closed: parity golden, chat template, window eviction.
2. ✅ **Qwen 3.6 (35B-A3B)** — landed as the largest forward-pass addition to
   date (Gated DeltaNet hybrid). **Real-checkpoint int8 full-model parity
   validated** (Gate 2, `e3eb033`). Remaining before a headline "Qwen 3.6
   support" claim: the GGUF loader (loads safetensors today). See Track A.

## v0.4 — theme: "new models + serve polish"

### Track A: model freshness — LANDED (2026-06-07), with follow-ons

- ✅ **Mellum2** (`f70cf51`, `4b0e7be`) — fully closed: chat template
  (named-ChatML alias, byte-exact golden), HF logit-parity golden,
  sliding-window eviction exercised.
- ✅ **Qwen 3.5/3.6-MoE** (`f70c013`) — **the largest forward-pass addition
  to date**: Gated DeltaNet hybrid (3:1 linear/softmax attention), new
  `deltanet.go` sequence-mixing primitive, hybrid cache (per-layer
  deltaState alongside KV), 256-expert + shared MoE. Bit-exact vs the HF
  oracle (argmax + cosine 1.0) on a generated tiny checkpoint.
- ✅ **Real-checkpoint parity — DONE** (`e3eb033`, Gate 2 GREEN). The real
  Qwen3.6-35B-A3B ships as a **VL model**; goinfer loads its **text decoder**
  (`language_model.` prefix + fused/stacked 256-expert MoE unpack; vision/MTP
  ignored). Gate 1 (slice, f32) cosine 1.0 bit-exact; **Gate 2 (full 40-layer,
  int8, ~39 GB resident via streaming-quant fused experts, vs bf16 golden):
  argmax 74/80 (92.5%), 8/10 prompts reproduce the golden continuation, logit
  cosine min 0.99466 / mean 0.99859** (both misses near-ties, <3% of range).
  **It runs at int8 from safetensors** — not f32-only.
- **Open follow-ons** (Track A isn't *practically* done until these land):
  - [x] **GGUF loader for `qwen3_5_moe`** — LANDED (`166d957`, the
        transform-reverser). It reverses the layout transforms the GGUF
        converter bakes that the safetensors path never sees (V-head un-tiling,
        q‖gate split, `A_log` as pre-baked −exp, norm `(1+w)` un-baking).
        Bit-level proven by `TestQwen35GGUF_weightDiff` (every transform-bearing
        tensor diffed against the bit-exact safetensors loader, to Q8_0
        tolerance — no oracle/torch needed).
  - [x] **Release-headline verification — DONE** (2026-06-10, Linux box, real
        Qwen3.6-35B-A3B Q8_0 GGUF 36 GB + safetensors + bf16 golden all on disk).
        `TestQwen35GGUF_weightDiff` GREEN (worst tensor cos 0.999980 = Q8_0 dequant
        tol; norms/negExpA/dt_bias bit-identical). Full Q8_0-vs-bf16 gate
        `TestQwen35GGUF_gate` GREEN: **argmax 72/80 (90.0%), 7/10 prompts coherent,
        cosine min 0.99583 / mean 0.99850, worst div gap 0.0052** — the 3 misses all
        rank-2 near-ties (Q8_0 quant noise, not a loader defect). (Ran ~4.8 min, not
        the stale ~40 min estimate — the SIMD prefill attention sped it up. Cosine min
        now *exceeds* the original Gate-2 bf16 bar 0.99466; only argmax 72<74, so the
        relaxed 66/80+0.9943 GGUF bars stay and pass with margin.) **"Qwen 3.6
        support" is now release-headline-ready.**
  - [x] **Chunked DeltaNet scan — kernel** — `deltanet_chunked.go`: the
        chunked-parallel gated-delta scan, proven algebraically equivalent to the
        sequential recurrence over random inputs/chunk sizes
        (`TestGatedDeltaNet_chunkedMatchesSequential`; self-contained, no
        asset/torch). gt = exp(g) ∈ (0,1) keeps every decay ratio ≤ 1 → stable.
  - [ ] **Wire the chunked scan into a batched `qwen3_5_moe` prefill** — the
        forward is still single-token streaming (`runLayersQwen35`), so the scan
        kernel isn't yet on the hot path; this + batched projections (+ quantized)
        is the perf win, to be validated end-to-end against the checkpoint.
  - [x] **Hybrid models opt out of prefix-reuse and speculative** — the
        recurrent deltaState isn't position-truncatable; fall back to the
        staged path, documented. (A deltaState snapshot-at-position scheme is
        possible but priced separately.)
- Process note (validated twice now): each family lands as descriptor +
  loader + parity golden, and the whole v0.2/v0.3 surface (templates,
  tools, constrain, serve) inherits automatically. The moat compounds.

### Track B: serve polish — LANDED (shipped v0.4.0; Anthropic Messages API added v0.5.0)

`cmd/serve` grew from one-model/Chat-Completions to a model host, pure stdlib:

- ✅ **Multi-model** — `--model name=path` repeatable; requests route on the
  OpenAI `model` field; `/v1/models` lists all; per-model mutex (distinct
  models run in parallel) + per-model warm-KV dir.
- ✅ **Admin API** (`--allow-admin`, default off — RCE-adjacent):
  `POST /admin/models/{load,unload}`; unload refuses a busy model (409) and
  snapshots its warm KV.
- ✅ **OpenAI Responses API** (`/v1/responses`): `input`/`instructions`/
  `text.format`(→constrain)/`tools`, streaming event shapes,
  `store`/`previous_response_id` (in-memory ring riding the per-model
  sessionLRU).
- ✅ **Backpressure** — bounded per-model queue (`--max-queue`, default 8) →
  429 + Retry-After when full (single decode worker/model; not continuous
  batching, which stays deferred).

### Track C (unplanned): GPU decode investigation — CLOSED (2026-06-08)

**Full writeup + the decision matrix: `docs/gpu-assessment.md` (§0.0, §1).**

The arc that began as "is the staged hybrid optimal?" closed in the opposite
place: a **full-token on-GPU residency forward** wins. Shipped:

- **GPU full-residency decode wired into `decoder.Generate`** (webgpu backend,
  dense Qwen2/Llama). Real int4 `.giw` models now run **pure-GPU** end-to-end,
  not the per-matmul staged path. Greedy output matches the CPU decode.
- **W4A8 (int4) is the capability unlock:** a **7B int4 fits and runs pure-GPU
  on the 8 GB card at ~51 tok/s** — the model class that does NOT fit at int8 —
  at **~71% of llama.cpp-CUDA** tok/s at equal 4-bit quant (1.5B: 102 tok/s,
  ~55%). The engine gap *narrows* with model size; this is a footprint/capability
  win, not a speed record (the WebGPU dispatch model still trails CUDA's
  megakernel). int8 decode peaks ~89.7 tok/s on the 1.5B (3.5× the staged hybrid).

Deferred follow-ons (named, **not scheduled** — pick up if a use case pulls):

- **Batched on-device GPU prefill** — today's prefill is option-(a) sequential
  GPU `Run`, O(prompt-len); a batched M=len pass *might* fix long-prompt TTFT.
  **Evaluated 2026-06-09 on BOTH backends — WASH until dp4a, gated on
  `dot4I8Packed`.** Weight-stream-bound; the lever is amortizing the VRAM weight
  read via a compute-bound tiled GEMM, but the WGSL tiled kernel can't reach DP4A
  so it tops out below the bandwidth-bound GEMV: Metal ≈1.2×, RTX ≈0.91× (tiled
  680 GFLOP/s vs the M=1 GEMV's 748 GFLOP/s-equiv). Not silicon-limited — kernel-
  limited. See Backlog → GPU long-context.
- **f16 KV cache** — unlocks 32k context for a 7B on 8 GB (v1 caps at 16k, f32).
- **CPU-int4 decode — RESOLVED (2026-06-08, aikit v1.1.1).** The §1 matrix
  measured CPU int4 at ~0.15–0.18 tok/s because `MatmulBTQ4` (f32 activation) is
  dequant-bound at M=1. aikit v1.1.1's `MatmulBTW4A8` (int4×int8 integer decode
  kernel) replaced it for M=1 decode (`MatmulBTQ4` stays prefill); re-measured at
  **2.1–4.3 tok/s, 1.4–1.9× of CPU int8int8** — CPU int4 is now usable, not only a
  GPU footprint win.
- **W4A8-on-disk format** — f16 group scales in the `.giw` (smaller files; the
  kernel is ALU-bound so it wouldn't speed decode).

**Pivot (updated 2026-06-09):** the GPU arc is closed and Track A/B landed,
**including the `qwen3_5_moe` GGUF loader** (`166d957` — the transform-reverser,
weightDiff-proven). So **v0.4.0 was feature-complete**; the remaining work is the
backlog below (mostly deferred/triggered, none release-gating). See "Backlog".

**v0.5.0 (tagging 2026-06-11):** the Anthropic Messages API (`/v1/messages` —
Claude Code points at goinfer), the f16 GPU KV cache (`--kv f16`, ~32k context),
~2.4× batched sparse-MoE prefill, fuzz-hardening now soak-verified (clean 8h
`FuzzGGUFConfig`), and `demo/agent`. Multimodal vision P0–P3 (Gemma 3 VL
image→logits at HF parity) is on `main` but **held from the release notes** —
no user-facing serve path yet (P4–P5 remain).

### Watching, still not doing

- MTP speculative — parked (compute-bound CPU; see survey). Re-evaluate the
  moment the WebGPU backend graduates from matmul-only.
- TurboQuant — promoted from "watch" to a **scoped, timeboxed spike** inside the
  KV-memory program below (`task-turboquant-spike.md` — pointed at KV, not
  weights, with a written go/no-go bar).
- Continuous batching / PagedAttention — unchanged deferral. Multimodal —
  no longer deferred: P0–P3 landed (`docs/multimodal.md`), P4–P5 open.

## v0.6 candidate program: KV-cache memory reduction (scoped 2026-06-11)

**Umbrella + index: [`task-memory-program.md`](completed/task-memory-program.md).** Four
independently-shippable steps, each with its own gates; the default decode path
stays **bit-exact** throughout — every lossy step is an opt-in knob (house
rule). Combined headline (steps 1+2): **~20× smaller KV on Gemma-class local
layers at long context, 4× on full-attention families, 4× smaller `.giw-kv`
warm sessions.** This is the natural sequel to the v0.5.0 f16-KV work: same
"context per byte" battleground, attacked on CPU and GPU.

**Status (2026-06-14): the KV-memory program is fully resolved.** Steps 1 + 2 + 3
SHIPPED to main (ring eviction + CPU int8 KV Inc 1–3 + GPU int8 KV — the ~20×-stacked
headline on Gemma local layers plus ~64k GPU context, opt-in, default bit-exact;
step 3 in **v0.6.0**, `3e27dcc`). Step 4 (TurboQuant spike) **spiked → NO-GO**
(`completed/task-turboquant-spike.md`): 3-bit KV with a CPU-cheap foldable Hadamard
is 22–30× worse than int8 in recon — the watch item closes. int4 KV stays deferred
behind the sufficient int8 (see step 2).

In order (rationale: free memory with zero precision argument first, then the
quant rungs with pre-validated gate bars):

1. ✅ **DONE — [Ring-buffer eviction for sliding-window layers](completed/task-kv-ring-eviction.md)**
   (commit `4a9c2b5`, 2026-06-11/12). Rings on local layers: **~5.2× KV on
   Gemma-class, LOSSLESS** (bit-exact gate — real Mellum2 window golden + 3
   model-free gates). Gotcha the doc missed: batched prefill (K>W) needed a
   deferred-write + assembled-window path, not a naive `s%W`.
2. ✅ **DONE — [CPU int8 KV cache, Increments 1–3](completed/task-cpu-kv-quant.md)**
   (commits `71d4ef1` / `d78eea9` / `d4e7830`). Per-head int8 on shipped aikit
   kernels (`DotI8`/SDOT). Inc 1 storage+decode, Inc 2 batched prefill + serve
   `--kv-quant i8` + spec-decode rollback parity (**TTFT 1.023×** after killing
   alloc churn), Inc 3 snapshot-v2 (merged with step 1's windowed persistence —
   **the format bumped once**). Opt-in, **default f32 bit-exact**. Measured gate:
   argmax ~87–93% / cosine ~0.993 on gemma-3-4b-it, coherent generation (in line
   with the shipped full-int8-*weights* precedent, 92.5%). The 0.999 cosine bar in
   the original plan was wrong — unreachable over 34 layers; argmax/cosine on a
   *model-tokenized* prompt is the right gate (a garbage-input test bug — foreign
   token-ids — once faked a 0.73 "failure"; lesson logged).
   - **Inc 4 (int4 KV, group=32) — DEFERRED, by design.** Trigger: int8 shipped
     clean (done) **AND** a concrete >32k-single-context or fat-multi-session RAM
     wall that 20×-stacked int8 doesn't clear. Why not now: int4 buys only **~1.6×
     over int8** (at group=32 the scales dominate — 0.5 + ~0.125 B/value), and on
     the quant-sensitive arch (gemma3: QK-norm, 256-dim heads, 34 layers) it
     **likely needs per-channel keys (KIVI) → chunked storage + an f32 residual** —
     the exact streaming-cache complexity int8 was validated NOT to need. So
     int4-now ≈ a week of high-risk work that may miss its own gate, plus permanent
     code surface, for ~1.6× over a 20× we already ship, with no demand. Easy call
     to flip the moment a real RAM wall appears.
3. ✅ **DONE — [GPU int8 KV cache](completed/task-gpu-kv-i8.md)** (commit `3e27dcc`,
   shipped in v0.6.0). Third rung after `--kv f16`, via the existing `runQuant`/W8A8
   WGSL idiom: **~64k context on the 8 GB card in the 32k-f16 envelope** (`--kv i8`).
   The three int8 kernels (`ropeStoreI8`/`kvStoreI8` per-head absmax+quantize,
   `attnI8` unpack+scale READ arm) + residency integration + tri-state knob. Real-HW
   gate (Qwen2.5-7B int4, 8 GB): int8-vs-f32 decode **mean cosine 0.99739** (argmax
   3%-near-tie rule), 64k peak **6977 MiB** (fits), decode **0.96× at 1k ctx**
   (weight-stream-bound short-ctx). The WRITE-kernel per-head reduction risk retired
   (kernels bit-exact vs CPU, cosine 1.000000). **Long-ctx residual now MEASURED
   (2026-06-15): the KV-read-bound speedup does NOT materialize** — i8/f32 stays
   ~0.96× *flat* at 1k/4k/8k (f16 ~0.98×). Decode is steeply attention-bound at depth
   (~5× falloff) but the attention kernel is compute-bound on the f32 dequant+dot, not
   bandwidth-bound on KV bytes; fewer bytes don't speed it. int8/f16 KV are **capacity
   wins (4×/2× context per VRAM), not decode-speed wins** pre-`dot4I8Packed`/DP4A.
4. ✅ **SPIKED → NO-GO — [TurboQuant spike](completed/task-turboquant-spike.md)**
   (2026-06-14). Recon screen on real qwen2.5-coder-1.5B post-RoPE K/V: 3-bit KV
   with the CPU-cheap foldable random-sign Hadamard is **22–30× worse than int8**
   (K int8 0.012 → int3+rot 0.259; V 0.009 → 0.267) — the rotation helps 25–41% but
   can't overcome a 42× level-count deficit (int3 ≈ 42× int8, matching 127/3). The
   published near-zero-loss 3-bit KV doesn't transfer to a foldable rotation; closing
   it needs data-dependent/per-channel cost the premise rejects. Watch item closed;
   TurboQuant stays weights-only-if-triggered.

Assessed and rejected for the program (reasons in `completed/task-cpu-kv-quant.md`
§Non-goals): CPU f16 KV (dominated by int8 on CPU), XQuant rematerialization
(wrong direction when compute-bound), attention-score eviction / AhaKV
(quality-risky, breaks exact prefix-reuse).

## Backlog — every surfaced idea (2026-06-09)

A full capture of ideas raised across the GPU/W4A8/qwen3.6 work, with honest
status + the trigger that would promote each. **None of these is release-gating;
v0.4.0 ships without them.** Grouped by theme; ordered within a group by leverage.

### GPU decode performance (the residency path)

- **More decode-fusion headroom exists, but the WebGPU wall is near.** Decode is
  **glue-serialization-bound**, not bandwidth-bound (`gpu-assessment.md` §0.5,
  `internal/completed/task-gpu-decode-fusion.md`): the matmul roofline floor is ~4.1 ms ⇒ ~240 tok/s
  if glue were free, but the real token is ~11 ms (89.7 tok/s int8 1.5B) because
  each glue link forces a barrier the GPU can't hide. Fusion + the attn
  warp-per-head rewrite already took it 22→89.7. The remaining ~4 ms glue + ~1 ms
  encode is the gap; squeezing it further needs a single-dispatch **megakernel
  (whole layer in one dispatch), which WGSL cannot express.** So ~90–100 tok/s is
  roughly the WebGPU ceiling on this card; past it needs native CUDA/Metal.
  **Status: effectively at the practical ceiling. Don't grind.**
- **`dot4I8Packed` (dp4a) — prefill compute boost, upstream-blocked.** The native
  int8-dot WGSL feature would speed the M>1 prefill GEMM (TU104 has DP4A
  hardware). Blocked on the `cogentcore/webgpu` binding exposing it. **Trigger:
  the binding ships it.** **This now gates _batched on-device GPU prefill_** (GPU
  long-context) — the 2026-06-09 RTX measurement showed the tiled GEMM stuck at
  680 GFLOP/s (≈50× under the DP4A int8 ceiling, *below* the 748 GFLOP/s-equiv
  bandwidth-bound GEMV), so prefill batching is a wash until this lands.

### GPU long-context (both deferred, de-risked, coupled)

These two are a natural pair (long context ⟹ long prompts) and both target the
8 GB-residency users the W4A8 work serves. The residency path is **stateless in
v1** (no prefix-reuse), so a multi-turn / RAG user re-prefills the whole history
every turn at O(len) AND hits the 16k cap — there's no warm-KV escape hatch like
the staged path has. **That coupling is the felt-pain trigger for both.**

- **Batched on-device GPU prefill** (`task-gpu-batched-prefill.md`) — **EVALUATED
  2026-06-09 on BOTH backends (M1 Pro Metal + RTX 2070 SUPER / TU104, component-level
  on real HW). Verdict: WASH on both — shelved until `dot4I8Packed` (dp4a) unblocks;
  that item is now the PREREQUISITE, not a parallel nice-to-have.** Mechanism
  corrected: option-(a) prefill is **weight-stream-bound, not fence-bound** — a 1.5B
  M=1 token is 16 ms, **91% (14.6 ms) re-reading the 1.55 GB resident weights from
  VRAM** at M=1 GEMV (Metal 106 GB/s, 30% roofline; **RTX 374 GB/s, ~83% of the
  448 GB/s peak — already bandwidth-saturated**); sync ~1 ms, glue ~0.4–0.5 ms. So
  batching's lever is amortizing that VRAM weight read (same as the CPU prefill win),
  NOT cutting fences — and prefill competes against the GPU's own bandwidth-bound
  GEMV, NOT the CPU.
    - **Metal:** M=L tiled tops out at 245 GFLOP/s ≈ the GEMV's 212 GFLOP/s-equiv →
      ≈ **1.2×**, wash.
    - **RTX (measured `TestTiled_microbench`+`TestDecode_instrument`):** tiled M=512
      = **680 GFLOP/s** (2.87× CPU's 237, 1.54× the naive GPU path) vs the M=1 GEMV's
      **748 GFLOP/s-equiv** (374 GB/s × 2 ops/byte) → ratio **≈0.91× — wash, slight
      loss.** The 2.87×-CPU optimism (gpu-assessment) was a red herring: the CPU was
      never the competitor.
    - **Why it's a wash and why it's recoverable:** 680 GFLOP/s is ~50× below the
      card's int8 ceiling (~36 TOP/s via DP4A) — the WGSL tiled kernel is **ALU/binding-
      limited (no `dot4I8Packed`), not silicon- or bandwidth-limited.** The RTX has the
      bandwidth headroom AND the DP4A hardware; the `cogentcore/webgpu` binding just
      can't reach it. **When dp4a unblocks, tiled should jump toward the 36 TOP/s
      ceiling, clear the 748 bandwidth wall, and batched prefill wins big** — see the
      `dot4I8Packed` item under *GPU decode performance*.
  De-risked separately: residency is full-attention-only (`SlidingWindow == 0`), so
  the batched attention is **plain causal, no per-query sliding-window mask** — the
  doc's main parity risk is moot.
- **f16 KV cache** (`completed/task-gpu-f16-kv.md`) — **LANDED 2026-06-10 (RTX 2070 SUPER),
  Increments 1–2.** Opt-in `--kv f16` (default f32/bit-exact): cache is array<u32>,
  2 f16/word via core pack2x16float/unpack2x16float — **no shader-f16 feature, CI
  software adapter still compiles.** Gates: f16-vs-f32 decode over an 8k-key context
  cosine min 0.99868 (15/16 argmax, the flip a 0.22% near-tie); f32 path still
  cosine 1.000000. **Fit (the headline): Qwen2.5-7B int4 + 32k f16 KV peaks at
  6912 MiB whole-device ≈ 16k f32's 6926 MiB — fits the 8 GB card, 2× context for
  free.** Commits `138d5e0` (Inc 1) + the Inc-2 knob. **MEASURED (2026-06-15): the
  predicted *faster* long-context decode does NOT materialize** — f16/f32 stays ~0.98×
  *flat* at 1k/4k/8k (int8 ~0.96×). Decode is steeply attention-bound at depth but the
  attention kernel is compute-bound on the f32 dequant+dot, not bandwidth-bound on KV
  bytes, so fewer bytes don't speed it (gated on `dot4I8Packed`/DP4A, like prefill).
  f16/int8 KV are **capacity wins (2×/4× context), not speed wins** on this card. The
  2×-context win stands; the speedup claim is retracted.
- **Decision (was "gates both"):** f16 KV landed cheaply and stands alone (2× context,
  no speed regression). Batched prefill stays shelved on the dp4a gate above. If GPU
  residency is marketed as "run a 7B locally on a small GPU," f16 KV is now the
  long-context answer; the long-context *speedup* and batched prefill wait for their
  triggers (a real >16k workload; dp4a). **Rung after f16: GPU int8 KV — SHIPPED**
  (`completed/task-gpu-kv-i8.md`, step 3 of the KV-memory program above; v0.6.0,
  `3e27dcc`): `--kv i8`, ~64k in the 32k-f16 envelope. The long-context *speedup*
  residual is now MEASURED (2026-06-15) — it does NOT materialize (i8/f32 ~0.96×
  flat at 1k/4k/8k; attention is compute-bound, not KV-byte-bound, until DP4A). KV
  precision is a capacity lever, not a speed lever, on this card.

### GPU residency coverage

- **Hybrid models (qwen3_5_moe) stay on the staged path** — `deltaState` isn't
  residency-eligible and isn't position-truncatable (so no prefix-reuse /
  speculative either). A **deltaState snapshot-at-position** scheme could enable
  prefix-reuse for hybrids; priced separately, no demand yet.
- MoE / Gemma 4 also excluded from residency (eligibility = dense Qwen2/Llama,
  full attention). Expanding the eligible set is open-ended; do it per-arch on
  demand.

### Quantization & CPU kernels

- ✅ **Scalar attention → SIMD prefill (LANDED, post-0.4.0).** An end-to-end CPU
  profile (the aikit attention bug, checked for transfer) found per-position
  scalar QKᵀ + scores·V was **~49% of dense prefill CPU**, and `gemma4Attend`
  ~99 s flat (the #1 gemma4 hotspot). Routed both O(L²) terms onto the SIMD A·Bᵀ
  kernel (`MatmulBT`): dense prefill **3.4×** (12.3→42.1 tok/s, L=2048 1.5B),
  gemma4 **1.7×** (9.5→16.25, E2B); qwen3_5_moe weight `matvec`→`MatmulBT`
  (DeltaNet cosine 1.0). Parity = argmax-exact + cosine (f32 SIMD, the
  GPU-attention standard), not bit-identical. Commits `7fa82c2`/`88b7aaa`/
  `443ab7a`/`4fcc069`.
  - **Deferred follow-on — profile-first on the 64 GB box:** the
    `forwardN`-ineligible MoE families + GPT-2 still take scalar `attendQuery` per
    prefill token. Likely win: extend `canBatchN` to the MoE families (their
    attention is standard; `moeMLP` is already SIMD) so `attendBatchedHeads`
    applies — **but profile a real MoE prefill first**. Also measure the
    qwen3.6-35B `matvec`→SIMD delta there (correctness proven; tok/s not, no MoE
    model on the Mac).
- ✅ **DONE — Route int4 *prefill* to W4A8 too** (commit `17a9ff4`, 2026-06-12).
  `matmul()` now runs `MatmulBTW4A8` at EVERY M (decode + prefill) and drops the
  dequant-bound `MatmulBTQ4` (f32-activation) path — faster at every M on AVX2 and
  it unifies activation precision (an int4 generation is int8-activation for the
  prompt AND generated tokens; W4A8 is per-output M-independent, so batched prefill
  is bit-identical to sequential decode). The lone remaining refinement — a *fused*
  W4A8 batch (one activation-quant for qkv/gate-up, like `MatmulBTW8A8Batch`) —
  needs an aikit `MatmulBTW4A8Batch` (doesn't exist yet; cross-repo, not on this box).
- **W4A8 CPU VNNI variant** — aikit deferred the `VPDPBUSD` (AVX-512 VNNI) path;
  drop-in behind the same CPUID gate, needs a VNNI box to validate.
- **`.giw` f16 group scales** (`task-giw-f16-scales.md`) — **lowest value of all**:
  ~10% smaller files, zero decode/fit win, adds a permanent format-version branch
  (`giwVersion 1→2`, distinct from the container `bundleVersion=2`) + a lossy CPU
  re-gate. f16 scales is the *only* file-size lever left (zstd was dropped — q4 is
  high-entropy, ~3%). **Trigger: a real distribution-size complaint** (model zoo,
  bundled release artifact).
- **TurboQuant** (llama.cpp #20969) — a CPU-implementable 3–4-bit quant with
  near-paper MSE. **No longer just a watch item:** scoped as the timeboxed
  KV-pointed spike in the KV-memory program (`task-turboquant-spike.md`,
  step 4 — go/no-go bar written). The *weights* angle (embed-demo
  size/quality curve) stays a watch.

### Speculative / MTP

- **Re-evaluate spec-decode against the *measured* GPU numbers.** It was parked on
  CPU (compute-bound). The GPU unpark condition was "bandwidth-bound decode," but
  decode turned out **glue-serialization-bound** (not cleanly bandwidth-bound), so
  the batched-verify economics need re-deriving against real numbers before
  building. The exact `GenerateSpeculative` + `TruncateTo` machinery is ready.
- **MTP heads** (the whole field shipped them) — same gating logic; only worth it
  on a backend where the M=K verify is genuinely cheap.

### Serve hardening (as usage appears)

- True **incremental tool-call streaming** (today: buffer full output, then emit
  deltas — correct, not progressive).
- **N decode workers per model** (today: one worker + a bounded queue). True
  continuous batching stays deferred (server-scale, wrong weight class).

### Model freshness

- **GLM-4.5/4.6 + Granite-4.0-H SHIPPED** — the recurring muscle (descriptor +
  loader + parity golden) made each cheap; the v0.2/v0.3 feature surface inherited
  automatically. Granite added the **Mamba-2 (state-space)** axis on the
  deltanet/hybrid-cache shapes — the **v1.0 trigger** (a second hybrid family,
  *different* mixer, shapes unchanged: the loader contract has settled).
- **Axis-driven** (see "Positioning: attention-coverage axes" + the full plan
  in `docs/completed/task-model-families-next.md`): **Nemotron-H** (free — reuses Mamba-2,
  fully bf16-oracle-testable) ✅ and **MLA/DeepSeek** (the third axis, latent-KV —
  the one real new primitive) ✅ have shipped, completing all three efficient-attention
  axes. **Phi-3/Phi-4** shipped (momentum, no new axis — fused qkv/gate_up on the llama
  rails); **Llama 4** remains as the last popularity-momentum pick.

### Measurement / calibration gaps

- **7B-int4-specific Vulkan llama.cpp number.** The engine ratios are CUDA-only;
  a from-source `-DGGML_VULKAN=ON` build gives the fair *same-API* (WebGPU-vs-
  Vulkan) comparison. Moderate yak-shave; the CUDA number was the priority and is
  done.
- **qwen3.6 full-model GGUF *forward* parity gate** is OOM-skipped (two ~39 GB
  models won't coexist on the 62 GB box; `weightDiff` is the proof instead).
  Re-run the full bf16 forward oracle if a larger-RAM box becomes available — nice
  to have, not needed (weightDiff + the shared Gate-2-validated forward is sound).

### Pure-Go / packaging

- **wasm/browser GPU — the uniquely-Go demo, and it's cgo-free.** Native GPU is
  cgo (`cogentcore/webgpu` → wgpu-native). But browser WebGPU via `GOOS=js` reaches
  `navigator.gpu` through `syscall/js` — **no cgo**. A "0.5B in a tab, pure Go
  wasm" demo (gpu-assessment Stage 4) is something no other Go runtime can ship.
  Needs a `syscall/js` WebGPU backend; bounded to small models (browser memory +
  download, and `dot4I8Packed` support is spotty in browsers → likely f32).
- **purego native GPU (if cgo-on-the-GPU-build ever becomes a problem)** — dlopen
  wgpu-native via purego instead of the cgo binding, the way yzma/gollama.cpp stay
  cgo-free. Different dependency, a real port; not worth it while the `-tags gpu`
  quarantine already keeps the default build pure.

### The v1.0 question (rolling — updated 2026-06-07)

Track A's verdict cuts both ways. Mellum2 landed clean (descriptor held —
good evidence). But Qwen 3.6 *grew* the surface by its largest increment
ever: a second sequence-mixing primitive, a second cache type, per-layer-kind
descriptors. The loader/descriptor contract now contains its newest,
least-settled member — and hybrid/linear-attention models are clearly a
trend (Qwen, Granite Hybrid, GLM), not a one-off. Freezing now would lock
day-one shapes for the hybrid abstraction. Revised criterion: v1.0 waits
until a *second* hybrid family lands on the existing deltanet/hybrid-cache
shapes without breaking them — same bar the transformer descriptor already
passed.

---

## Appendix: original peer survey (2026-06-06, pre-v0.2)

Kept for the record — this table drove the v0.2/v0.3 scope. Every "✗ then"
cell is now ✓ except the two deliberate deferrals.

| Feature | llama.cpp | Ollama | mistral.rs | goinfer then | now |
|---|---|---|---|---|---|
| min-p sampling | ✓ | ✓ | ✓ | ✗ | ✓ |
| repetition/presence/frequency penalties | ✓ | ✓ | ✓ | ✗ | ✓ |
| logit bias / logprobs | ✓ | ✓ | ✓ | ✗ | ✓ |
| chat template from model metadata | ✓ | ✓ | ✓ | demo heuristic | ✓ |
| JSON **Schema** constrained output | ✓ | ✓ | ✓ | JSON only | ✓ |
| OpenAI-compatible HTTP server | ✓ | ✓ | ✓ | ✗ | ✓ |
| tool calling | ✓ | ✓ | ✓ (+MCP) | ✗ | ✓ |
| prefix/prompt caching | ✓ | ✓ | ✓ | `TruncateTo` only | ✓ |
| LoRA adapters | ✓ | ✓ | ✓ | ✗ | ✓ |
| K-quants beyond Q4_K_M | ✓ | ✓ | n/a | ✗ | ✓ |
| speculative decoding | ✓ (variants) | ✗→MTP | ✓ | ✓ greedy | ✓ greedy |
| continuous batching / PagedAttention | ✓ | ✓ | ✓ | ✗ | deferred |
| multimodal | ✓ | ✓ | ✓ | ✗ | deferred |

### Positioning (unchanged, still the filter)

The "Go LLM inference" field is purego/cgo *bindings* to llama.cpp
(gollama.cpp, yzma) — nobody else is a real pure-Go runtime. goinfer wins by
being the best **embeddable Go library**: idiomatic API, single static
binary, parity-gated numerics, structured outputs straight into Go structs —
not by chasing kernel throughput.

### Survey sources

- 2026-06-07: codersera Local AI Runtime Update (May 2026 — Ollama 0.23/0.24,
  vLLM 0.21, llama.cpp MTP b9196+, MLX/M5, LM Studio 0.4.14);
  mistral.rs v0.7.0 release notes; Mellum2 announcement coverage.
- 2026-06-06: llama.cpp server/speculative docs; Ollama structured-outputs +
  streaming-tool blogs; mistral.rs README; gollama.cpp / yzma repos.
