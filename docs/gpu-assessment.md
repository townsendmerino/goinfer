# GPU backend assessment — full-graph WebGPU: difficulty, ceiling, tradeoffs

> Internal planning doc (gitignored). Question under assessment: can a
> serious GPU effort make goinfer competitive with the Ollama/llama.cpp
> class, and what does it cost? Grounded in the current `gpu/` module
> (~750 lines) and `decoder.Backend` as of v0.3.0.
>
> **STATUS (2026-06-08): RESOLVED — GPU residency wins. 89.7 tok/s, 3.50×
> the CPU-glue hybrid, 10.6× CPU, ~52% of the CUDA ceiling.** Read §0.0
> first; §0 and §0.5 below are the (now-superseded) intermediate
> conclusions, kept as the investigation trail. Reading order: §0.0 → §0 →
> §0.5 → §3.

## 0.0 RESOLVED (2026-06-08): full-token GPU residency wins decisively

The decode question is closed in favor of **a full-token on-GPU forward**,
the opposite of §0's "staged hybrid optimal." §0 was a local optimum of an
*unfused, single-thread-attention* implementation; once the real bottleneck
was instrumented and fixed, residency wins by 3.3×.

Campaign on the 1.5B int8 `.giw` (RTX 2070 SUPER / 3700X), bit-exact at
every step:

| step | tok/s | GB/s | whole-token GPU |
|---|---|---|---|
| §1 single command-buffer (KV copies removed, one pass) | 22.4 | 34.7 | 41.9 ms |
| + rms+quant fused | 23.6 | 36.5 | 39.7 ms |
| + swiglu+quant fused (36 KB intermediate off the spine) | 27.0 | 41.8 | 34.7 ms |
| + residual→gemv epilogue | 27.1 | 42.0 | 34.7 ms |
| + **attn warp-per-head** | **84.5** | **130.8** | **9.7 ms** |
| + §4 share per-token uniforms | **89.7** | **138.8** | 9.7 ms (host 1.0) |

89.7 tok/s = **3.50× the staged hybrid (25.6)**, 10.6× CPU, ~52% of the
CUDA ceiling, 40% of the streaming roofline. Commits `fbdb71f`, `decfccd`,
`93d53c9`, `eaf9a6c` (main).

**The finding that corrects the §5 model: the serialization tax was
concentrated in ONE kernel, not spread across links.** The fusions
(rms/residual, bordering cheap 6 KB buffers) landed only ~1.2× combined;
swiglu+quant was the exception (+0.8×) because it kept a 36 KB intermediate
off the spine. The attn rewrite alone landed **3.1×**: the
`@workgroup_size(1)` kernel (12 single-thread workgroups) sat 28× on the
un-overlappable RAW spine and *was* the bottleneck. Warp-per-head (one
workgroup/head, 128 lanes, tree-reduced scores) dropped it 5.8 ms → 0.3 ms
and collapsed the whole-token GPU time 34.7 → 9.7 ms. **Lesson: find the
one kernel pinning the critical path before fusing anything — link-counting
mispredicted by ~3×.**

**The ceiling, now visible from the decomposition.** Token = 11.8 ms =
9.7 GPU + 2.1 host. GPU 9.7 ms = gemv **4.3 (at roofline, ~360 GB/s,
irreducible without W4A8)** + glue ~4.0 + attn 0.3. So:

- **§4 done (commit eaf9a6c): 89.7 tok/s, 138.8 GB/s, 40% roofline.** The
  per-token uniforms are layer-invariant (depend only on pos), so they were
  coalesced from ~112 buffers/writes to 4. Host `write` 0.3→0.0 ms, and the
  hidden per-buffer staging cost in submit fell too (submit+poll 10.1→9.8).
- **This CORRECTS the "host overhead is the lever" call: it wasn't.** §4
  removed essentially all the attackable host cost (write→0), yet we land at
  89.7, not >100. What remains is the ~1.0 ms `encode` (re-recording ~420
  dispatches via cgo — irreducible in WebGPU; compute command buffers are
  single-use, no reusable bundles) plus the 9.7 ms GPU. So the token-level
  **≥100 gate is blocked by the GPU glue wall, not host**: GPU alone is
  ~103 tok/s (9.7 ms), and with the irreducible encode the token floors near
  ~93 tok/s. Crossing 100 needs cutting the 9.7 ms GPU itself.
- The **WebGPU decode wall**: GPU 9.7 ms = gemv **4.3 (at roofline)** + glue
  **4.0** + barriers ~1.4. The 4.0 ms glue is residual per-link serialization
  over small (6 KB) buffers — §5 + this campaign prove it is only marginally
  reducible without a single-dispatch megakernel (persistent-thread, whole
  layer in one dispatch) that **WGSL cannot express**. That megakernel is the
  native-CUDA/Metal advantage and the honest cap on the pure-Go/WebGPU path;
  practical WebGPU decode tops out near **~90–100 tok/s token-level** on this
  card (the 4.3 + 4.0 split is the evidence — no grind needed to know it).

**Strategic verdict:** the bet is won. At 3.50× the prior hybrid, ~10.6× CPU,
52% of CUDA on one import with zero install and the same binary running
CPU-only elsewhere, GPU residency is the right architecture and the
embedding use case is served. The remaining ~10% to the 100 headline would
come only from cutting GPU glue (the megakernel wall) — not worth grinding;
W4A8 (fewer weight bytes → the only lever on the 4.3 ms gemv floor) is the
real future decode win. `dot4I8Packed` remains a *prefill* lever,
upstream-blocked.

## 0. Measured outcome (2026-06-08) — SUPERSEDED by §0.0

| Stage | Verdict |
|---|---|
| W8A8 WGSL kernel | bit-exact vs CPU |
| Stage-2 premise ("residency removes the latency") | **false** — coalescing dispatches was the real lever, not keeping activations resident |
| Coalesced GEMV (decode) | 1.83× CPU |
| Tiled GEMM (prefill) | ~3× CPU |
| Decoder wiring | 16/16 tokens match CPU |
| Fused-batch matmul groups (qkv / gate-up) | 1.58× vs separate dispatches — the sweet spot |
| Stage-3 full fusion (MLP entirely on-device, 1 sync, parity cosine 1.0) | **negative** — 2.75× CPU vs staged 4.0× CPU; ~8 elementwise-glue dispatches cost more than the one saved sync |

**Settled decode strategy:** batched matmul *groups* on GPU + microsecond
elementwise glue on CPU — the staged shape the decoder already runs —
delivering **4× CPU on the matmul-heavy FFN block at M=1**. Whole-layer
GPU residency is the wrong move at M=1; the original Stage-2/3 ladder
(§3) is therefore closed, not pending. This also retires the Stage-3
"attention+KV on-device" plan as written: the measured dispatch-overhead
economics say small-vector glue belongs on CPU.

**Remaining items — both outside kernel/fusion space:**

1. **E2E tok/s number** needs an int8-format model that fits 8 GB
   (q4 GGUFs bypass W8A8; Mellum2 too heavy). Packaging, not an unknown —
   the block-level 4× bounds it.
2. **`dot4I8Packed`** (prefill-compute boost) blocked on the
   `cogentcore/webgpu` binding exposing it — the §4 single-binding risk,
   realized exactly as predicted. Upstream when convenient.

## 0.5 REOPENED (2026-06-08, after hardware specifics): decode is NOT closed on this box

§0's measurements were taken on an **RTX 2070 SUPER (448 GB/s GDDR6,
Vulkan) vs a Ryzen 3700X (dual-ch DDR4-3200, ~51 GB/s peak / ~40 GB/s
streaming)**. Roofline: the decode ceiling on this pair is **~8–10× CPU**.
Measured 1.83× ⇒ the GPU GEMV path achieves ~60–90 GB/s effective of 448 —
roughly 20% of the card. This is a software gap, not physics. Two
artifact diagnoses:

1. **Sync tax.** Staged-with-CPU-glue costs a fence-wait + map round trip
   (~50–200µs through wgpu/Vulkan) at every glue point — several per
   layer × 30 layers ≈ milliseconds/token, vs ~3.3 ms/token of *total*
   kernel work for a 1.5B int8 at full bandwidth. The Stage-3 "fusion
   negative" result is best explained as per-dispatch submission/barrier
   overhead in how the backend drives wgpu: on NVIDIA Vulkan, 8 extra
   elementwise dispatches recorded in ONE command buffer cost tens of µs
   and cannot lose to one saved sync. The measurement was honest; it
   measured submission structure, not fusion economics.
2. **dp4a left on the table.** TU104 has native int8 dot (DP4A;
   `VK_KHR_shader_integer_dot_product`), which is exactly what
   `dot4I8Packed` lowers to. The blocked binding means scalar unpack math
   on hardware with int8 dot units.

**Probe results (2026-06-08, commit `141e6ce`, held):**

- **Syncs/token confirmed as the tax:** staged dense-1.5B path = 113
  syncs/token (4/layer × 28 + lm_head); MoE Mellum2 ≈ 757. Measured
  ~74µs/fence.
- **One-command-buffer probe** (367 dispatches, 152k vocab, single
  submit + fence, logits-only readback): 14.1 ms/token vs staged
  20.8 ms — **1.48× from fence elimination alone**, on a synthetic with
  no attention compute (delta valid; absolute tok/s not a decode rate).
- **Stage-3 "fusion negative" formally retracted** — the original test
  was per-block fusion with per-call allocation and ~92 syncs/token
  still present; it never tested the one-submit architecture.
- **CPU baseline correction:** q4 GGUF CPU decode on the 3700X runs at
  ~9.5 GB/s effective of ~45 available — **dequant-compute-bound, not
  bandwidth-bound**. Two implications: (a) every "Nx vs CPU" figure
  in this doc compares against a CPU path with its own ~4x headroom;
  (b) the W4A8-style kernel idea cuts both ways — fewer bytes AND less
  dequant work.
- **Two real device limits found + fixed en route** (predicted in §3
  Stage 1 risks): default 128 MB `maxStorageBufferBindingSize` < the
  233 MB int8 LM head → request adapter max (2 GB); 65,535
  `maxComputeWorkgroupsPerDimension` < a 152k-vocab GEMV grid → 2D
  dispatch. W8A8 parity stays bit-exact.

**Still open before any conclusion:**

1. **llama.cpp calibration — blocked on tooling** (no CUDA toolkit;
   Vulkan needs a source build). Cheap unblock for the CUDA *ceiling*
   number: **Ollama's prebuilt bundle ships its own CUDA runtime** — no
   toolkit needed; `ollama run --verbose` reports eval tok/s on the same
   q4 1.5B in minutes. The same-API Vulkan comparison still wants a
   llama.cpp source build (moderate yak-shave; worth it once).
2. **Real full-token on-GPU forward** (attention + KV resident, zero CPU
   interleave per token) — the probe's 1.48× floor plus the roofline
   says build it. **The E2E model-packaging blocker is already solved
   by existing tooling:** `cmd/prequant` turns the q4 1.5B GGUF into an
   int8 `.giw` (~1.7 GB — fits 8 GB with room for KV + logits), giving
   the W8A8 GPU path a real model without waiting on expert paging for
   Mellum2.

**Design requirement for the full-token build (MoE):** dense models keep
all dispatch shapes known at encode time, but an MoE layer's router
output chooses the expert matmuls — a CPU-side choice forces a mid-token
readback and destroys the one-submit architecture for exactly the models
(2.5B-active MoE) most attractive on 8 GB (the ~757 syncs/token Mellum2
count is this, structurally). The WebGPU answer is GPU-side top-k routing
+ `dispatchWorkgroupsIndirect`. Build the full-token pass dense-first,
but do not bake in "dispatch arguments known on CPU" — the indirect path
must be retrofittable. (Compute has no `GPURenderBundle` equivalent;
command buffers are single-use, so per-token re-encoding of ~367
dispatches is unavoidable — already inside the measured 14.1 ms and
acceptable.)

Speculative decoding's unpark condition (strongly bandwidth-bound decode)
is not yet met; revisit after the full-token build lands a real E2E
number.

## TL;DR

A full-graph WebGPU backend is a realistic **3–5 month** effort that buys
roughly **5–15x decode** and **10–30x prefill** on common GPUs, moves the
usable model class from ~1.5B to ~7–12B, and unparks speculative decoding.
It will **not** reach CUDA-llama.cpp throughput (don't aim there). The
positioning survives intact because the cgo stays quarantined in the opt-in
module and the user-facing surface is one flag that already exists. The
killer adjacent payoff: the same WGSL targets **browser WebGPU via wasm** —
"an LLM in a browser tab, pure Go" is a demo nobody else in the Go world
can ship. Biggest risk: WebGPU's buffer-size limits and the shader debugging
tail across three native APIs.

## 1. Where the GPU module actually is today

Honest inventory (`gpu/backend.go`, `gpu/gpu.go`, `decoder/backend.go`,
`decoder/weightmat.go`):

- `Backend` abstracts exactly one op: `MatmulBT(a, b, dst, M, K, N)` on
  **f32**. Registered for both decoder and aikit encoder; cgo confined to
  the `-tags gpu` submodule; clean CPU fallback on missing adapter or
  per-call error. Resident-weight caching already implemented (weights
  upload once, keyed by backing pointer).
- **The int8/int4/W8A8 kernels bypass the backend entirely**
  (`weightMat.matmul` → `linalg.MatmulBTW8A8/Q8/Q4` direct). Since every
  practical model runs quantized (`int8int8` default), `--backend webgpu`
  today accelerates only the f32 path nobody uses at size. The current GPU
  story is matmul-only AND f32-only — i.e., approximately nothing on a
  real workload.
- Every dispatch is synchronous with a full readback; activations never
  stay on-device. Per-layer: ~7 matmul round-trips × 30+ layers × per-token.
  Latency-bound by design; the doc comment in `backend.go` says as much.

So "do GPU stuff" is not an optimization of what exists — it's three
missing tiers: quantized GPU matmul, on-device activations, on-device
attention/KV.

## 2. Ceiling analysis — what's the maximum win?

Decode (M=1) is **memory-bandwidth-bound**: every token reads all resident
weights once. Effective bandwidths (order-of-magnitude, not vendor specs):

| Hardware | Effective BW | vs M-series CPU (~30–50 GB/s) |
|---|---|---|
| Apple M-series GPU (base→Max) | ~70–400 GB/s | 2–8x |
| Mid discrete (RTX 4060/Intel Arc) | ~270–450 GB/s | 6–10x |
| High discrete (4090-class) | ~1000 GB/s | 20x+ |

Realistic decode improvement after WebGPU overheads (dispatch latency,
no vendor-tuned kernels, f32 accumulation): **~5–15x** on discrete GPUs,
~2–6x on Apple base silicon. Concretely: the 0.5B from ~70 → several
hundred tok/s (uninteresting), the **1.5B from ~36 → 150–400**, a **7B q4
from ~8–10 → 50–120**, a **12B q4 from ~5 → 30–80**. The headline isn't
"faster small model," it's **the 7–12B class becoming interactive** — that
is the gap between goinfer-as-demo and goinfer-as-daily-tool.

Prefill (M=N) is compute-bound; GPUs have 50–100x the FLOPs. Realistic
**10–30x TTFT** on long prompts. This also changes the serve story (agent
loops re-sending long prompts) more than raw decode does.

Compounding effects once decode is bandwidth-bound on-device:

- **Speculative decoding unparks.** The CPU analysis ("M=K verify costs ~K
  decodes") inverts on a GPU: batched verify is nearly free vs sequential.
  The existing exact `GenerateSpeculative` + `TruncateTo` machinery ships
  as-is. Expect a further 1.5–2.5x on top.
- MTP-style heads become worth implementing for the same reason.

Hard ceiling honesty: llama.cpp CUDA on a 4090 runs a 7B q4 at ~150–200
tok/s with years of fused-kernel tuning. The realistic goinfer/WebGPU
number on identical hardware is maybe **40–60% of that**. That is
"competitive enough to be chosen for the embedding story," not "wins
benchmarks." Anyone selecting purely on tok/s keeps choosing llama.cpp.

## 3. Staged plan with difficulty

### Stage 1 — quantized matmul on GPU (the unlock) · 3–5 weeks · medium

Widen `Backend` (or add an optional `QuantBackend` interface, detected by
type assertion, so the CPU backend and existing API don't churn) to cover
the quantized kernels. **Scope honestly in two halves:**

- **1a — W8A8 (int8×int8): the Stage-1 deliverable.** Maps naturally onto
  WGSL's `dot4I8Packed`. This covers the `int8int8` default and everything
  loaded via prequant `.giw`. Weights upload once in quantized form (a 7B
  at int8 ≈ 7 GB resident instead of 28 GB f32 — also what makes big
  models *fit*).
- **1b — K-quants in-shader: a separate, harder sub-task.** Real GGUFs
  ship Q4_K_M/Q5_K/Q6_K — super-block layouts with d/dmin sub-scales, not
  plain int4. Unpacking Q4_K in WGSL is meaningfully harder than "int4
  unpacks in-shader"; budget it separately (1–2 weeks) or defer it: the
  CPU loader already dequant-requants GGUF into the resident int8/int4
  formats, so 1a alone covers GGUF-sourced models at int8. Direct K-quant
  shaders are a memory/bandwidth optimization, not a blocker.

Risks/gates:

- `maxStorageBufferBindingSize` is 128 MB–2 GB depending on adapter;
  large matrices need sharded bindings or split dispatches. Solvable, but
  real plumbing. Request adapter limits up front; fail loud with the
  limit report.
- **`dot4I8Packed` is an optional WGSL feature** — detect at adapter init
  and keep a non-packed int8 shader fallback (slower but correct);
  report which path is active in the startup line.
- Gate: per-kernel parity vs CPU kernels (existing quant tests as
  goldens, ~1e-4 accumulation tolerance) + a **matmul microbenchmark** showing the WGSL W8A8 kernel
  beats the CPU kernel at the relevant shapes (M=1 decode and M=64
  prefill, weights resident). **No end-to-end decode gate here** —
  Stage 1 alone is still readback-bound per matmul (~7 round-trips/layer),
  so E2E tok/s would measure dispatch overhead, not the kernel. The first
  E2E gate sits after Stage 2, which exists to remove exactly that
  overhead.

### Stage 2 — activations resident across a layer · ~~3–4 weeks~~ · **VERDICT: premise false (see §0)**

Backend ops take/return device-buffer handles; the per-layer chain
(qkv → attn out-proj → gate/up → down) runs without readback; one
readback per layer boundary at first, then per token. Requires a small
device-buffer arena keyed to the decode scratch.

- Payoff: removes most dispatch latency; gets to ~60–70% of the ceiling.

### Stage 3 — attention + KV on-device · ~~5–8 weeks~~ · **VERDICT: full fusion measured negative at M=1 (see §0); closed**

RMSNorm, RoPE, softmax, the KV append, and the attention matmuls in WGSL;
KV cache lives in GPU memory (`Session`/`TruncateTo` need device-side
equivalents — snapshot/restore copies through host). CPU keeps tokenizer,
sampler (logits read back per token — small), and orchestration.

- This is where the debugging tail lives: numerics drift per backend
  (Metal vs Vulkan vs DX12 fma/rounding), sliding-window masks, GQA
  head-mapping bugs that only show at layer 17. The Gemma-4-style
  layer-trace diff harness (`scripts/dump_*_trace.py` pattern) is the
  right tool — build the GPU equivalent early, not after the first
  garbage-argmax.
- Payoff: full ceiling; KV residency is also what makes long contexts not
  thrash the bus.

### Stage 4 — wasm/browser target · 2–3 weeks · medium (after 1–3)

cogentcore/webgpu targets browser WebGPU under wasm **without cgo**.
`demo/gemma-web` already exists as an HTTP demo; this turns it into
*client-side* inference. Marketing value is outsized; no one else in the
Go ecosystem can ship it. **Bill it honestly as "a 0.5B in a tab," not
the 7B story in a browser**: browser WebGPU support for the int8 packed
path (`dot4I8Packed`) is spotty today, so assume the browser build runs
the f32 (or fallback-int8) shaders on a small model; wasm memory caps and
model download size bound it anyway. Demo surface, not a product surface.

Total: **~3–5 months** of focused work, plus a permanent maintenance tax
(driver quirks, webgpu dep upgrades, a second numerics surface to gate).

## 4. Tradeoffs

| Tradeoff | Assessment |
|---|---|
| cgo in the GPU build | Already accepted and quarantined (`-tags gpu` submodule; CI guards the core graph). Default build stays pure — the moat is intact. **Non-negotiable to keep it this way.** |
| Two numerics surfaces | Every new model family now needs CPU parity AND GPU parity. Mitigate: GPU parity gates run vs CPU outputs (not HF), so the HF-golden burden doesn't double. |
| WebGPU vs native APIs | One shader set for Metal/Vulkan/DX12 is the only sane choice for a solo maintainer — but it leaves 2–3x on the table vs hand-tuned CUDA/Metal, forever. Accept this explicitly; do not start writing CUDA kernels later "just for the 4090 path." That way lies llama.cpp's headcount. |
| Apple Neural Accelerators (M5) | MLX-only; WebGPU-on-Metal can't reach them. Apple-silicon users get the bandwidth win, not the matrix-unit win. Acknowledge in docs. |
| Opportunity cost | 3–5 months not spent on model freshness / serve / library polish — the things currently winning. This is the real cost; see recommendation. |
| Maintenance tail | webgpu dep upgrades, driver regressions, adapter-limit edge cases filed by users with exotic GPUs. Budget ongoing time or hold the feature to "experimental" honestly. |
| Single-binding dependency | The whole bet rides on `cogentcore/webgpu` tracking the spec for what we need (`dot4I8Packed`, subgroups, timestamp queries for profiling). If it lags, we're upstreaming patches or blocked. De-risk in week 1 of Stage 1: spike-test that the binding exposes packed int8 dot and timestamp queries on all three native backends before committing the schedule. |

## 5. Ease for users / transparency

The UX is already designed and barely changes — this is a genuine strength:

- **Same flag**: `--backend webgpu` exists in demo + serve today; it just
  starts meaning something. Library: `decoder.Options.Backend = "webgpu"`.
- **Same binary semantics under `-tags gpu`**: no CUDA toolkit, no
  `LD_LIBRARY_PATH`, no Python — WebGPU talks to the OS's native API
  (Metal/Vulkan/DX12) directly. One prebuilt `-tags gpu` asset per
  platform alongside the pure ones.
- **Graceful degradation already implemented**: no adapter → CPU with an
  explanatory note; per-call GPU error → CPU fallback, correct results.
  Keep and extend: a startup line reporting adapter name, requested vs
  granted limits, and which stages (matmul / layer / attention) are
  GPU-resident, so "why is it slow" is always answerable.
- **Transparency policy** (write into README when Stage 1 ships): GPU
  numerics are parity-gated vs CPU per kernel but are NOT bit-identical
  to CPU (accumulation order); greedy argmax stability is gated on real
  models; any divergence beyond tolerance is a bug. This honesty is
  on-brand and exactly what the HN crowd rewards.
- Failure honesty: if granted buffer limits can't fit the model resident,
  say so at load and either shard or fall back — never silently run a
  hybrid that's slower than CPU.

## 6. Recommendation

Do it, but **sequence it behind v0.4's cheap wins** (model freshness +
serve polish are weeks, not months, and keep momentum) and **commit to the
positioning discipline**: WebGPU only, quarantined cgo, parity-gated, no
vendor-kernel arms race ever.

Go/no-go gates so this can't become a sunk-cost march (E2E gates start at
Stage 2 — gating Stage 1 on end-to-end decode would kill the effort for
the overhead Stage 2 exists to remove):

- **Week 1 of Stage 1 (binding spike)**: `cogentcore/webgpu` exposes
  `dot4I8Packed` + timestamp queries on Metal, Vulkan, and DX12. Miss →
  size the upstream patch before committing the schedule.
- **After Stage 1**: W8A8 kernel parity + microbenchmark beats the CPU
  kernel at M=1 and M=64 with resident weights. Miss → the kernel itself
  is the problem; stop and postmortem.
- **After Stages 1+2 (first E2E gate)**: ≥2x measured decode on the 1.5B
  on at least two of {Apple, NVIDIA-Vulkan, Windows-DX12}, and ≥5x on a
  discrete GPU / 7B int8 ≥ 40 tok/s on a 4060-class card. Partial miss →
  ship as "GPU-assisted," skip Stage 3.
- Stage 4 (wasm) is justified by marketing alone if Stages 1–2 hit.

Competitive framing to keep written down: the goal is not beating Ollama
on tok/s — it's removing the last reason a Go developer needs an Ollama
sidecar. At 40–60% of llama.cpp's GPU throughput with one import, zero
install, structured outputs into Go structs, and the same binary running
CPU-only on the build farm, the embedding use case is won. That's the
whole bet.
