# Spike: cgo-free native CUDA — does a hand-authored megakernel clear the WGSL wall?

> **Status: SPIKE, not a commitment. NOT SCHEDULED.** In the lineage of
> `task-turboquant-spike.md` / `gpu-assessment.md` — a research-risk item with a
> written go/no-go bar, executed on the box, findings appended here. Prompted by
> `gocudrv` (eitamring, MIT): a cgo-free CUDA *driver* API binding for Go (dlopen
> `libcuda.so.1` at runtime, `CGO_ENABLED=0`, embedded PTX + driver JIT, static
> binary). It's the existence proof that the option goinfer's roadmap shelved —
> native CUDA *without* cgo — is real.
>
> **Trigger (why not now):** the standing roadmap verdict is "don't fight on
> kernels; the lane is CPU + portability." Fire this ONLY if, after the WebGPU
> **buffer-coalescing** win (the queued 0.8.1 item), a **felt NVIDIA-desktop-speed
> gap still matters to a real adopter** AND you decide that gap is worth CUDA
> kernel R&D. Until both hold, this is a watch item.

## The question (one sentence)

Can a **cgo-free** native-CUDA path (gocudrv-style dlopen + embedded-PTX megakernel)
decode a dense residency model **meaningfully faster than goinfer's WebGPU ceiling**,
while keeping `CGO_ENABLED=0` and the single-static-binary property intact?

## Why it's worth asking (and why it might fail)

- **The wall is a kernel-expressiveness problem, not a binding problem.** WebGPU
  tops out (~90–100 tok/s int8 on the 2070 SUPER; 61%/71% of CUDA at equal quant,
  `gpu-assessment.md` §0.0) because **WGSL cannot express the single-dispatch
  megakernel** that is CUDA/Metal's edge. Native CUDA *can*. So the ceiling that's
  structural for WebGPU is not structural for CUDA.
- **gocudrv dissolves the constraint that forced WebGPU.** "Native CUDA = cgo =
  breaks the static-binary/no-toolkit identity" was the reason it was off-strategy.
  A dlopen-at-runtime driver binding (the yzma/purego approach, but for `libcuda`)
  keeps `CGO_ENABLED=0` and the 1.8 MB-static-binary story.
- **Why it might fail anyway (the honest risks):**
  1. **The binding is the easy 20%; the kernel is the 80%.** gocudrv gives you
     `cuLaunchKernel` cgo-free — not cuBLAS/cuDNN. Beating WebGPU needs a *good*
     hand-authored PTX GEMM/attention megakernel. Reaching cuBLAS-class by hand is
     the "years-tuned CUDA" advantage the roadmap conceded. This spike tests whether
     a *competent, not world-class* megakernel already beats the WGSL ceiling — a
     much lower bar than beating cuBLAS, and the only bar that matters here.
  2. **Per-call channel-hop tax.** gocudrv funnels every driver call through a
     `LockOSThread` worker via a channel (its correct fix for the CUDA-context-is-
     per-OS-thread footgun). That's a per-call cost in exactly the launch-heavy
     regime that already bit WebGPU (glue-serialization). A **megakernel design
     amortizes it** (one dispatch/layer/token) — which is the point — but the spike
     must confirm it, not assume it.
  3. **JIT cold-start** could hurt boot (the "instant one-file" story). Measure it.

## Deliberately minimal scope (resist all sprawl)

**One card (RTX 2070 SUPER), one family, one kernel.** Explicitly NOT:
cuBLAS/cuDNN bindings, all ~380 driver functions, CUDA Graphs, multi-GPU, prefill,
MoE/hybrid/MLA. The spike is dense **residency-eligible Qwen2/Llama decode only**
(the same `DecodeRunnerEligible` shape the WebGPU 51.7/89.7 numbers come from — so
it's an apples-to-apples swap of the backend, nothing else).

Build it from:
- **gocudrv (vendored, MIT)** or its approach — the surface needed is small and it
  already has it: dlopen `libcuda`, `Alloc[T]`, memcpy H2D/D2H, module-from-embedded-
  PTX + JIT, `cuLaunchKernel`, streams, **events** (the measurement — see below).
- **One hand-authored fused decode-layer megakernel** in `.cu` → PTX (compiled once
  by nvcc on the dev box via `go generate`), `go:embed`'d, JIT'd on the target. int4
  (W4A8) and/or int8 weights — match the WebGPU residency quant so the comparison is
  equal-quant. Correctness gate: greedy output **token-identical** to the CPU decode
  (the existing residency-parity bar).

## The GPU-op surface — what a CUDA backend must cover

goinfer uses **zero CUDA today** — the GPU backend is WebGPU (`goinfer/gpu` →
cogentcore/wgpu-native); aikit is CPU-only. So a CUDA path re-implements the
WebGPU surface. It splits in two, and the split is the whole strategic point:

**Layer A — driver / plumbing (gocudrv already covers this).** What
`gpu/backend.go`/`device.go`/`gpu.go` do via wgpu, 1:1 with the CUDA driver API:
device/context init (`cuCtxCreate`), VRAM alloc (`cuMemAlloc` — the weight upload
`UploadW4A8Packed`/`CreateBufferInit`), H↔D copy (`cuMemcpy*`), module-from-PTX +
JIT (`cuModuleLoadData`), launch (`cuLaunchKernel`), streams, **events** (the
timing), and sync. **All of this is in gocudrv's 106 functions.** The plumbing is
not the risk.

**Layer B — the compute kernels (must be hand-authored `.cu`/PTX).** The WGSL
kernels in `gpu/` that a CUDA backend would need to re-write:

| Kernel group | goinfer files | notes |
|---|---|---|
| **Quantized matmul (the hot path)** | `gemv_w4a8`, `gemv_w8a16`, W8A8, `gemm`, `gemv`, `gemm_rows` | int4×int8 / int8×int8 with goinfer's packing |
| Attention + QK-norm + RoPE | `attention.go`, `qknorm.go` | QKᵀ·softmax·V |
| Norm / activation glue | RMSNorm, residual, SwiGLU/GeGLU, `relu2` | the elementwise/reduce glue |
| quant / dequant | `quant.go` | |
| Mamba-2 SSM | `mamba.go`, `mamba_f16.go` | conv1d + selective scan + gated norm |
| MLA latent attention | `mla.go` | |
| MoE routing + experts | `moe.go`, `moe_w4a8.go` | |
| Vision (SigLIP/ViT) | `vision*.go` | separate path |
| Fused decode | `decodefuse`, `decodelayer`, `decoderunner`, `decodetoken_fused` | partial glue-fusion — the megakernel is the CUDA-only next step |

**cuBLAS/cuDNN would NOT save you here.** goinfer's hot path is *quantized integer*
matmul (int4-nibble-packed W4A8, int8 W8A8); cuBLAS does f32/f16 GEMM, not goinfer's
packing. So even *with* cuBLAS bindings you still hand-author the quant GEMV/GEMM +
all the glue + Mamba/MLA/MoE. This is the concrete form of "the binding is the easy
20%, the kernel is the 80%": Layer A is done, Layer B — and specifically the quant
kernels that are the actual bottleneck — is the real work.

**But the spike needs only a slice of Layer B.** Dense residency decode = one fused
kernel of `gemv_w4a8` (or f32/int8 GEMV) + RMSNorm + RoPE + attention + SwiGLU.
**Not** Mamba, MLA, MoE, or vision — those are out of the spike's dense-only scope.
That single fused layer *is* the megakernel WGSL can't express, and the only kernel
this spike writes.

## Methodology (the measurement is the deliverable)

Learn from gocudrv's own 18× lesson: **CPU clocks measure enqueue, not kernel.**
Time with **CUDA events + a sync**, warm (drop first-run JIT + driver warmup),
steady-state over ≥3 runs. Report:

- decode tok/s (CUDA-event-timed), **equal quant, same 2070 SUPER**, vs:
  - goinfer **WebGPU residency** (the incumbent to beat): 51.7 tok/s int4 / 89.7 int8
    (`benchmarks.md §B`);
  - **llama.cpp-CUDA** (the ceiling reference): 72.8 tok/s (7B int4).
- cold-start: JIT + first-token wall (does it break the boot story?).
- the channel-hop tax: per-dispatch overhead at the megakernel's dispatch count.
- confirm end-to-end **`CGO_ENABLED=0`** and a single static binary that runs on a
  driver-only box (the whole point).

## Go / no-go bar

- **GO** ⇒ the cgo-free CUDA megakernel is **clearly past the WGSL ceiling** — as a
  starting bar, **≥1.3× goinfer's WebGPU number at equal quant** (i.e. it's doing
  something WebGPU structurally can't), **AND** `CGO_ENABLED=0` + static-binary hold,
  **AND** cold-start doesn't wreck boot. Bonus signal: within ~85% of llama.cpp-CUDA.
  ⇒ promote to a scoped "cgo-free CUDA residency backend" track (still dense-only).
- **NO-GO** ⇒ the hand-kernel + channel-hop land it **≤ WebGPU**, or it can't stay
  cgo-free in practice, or JIT cold-start is unacceptable. ⇒ close the item; WebGPU +
  buffer-coalescing remains the NVIDIA story, and the roadmap's "don't fight on
  kernels" verdict stands, now *measured* rather than assumed.

Timebox: a long weekend of kernel + integration + measurement. If it's not clearly
GO by then, it's NO-GO — a competent megakernel that only ties WebGPU means the
juice isn't worth the CUDA-kernel maintenance burden.

## Non-goals / what a GO does NOT commit to

A GO validates *one dense residency megakernel beats WebGPU cgo-free* — it does
**not** commit to cuBLAS-class perf, MoE/hybrid/MLA CUDA kernels, multi-GPU, or a
llama.cpp-beating engine. Those are separate, much larger, and each would be its own
decision. The spike answers exactly one thing: is the *cgo-free-native-CUDA lane*
open at all.

## Execution + provenance

Runs **on the box** (2070 SUPER) — it's a hardware measurement; it cannot be run on
a GPU-less CI/dev machine. When triggered, a short prompt points the box agent at
this doc. Findings (the tok/s table, the cold-start number, the channel-hop measure,
GO/NO-GO) get **appended here**, same discipline as the naga-probe and v29 write-ups:
no performance claim without a CUDA-event-timed run, versioned (driver + card +
commit).

## Relationship to the other GPU items

- **Buffer-coalescing (0.8.1, WebGPU)** comes first and is unrelated — it's the
  cheap in-lane win. Do it regardless; it also sharpens the "number to beat" here.
- This spike is the **only** path that could move the *decode ratio* itself (WebGPU
  is walled). It is off the default roadmap and gated on real demand — keep it that
  way until an adopter makes NVIDIA-desktop speed a priority.

---

## Phase-2 execution log (2026-07-14, RTX 2070 SUPER, driver 595.58.03)

Ran on the box per the contract. **Interim feasibility result — NOT a measured GO/NO-GO.**
Per this doc's own discipline ("no performance claim without a CUDA-event-timed run"), no
decode-perf verdict is claimed: the megakernel was not yet compiled + measured, so the GO
condition (measured ≥1.3× fresh-WebGPU) is **unproven**, which by the timebox rule defaults
to **NO-GO-pending** — distinct from a measured "it ties/loses." What *was* established:

### Step 0 — baseline re-measured (the number the GO bar keys off)
`TestDecodeRealModel_throughput`, qwen2.5-coder-1.5b, resident int8, best of 6×48-tok greedy,
this 2070 SUPER: **111.6 tok/s int8** (8.96 ms/tok). This is up from the stale §B 89.7 — the
buffer-coalescing win (`f8ef42b`/`5c3777f`) closed the native-CUDA gap **61% → 76%** (vs
Ollama-CUDA 147). `benchmarks.md §B` refreshed. **⇒ GO bar = ≥1.3× = 145 tok/s int8.**

### Feasibility triage (the hard blockers — all surmountable)
| gate | status |
|---|---|
| Card + driver + `libcuda.so.1` (the dlopen target) | ✅ 2070 SUPER, 595.58.03, 8 GB |
| **gocudrv** (cgo-free driver) | ✅ `github.com/eitamring/gocudrv@v0.2.0` fetched; rich generic API (`Alloc[T]`/`Buffer[T]`/`Arg[T]`/`LaunchConfig`, `Event.Record/Elapsed` for CUDA-event timing) |
| **Cooperative launch** | ✅ **PRESENT in v0.2.0** (`cuda/coop.go`: `LaunchCooperative`, `MaxCooperativeGridBlocks`) — **corrects spec §5.1's assumption that gocudrv lacks it.** The true single-launch/layer megakernel (route §5.1) is available, not just the 3-super-kernel fallback. |
| `ptxas` + `libnvvm` + `libdevice` | ✅ via `pip nvidia-cuda-nvcc-cu12` (ptxas-only wheel) |
| **`nvcc`/`cicc` (.cu→PTX frontend)** | ⚠️ not in the pip wheel and this clang has no NVPTX target; obtained via the CUDA 12.6.3 runfile `--extract` to `$HOME` (no sudo) — nvcc runs, but its `cicc` **cannot parse this box's gcc-15 libstdc++** (bf16 literal in `<type_traits>`). Fixable (older host headers via `-isystem`, or a CUDA ≥12.9 whose cicc supports gcc-15) but not resolved in-session. |

### Verdict status
The **cgo-free-native-CUDA lane is feasibility-confirmed**: every *hard* blocker cleared —
a cgo-free driver with cooperative launch exists, `libcuda` is present, the baseline is
established, and the toolchain is obtainable (the one wall, nvcc-vs-gcc-15, is a known
surmountable friction, not a fundamental one). **What remains is Layer B — the 80% the spec
itself calls out:** the 3 fused quant super-kernels (or, given coop launch is available, the
single cooperative megakernel) matching the exact W4A8/W8A8 packing + KV layout, the
gocudrv-backed `driver.go`, `BuildResident` weight extraction + JIT, the argmax-parity
correctness gate, and the CUDA-event decode measurement. That is the budgeted long-weekend
bulk and was **not** completed here. **No GO** is claimed (nothing measured); the lane is
open to *attempt* the build, and the number to beat is **145 tok/s int8**.

Provenance: goinfer commit `d1f85bd`, 2070 SUPER, driver 595.58.03, CUDA 12.6.85 (nvcc),
gocudrv v0.2.0.

### Phase-2 update — compiler unblocked + Layer A PROVEN end-to-end (2026-07-14)

**Compiler resolved (NVRTC, cgo-free).** The nvcc-vs-gcc-15 wall is real (CUDA 12.6's `cicc`
can't parse this box's gcc-15 libstdc++ — bf16 literals, `__is_array` builtins) and undefining
macros doesn't scale. The right tool is **NVRTC**: `libnvrtc.so` compiles CUDA C++→PTX with its
*own* headers (no host libstdc++), dlopen'd — *more* cgo-free than nvcc. Driven via ctypes here;
`megakernel.cu` scaffold and a verifiable `addone` kernel both compile to compute_75 PTX.

**Layer A — the cgo-free plumbing — is proven working on the 2070 SUPER** (`cuda/gocudrv_proof_test.go`,
`CGO_ENABLED=0 go test -tags cuda`): gocudrv v0.2.0 `Init` (dlopen `libcuda.so.1` + `cuInit`) →
`LoadModule`(NVRTC PTX) → `Alloc[float32]`/`CopyHtoD` → event-timed `LaunchOn` → `CopyDtoH` →
**correct GPU compute verified**. Measured (warm, best-of-8, CUDA-event):

| metric | value | note |
|---|---|---|
| JIT cold-start (`LoadModule` of embedded PTX) | **~0.16 ms** | does not wreck boot |
| per-dispatch CPU tax (gocudrv channel-hop + `cuLaunchKernel`, async burst = CPU-bound) | **~5 µs** (4.86–6.03) | the launch-heavy tax the megakernel amortizes |
| one dependent synced launch (idle GPU, incl. wake-up) | ~7–10 µs | noisy; not steady-state compute |

Both launch numbers are overhead-bound (trivial-kernel compute <1 µs) — the point. At ~13
dispatches/layer × 28 layers ≈ 364 dispatches/token, a ~5 µs/dispatch tax is ~1.8 ms/token of
*non-overlapped* CPU launch cost if serialized — exactly what collapsing to ~1–3 super-kernels/
layer (or the single cooperative megakernel, now that gocudrv exposes it) is meant to cut.

**Static-binary nuance (honest, matters for the GO criterion).** `CGO_ENABLED=0` **holds** — the
build needs no C toolchain and cross-compiles. But the binary is **not fully static**: purego
imports `dlopen` via dynamic linkage, so it links `libdl.so.2` + `libpthread.so.0` and dlopens
`libcuda.so.1` at runtime. So the property is precisely **"cgo-free + driver-only,"** not the pure
static-binary story — a real caveat against the spike's "1.8 MB static binary" framing.

**Still remaining (the decode verdict):** Layer B — the fused quant super-kernels matching the exact
W4A8/W8A8 packing + KV layout, `BuildResident` weight extraction + JIT, the argmax-parity gate, and
the CUDA-event **decode tok/s** measurement vs the 145 bar. That is the number that decides GO/NO-GO
and is not yet measured. Layer A being proven de-risks the "binding is the easy 20%" half; the
kernel is the 80% and the open question.

### Phase-2 update — hot-path quant GEMV hits 85% of peak (2026-07-14)

The decisive-proxy experiment (`cuda/gemv_bw_test.go`), since decode is weight-streaming-
bound and WebGPU sits below the bandwidth ceiling by its glue wall. A hand W8A8 GEMV
(`__dp4a` int8 dot, warp-per-row, matching the W8A8 packing), qwen2.5-1.5b FFN shape
[N=8960 × K=1536], NVRTC-compiled, launched via gocudrv, CUDA-event-timed:

| | value |
|---|---|
| correctness (cosine vs exact CPU dp4 ref) | **1.000000** — packing matches |
| kernel time | 36.3 µs (best of 8 × 50) |
| achieved bandwidth | **379 GB/s = 85% of the ~448 GB/s peak** |
| WebGPU decode, for contrast | ~37% of peak (111 tok/s ÷ ~300 tok/s bw-ceiling) |

**Read this precisely.** 85% is a *single GEMV in isolation* — back-to-back on the stream,
no inter-op dependencies, no attention/RoPE/norm glue, no per-dispatch launch tax. So it is
an **upper bound** on the fused megakernel, not the decode number. But it settles the spike's
sharpest risk — *"can a competent, not-world-class hand kernel beat the WGSL ceiling?"* — for
the hot path: **yes, decisively (85% vs 37%).** The 2.3× per-op headroom is exactly the
glue-serialization wall the megakernel exists to remove; the 1.3× GO bar (48% of peak) is
comfortably inside the 37%→85% gap.

**Signals so far (all GO-direction, none yet the verdict):** (1) cgo-free lane open — Layer A
proven end-to-end; (2) compiler solved (NVRTC); (3) the hot-path quant kernel is competent
(85% peak, correct). **What still gates GO:** the FUSED decode megakernel — GEMV + online-
softmax GQA attention + RoPE + RMSNorm + SwiGLU, with the real inter-stage dependencies and
the ~5 µs/dispatch tax — sustaining enough of that 85% ceiling across a full token to clear
145 tok/s. That end-to-end decode measurement is the remaining build and the only thing that
converts these signals into a measured GO.

### Phase-2 verdict — decode projection clears the GO bar 2.2× (2026-07-14)

The decisive measurement (`cuda/decode_projection_test.go`): the REAL per-token quantized-GEMV
workload of qwen2.5-1.5b (all 28 layers × {QKV, O, gate, up, down} + LM head = 141 GEMVs),
run as actual CUDA launches on the 2070 SUPER with the per-dispatch tax included, using the
cosine-1.0-validated W8A8 kernel. CUDA-event timed, best of 8×5:

| metric | value |
|---|---|
| per-token (weight-streaming bound) | **4.10 ms** |
| **decode tok/s** | **244** |
| sustained bandwidth | **374 GB/s = 83% of peak** (did NOT drop toward WebGPU's 37%) |
| vs WebGPU baseline | **2.2×** (112 tok/s) |
| vs GO bar (1.3× = 145) | **clears it by 68% — even UNFUSED** |

Why it holds at 83% across 141 chained launches: the 1.5B GEMVs (~13 MB each, ~36 µs GPU) are
large relative to the ~5 µs launch tax, so the CPU stays ahead of the GPU on the stream and the
tax overlaps — decode is GPU-bandwidth-bound, not launch-bound, at this model size. (The
megakernel's dispatch-collapse matters more for *small* models where the tax isn't hidden; here
plain CUDA quant GEMVs already win.) Honest caveats: this is the weight-streaming lower bound —
it omits attention *compute* (QKᵀ·softmax·V; cheap — ~0.3 M MACs/token vs the GEMVs' billions)
and the elementwise glue (RMSNorm/RoPE/SwiGLU/quant; cheap), and uses synthetic weights
(bandwidth is value-independent, standard). Real decode is slightly slower but the 2.2× margin
absorbs it comfortably.

## GO / NO-GO: **GO** (strong, measured — build the production megakernel)

Every gate the spike set came back positive, measured on the box:

1. **cgo-free lane open** — Layer A proven end-to-end (`CGO_ENABLED=0`, dlopen libcuda + NVRTC
   JIT + launch + CUDA-event + correct compute). Caveat: "cgo-free + driver-only," not fully
   static (purego dynamic-imports dlopen).
2. **cold-start fine** — JIT ~0.16 ms; does not wreck boot.
3. **kernel competent** — hand W8A8 GEMV correct (cosine 1.0) at **85% of peak** vs WebGPU's 37%.
4. **decode clears 1.3×** — the real-workload projection is **244 tok/s = 2.2× WebGPU**, past the
   145 bar even unfused, bandwidth sustained at 83% across the chain.

The spike's sharpest risk — *"a competent, not-world-class hand kernel can't beat the WGSL wall"*
— is **refuted with measurements**: the wall is glue-serialization (WebGPU stuck at 37% of peak),
and native CUDA quant GEMVs clear it structurally (83%). This flips the roadmap's "don't fight on
kernels, unmeasured" to **"the cgo-free-native-CUDA lane is real and worth a scoped backend track."**

**What a GO commits to (and does not):** promote to a scoped, dense-only "cgo-free CUDA residency
backend" track. It does NOT yet ship — the remaining engineering is the production megakernel
(the 3 super-kernels or the cooperative single-kernel, gocudrv v0.2.0 supports both), `BuildResident`
real-weight extraction, and the **argmax-parity correctness gate on a real checkpoint** (the perf is
proven; correctness of the full fused kernel is the build-out risk, not a viability risk). MoE / MLA /
Mamba / vision remain explicitly out of scope, each its own decision.

Provenance: goinfer commit `7e427c0`, RTX 2070 SUPER, driver 595.58.03, CUDA 12.6.85 (NVRTC),
gocudrv v0.2.0, `CGO_ENABLED=0`.

### Reality check vs Ollama — the projection is optimistic; recalibrated verdict (2026-07-14)

Comparing the 244 tok/s projection to the native-CUDA peer exposes projection optimism.
On the SAME 1.5B int8, **Ollama-CUDA = 147 tok/s** (`benchmarks.md §B`). So the projection is
1.66× Ollama — *not credible as a real result*: llama.cpp's CUDA kernels are mature, and a hand
GEMV is not 1.66× better than them. Reading it as bandwidth-fraction of the ~293 tok/s ceiling
(1.53 GB/token ÷ 448 GB/s) is clarifying:

| | tok/s | % of bandwidth ceiling |
|---|---|---|
| goinfer WebGPU | 111.6 | 38% |
| **Ollama-CUDA (real end-to-end native)** | **147** | **50%** |
| my GEMV projection (ideal streaming) | 244 | 83% |

Ollama — *real* native CUDA, end-to-end — sustains **50%** of the ceiling on this small model;
the remaining 50% is the real per-token cost my GEMV-only projection OMITS (attention compute,
the elementwise glue, activation requant, q8_0 dequant, sampling, sync). My 83% is the
weight-streaming *ceiling*, not achievable end-to-end — a hand megakernel carries those same
overheads and is **not** more optimized than llama.cpp, so its realistic landing is **near
Ollama (~147), not 244.**

**Recalibrated verdict — GO, but MARGINAL (not the 2.2× the projection implied).** The credible
range for a cgo-free CUDA megakernel is **[~147 Ollama-parity, ~244 ideal] ⇒ ~1.3–1.6× WebGPU**,
with the honest expected value near the low end (Ollama-parity ~1.3×). Even the floor clears the
1.3× GO bar — but *only just*, and the bar was set at exactly the point the spike says isn't worth
it if merely tied. So the sharper reading:

- **The lane is real and native CUDA does beat WebGPU here — but by ~1.3× (Ollama proves it), not 2.2×.**
- The prize for the whole cgo-free CUDA megakernel track is *matching native CUDA* (~1.3× WebGPU),
  weighed against the CUDA-kernel maintenance burden the roadmap wanted to avoid. That is a genuine
  GO on *viability* but a much closer call on *worth it* than the raw projection suggested.
- The earlier "2.2×" headline is the ideal-streaming ceiling and should be read as such; the
  Ollama-anchored ~1.3× is the number to plan against.

This does not change gates 1–3 (cgo-free lane open, cold-start fine, kernel competent — all still
measured and positive); it corrects gate 4's magnitude: decode clears 1.3× (real, Ollama-anchored),
not 2.2× (projection-optimistic).

### Phase-2 FINAL — real end-to-end decode measured; verdict PAUSE (2026-07-14)

The GEMV projection (244) and the ~1.3× inference are now **superseded by a measured end-to-end
number** (`cuda/e2e_decode_test.go`): the full per-token work — GEMVs PLUS the glue the projection
dropped (RMSNorm+quant, RoPE, GQA online-softmax attention, SwiGLU+quant, residual, argmax), the
non-trivial kernels cosine-validated vs CPU refs. Shippable config confirmed: **(1)** PTX compiled
offline (NVRTC) + `go:embed`'d + **driver-JIT'd** — `ldd`/strings show **no libnvrtc/libcuda** in
the binary (dlopen'd at runtime); **(2)** every launch goes through gocudrv's `LockOSThread`
executor channel — its hop is in the number; **(3)** `CGO_ENABLED=0`. CUDA-event, warm, best-of-8.

| | tok/s | % of ~293 ceiling |
|---|---|---|
| WebGPU (incumbent) | 111.6 | 38% |
| Ollama-CUDA (native peer) | 147 | 50% |
| **cgo-free CUDA E2E (this)** | **176** | **61%** |
| — GEMV-only projection (for contrast) | 244 | 83% |

**Breakdown:** GEMV 4.06 ms (72%) · glue+attention+argmax 1.61 ms (28%) · total 5.67 ms. The 28%
glue is exactly the cost the 83%→61% drop represents — the projection's omission, now paid.

**Anchored: 1.58× WebGPU, 1.20× Ollama.** Read the Ollama edge with discipline: it is within the
**measurement's optimism** — this omits the per-token KV-store, uses pure int8 (fewer bytes than
q8_0), is greedy with no sampler, runs a small pos=128 attention, and the Ollama 147 anchor is
unpinned. Beating mature llama.cpp CUDA with simple hand kernels is more likely optimism than a
structural win. So the honest reading is **~parity with native CUDA, and a solid ~1.5× over WebGPU.**

## Verdict: **PAUSE** (measured, supersedes the earlier GO)

- **NO-GO is refuted:** the glue + channel-hop did **not** eat the win — E2E is clearly above WebGPU
  (1.58×) and above the 145 bar. The cgo-free CUDA lane genuinely beats the WGSL ceiling end-to-end.
- **But it does not clearly beat native CUDA:** it *matches* Ollama (~parity after discounting). The
  prize for the whole cgo-free-CUDA-megakernel track is therefore **parity with llama.cpp, cgo-free**
  — a real capability (native-CUDA speed with `CGO_ENABLED=0`, no toolkit, driver-only), but **not**
  outperforming it.
- Whether "match native CUDA, cgo-free" is worth the **permanent CUDA-kernel maintenance burden** the
  roadmap wanted to avoid is a **maintainer business/judgment call, not an engineering step.** Park it
  as a **measured, documented option**: the lane is proven open and ~native-CUDA-fast; pull the trigger
  only if a real adopter makes cgo-free NVIDIA-desktop speed a priority worth the ongoing kernel cost.

This measurement is the number the track's future rests on: **~1.5× WebGPU, ~native-CUDA parity,
cgo-free** — GO on *viability*, PAUSE on *worth-it*. Provenance: commit `886b4ed`, RTX 2070 SUPER,
driver 595.58.03, CUDA 12.6.85 (NVRTC, offline), gocudrv v0.2.0, `CGO_ENABLED=0`.

### Phase-2 (A) — fresh apples-to-apples vs Ollama on THIS box (2026-07-14)

Re-anchored the peer instead of trusting the unpinned 147. Installed **Ollama v0.5.7** (prebuilt
CUDA runners, no rebuild), imported our **exact** `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`,
measured its decode `eval rate` on the 2070 SUPER:

- **Ollama-CUDA int4 (q4_k_m): ~149 tok/s** (152.6 / 146.0 / 148.3) — **confirms the 147 anchor was
  accurate, not stale-optimistic.** So the "we might actually be slower than a fresh, tuned
  llama.cpp" risk is testable — and refuted below.

Matching-quant (int4) on our side: `gemv_w4a8` (spec §3 nibble packing, cosine-1.0 validated) +
the same glue, full per-token loop, same shippable config (driver-JIT, executor hop, CGO_ENABLED=0):

| int4, same box, same GGUF/shape | tok/s | note |
|---|---|---|
| Ollama-CUDA (full decode, real tokens) | 149 | native peer, its runtime overhead included |
| **our cgo-free CUDA (W4A8 bare forward)** | **190** | **1.27× Ollama** |
| our cgo-free CUDA int8 (W8A8) | 176 | for reference |

**Honest reading of the 1.27×:** it is partly real and partly measurement asymmetry. Our number is
a bare per-token kernel loop; Ollama's `eval rate` includes its sampling/detokenize/server-loop
overhead our harness omits, and we run a fixed pos=128 attention vs Ollama's growing context, with
synthetic weights (perf-neutral, but no real-token correctness). Discount ~10–15% for those and it's
**~parity-to-modestly-ahead**, not a clean 1.27× beat. **The load-bearing finding is the direction:
we are NOT below native CUDA — at least parity — and our W4A8 is a NAIVE compute-bound kernel (43%
of peak); a tuned int4 unpack would be bandwidth-bound and widen the lead.** So there is real headroom
*above* Ollama, not below.

## Verdict update — perf resolves favorably; still PAUSE, but now on *worth-it* only

The fresh anchor removes the last perf uncertainty. Restated:
- **vs WebGPU:** clearly beats it — 190 int4 / 176 int8 vs 111.6 = **1.6–1.7×.**
- **vs native CUDA (Ollama, fresh, same box):** **at least parity, plausibly modestly ahead** (1.27×
  raw, ~1.1× discounted), with headroom from a tuned kernel. The earlier "matches but doesn't beat /
  might be below" is upgraded to **"matches or modestly beats, with headroom."**
- **So GO on viability AND perf.** The remaining PAUSE is **purely the business question** — is
  "native-CUDA-class decode, `CGO_ENABLED=0`, no toolkit, driver-only" worth the permanent
  CUDA-kernel maintenance? — not a doubt about whether it's fast enough. It is.

This supersedes the ~1.3× inference and the PAUSE-on-parity: the measured, same-box, same-model
comparison is **~native-CUDA parity-to-ahead, ~1.6× WebGPU, cgo-free.** Next (B, when funded): the
production backend + argmax-parity on a real checkpoint for the definitive end-to-end with correct
tokens. Provenance: commit `34a0818`, RTX 2070 SUPER, driver 595.58.03, Ollama v0.5.7, gocudrv v0.2.0.

---

## (B) Production backend — real-checkpoint parity + real e2e decode (steps 1–4 COMPLETE)

Task B ("done" = greedy output token-identical to CPU decode on a real 1.5B dense
checkpoint, then a real end-to-end tok/s), against the 7 guardrails. All met. This is the
line between "benchmarked" (A) and "shipped-quality backend" (B).

**Provenance:** RTX 2070 SUPER, driver 595.58.03, CGO_ENABLED=0, gocudrv v0.2.0 (dlopen
libcuda + driver-JIT), qwen2.5-coder-1.5b-instruct **q4_k_m** (real checkpoint), commit at
time of run 73ae0c8. Kernels: gemv_fwd.cu (gemv_w4a8_fwd / gemv_w8a8_fwd / kv_store) +
glue.cu, NVRTC→PTX→go:embed→cuModuleLoadDataEx.

### Step 1 — real-weight GEMV parity (foundation)
`TestRealWeightGemvParity`: real int4 GateProj [8960×1536] extracted via the BuildResident
seam (`m.Weights().Layers[0].GateProj → WeightMat.Int4()`), run through our W4A8 GEMV,
compared to goinfer's canonical dequant-matmul (`linalg.DequantizeRowInt4`). **cosine
1.000000.** Extraction + packing + kernel math correct on REAL weights.

### Steps 2–3 — full forward argmax-parity gate (the deliverable)
`TestRealForwardParity`: the full cgo-free per-token forward on the real checkpoint vs
goinfer's CPU decode, token-by-token. **8/10 positions token-identical**; the 2 flips are
genuine near-ties (CPU margin **0.08%** and **1.01%** of the logit range) — both far under
goinfer's own **3%-near-tie correctness rule** (gpu/kv_i8_parity_test.go), **0 hard fails**.
Same standard goinfer's WebGPU + int8-KV backends are gated by.

Real-checkpoint realities the synthetic benches hid, and the fixes:
- **Mixed precision.** q4_k_m stores tensors at different widths — layer-0 QProj came back
  **int8**, not int4. Added a forward W8A8 GEMV + per-weight kind dispatch, mirroring
  `gpu/residency.go:uploadProj` (int4/int8/f32). The activation layout (int8, 4/word) is
  identical across both; only the weight decode differs.
- **f32 group scales, not f16.** With exact int32 dp4a accumulation the per-group sums
  become bit-faithful to the CPU quantized matmul → moved 7/10 → 8/10 exact and shrank the
  residual flips to sub-1% near-ties. (f16 was a bandwidth trick; correctness wants f32.)
- qwen2's pre-MLP norm is **PreMLPNorm** (PostAttnNorm is nil for qwen2).
- On-device activation-scale threading (rmsnorm_quant writes aScale; GEMV reads
  *aScalePtr) + per-row QKV bias in the GEMV epilogue + separate q/k RoPE + kv_store.

### Step 4 — real end-to-end decode tok/s
`TestRealE2EDecode`: the parity-green path, driven **autoregressively** at real advancing
positions (growing KV), on-device argmax (argmax_reduce → 1 int back), per-token sync,
through a **LockOSThread-pinned CUDA executor fed by a channel (one round-trip per token)**
so the thread-safety executor cost is IN the number (guardrail #3). Wall-clock steady-state,
best of 6×16 warm tokens:

| backend (same box, q4_k_m, 1.5B dense) | tok/s | vs Ollama |
|---|---|---|
| **cgo-free CUDA (this, real weights, pinned executor)** | **183.5** (5.45 ms/tok) | **1.23×** |
| Ollama (native CUDA, pinned) | 149 | 1.00× |
| goinfer WebGPU | 111.6 | 0.75× |

Consistent with the (A) synthetic int4 e2e (190) but now on REAL mixed-precision weights
with the full correct forward + executor + on-device argmax + growing KV — slightly lower,
as expected (int8 layers are heavier than pure int4).

### Guardrails — all met
1. **Real-checkpoint argmax parity** ✓ (8/10 exact, rest sub-3% near-ties, 0 hard fails — not a synthetic bench).
2. **cgo-free verified, not assumed** ✓ — re-checked on the real-forward binary: CGO_ENABLED=0; `ldd` = libdl/libpthread/libc/vdso only (no libnvrtc/cudart/cublas/toolkit); loads via cuModuleLoadData(Ex), not nvrtcCompile.
3. **Executor cost in the number** ✓ — LockOSThread worker + channel, one round-trip/token, in the wall-clock.
4. **Dense residency only** ✓ — no MoE/MLA/Mamba/vision; opt-in `//go:build cuda` cuda/ submodule (own go.mod), not aikit.
5. **Backend-equivalence gate, no parity churn** ✓ — the CUDA-vs-CPU token gate lives in the cuda module; CPU forward-parity manifest untouched.
6. **Real e2e number** ✓ — full decode (attn + glue + requant + argmax + sync, real positions), vs same-box pinned Ollama 149 / WebGPU 111.6.
7. **Claim stays dense-qualified** ✓ — "dense-model GPU decode at native-CUDA-class speed, cgo-free" — NOT generalized to MoE/hybrid.

**Verdict:** B delivered. cgo-free, driver-only CUDA decodes a real dense 1.5B q4_k_m
checkpoint at **183.5 tok/s — 1.23× same-box Ollama, 1.64× our WebGPU** — with output that
matches goinfer's CPU decode under its own correctness rule. Kernels are still naive
(W4A8 nibble-unpack is compute-bound at 43% peak; no fusion) → headroom remains. Remaining
PAUSE is business/maintenance (a second GPU backend to own), not viability or perf.

---

## (B) Perf tuning — coalesced W4A8 GEMV: 43% → 71% peak, e2e 183.5 → 210.6 tok/s

Timeboxed kernel spike on the top lever identified post-B (the int4 GEMV was the slow one).
The isolated `w4a8_*` bandwidth tests (cosine-checked, % of ~448 GB/s peak) walked it:

| W4A8 GEMV variant | µs | % peak | note |
|---|---|---|---|
| naive (8-iter scalar nibble unpack) | 40.3 | 43% | baseline |
| FAST (even/odd mask + `__vsub4` SIMD unpack, same access) | 40.1 | 43% | **ALU was never the bottleneck** |
| V4 (uint4 group load, lane=group) | 35.1 | 49% | poor lane balance at K=1536 (48 groups/32 lanes) |
| **COAL (consecutive-word coalesced reads + 4-lane segmented reduction)** | **24.2** | **71%** | **winner, 1.67×** |

The real cap was the **strided read** (each lane read words at 16-byte stride → uncoalesced
warp loads), not the unpack. Making consecutive lanes read consecutive words (single 128B
transaction) and reducing each group's 4 words across a 4-lane segment took it to 71%. The
even/odd + `__vsub4` unpack rides on a static nibble-permuted layout (`permuteFast` at pack
time) so it pairs with the existing consecutive-activation packing.

Wired into the forward (`gemv_w4a8_fwd` rewritten coalesced, f32 scales + aScale-ptr + bias
kept). Re-measured on the real q4_k_m checkpoint:

| | before | after |
|---|---|---|
| real e2e decode | 183.5 tok/s | **210.6 tok/s** (+15%) |
| vs Ollama 149 | 1.23× | **1.41×** |
| vs WebGPU 111.6 | 1.64× | **1.89×** |

Parity gate re-run: still GREEN — 7/10 token-identical, **worst near-tie 0.194%** of logit
range (was 1.008% — flips got *tighter*), 0 hard fails. The 8→7 exact wobble is f32
summation-order in the segmented reduction, well within goinfer's 3% rule; the coalesced
kernel computes the same dot, just faster.

**Remaining headroom:** 71% vs W8A8's 85% — the per-word segmented-reduction shuffles are
the gap. Further levers (fusion, single-launch megakernel) are still untouched and were
found modest/high-effort elsewhere. Claim stays dense-qualified.

### The squeeze — ILP unroll: 71% → 80% peak, e2e 210.6 → 218.6 tok/s

Pushed the coalesced kernel further. The segmented-reduction shuffles turned out NOT to be
the gap (COAL2 dropped them via scale-per-word float-accumulate: 71% → 72%, a wash). The
real limiter was **memory-level parallelism**: int4 reads half the bytes of int8, so it
needs more loads in flight to saturate bandwidth. Extended ladder:

| variant | µs | % peak | |
|---|---|---|---|
| COAL (coalesced + seg-reduce) | 24.2 | 71% | |
| COAL2 (drop seg-reduce, scale-per-word) | 24.0 | 72% | shuffles weren't the cost |
| **COAL3 (COAL2 + 2× ILP unroll)** | **21.6** | **80%** | **winner — near W8A8's 85%** |
| COAL4 (4× unroll) | 23.8 | 73% | remainder penalty at K=1536 (192÷128 = 1 + big tail) |

2× is the sweet spot: clean for K=1536 (192÷64=3 exact) and near-clean for K=8960; 4× sends
half the K=1536 work through the slow 32-stride remainder. COAL3 wired into `gemv_w4a8_fwd`.

Real q4_k_m e2e: 210.6 → **218.6 tok/s** = **1.47× Ollama / 1.96× WebGPU**. Parity re-run
IMPROVED: **9/10 exact** (the 2×-unroll float order aligns better with CPU), worst near-tie
0.087%, 0 hard fails.

**Total W4A8 tuning: 43% → 80% peak (1.86× kernel), real e2e 183.5 → 218.6 tok/s (+19%),
1.23× → 1.47× Ollama.** Remaining 80→85% would need occupancy/vectorized-activation work —
diminishing returns. Fusion / single-launch megakernel still untouched. Dense-qualified.

### Reviewer verification — the three load-bearing facts, measured not asserted

1. **cgo-free (ldd on the actual binary).** `go test -tags cuda -c` → `go version -m` shows
   `CGO_ENABLED=0`; raw `ldd` = `linux-vdso / libdl / libpthread / libc / ld-linux` and
   NOTHING else — no libnvrtc/cudart/cublas/cudnn. `strings` = `cuModuleLoadData(Ex)` present,
   **0 nvrtc strings**, 19 embedded-PTX `.visible .entry`. Embedded PTX is driver-JIT'd; the
   driver is dlopen'd at runtime, not linked. The "driver-only, no toolkit" headline is true.
2. **218.6 is the executor-path number.** The timed loop routes every token through
   `do()` = `reqCh<-j; <-ackCh` to a `runtime.LockOSThread()`-pinned worker (the multi-goroutine
   path a real backend needs). Directly measured executor tax = **15.3 µs/token (0.34% of the
   4.55 ms token)** via 20k empty round-trips — it's IN the headline, not on top; the number
   does not shift when the channel is added because it was never absent.
3. **Equivalence is a committed hard gate.** `TestRealForwardParity` (git-tracked, commits
   73ae0c8/ba0085b) does `t.Fatalf` when any position flips > 3% of the logit range. Proved
   live: forcing the threshold 3%→0% trips the gate (`B GATE FAILED`, FAIL); reverted → green.
   It cannot silently regress — CI runs it.

---

## Headroom audit — the mandatory 0.5B baseline (MEASURED) + an OOB bug it surfaced

Executing the Fable headroom audit's mandatory first step ("every 0.5B tok/s figure is
speculative until the box runs"). RTX 2070 SUPER, driver 595.58.03, real
qwen2.5-coder-**0.5b**-instruct-q4_k_m vs the 1.5b.

### A latent correctness bug the 0.5B immediately exposed (fixed)
The coalesced `gemv_w4a8_fwd` remainder loop assumed `Kwords % 32 == 0`. True for the 1.5B
(H=1536→192, I=8960→1120) — **false for the 0.5B: H=896 → Kwords=112, 112%32=16**. Lanes
16–31 read **past the row into the next one** on 5 of 7 projections/layer (qkv/o/gate/up);
the last row could fault. `packWeight` guards only `K%32` (896 passes), so BuildResident
would have accepted the model and emitted garbage. Fixed with a per-lane `wi < Kwords`
guard (safe: the scale-per-word float accumulate has no cross-lane dependency, so
out-of-range lanes contribute nothing). **0.5B parity after the fix: 10/10 exact, 0 hard
fails** (cleaner than the 1.5B's 9/10); 1.5B unregressed (9/10, 0 hard fails).

### Measured baseline (both models, same harness)

| | 0.5B (24L) | 1.5B (28L) |
|---|---|---|
| tok/s | **321–390** (typ. ~325) | **218.5** |
| wall / token | 2.57–3.07 ms | 4.58 ms |
| launches / token | **434** (18/layer + 2) | **506** |
| weight stream / token | **360 MB** (LM head 137 MB = **38%**) | **1053 MB** (LM head 22%) |
| GEMV @ 374 GB/s | **0.96 ms** | **2.82 ms** |
| CPU issue (async burst) | **2.56–2.62 ms** | **3.98 ms** |
| GPU span (CUDA events) | **2.58–2.64 ms** | **4.58 ms** |
| per-dispatch CPU tax | 4.7–5.9 µs | 7.9 µs |
| verdict | **AT THE CROSSOVER** | **GPU-bound** (CPU hides under) |

### What it confirms (audit was accurate to the point of eerie)
Audit estimates vs measured: weight stream 330–385 MB → **360**; LM-head share 27–38% →
**38%**; GEMV 0.9–1.0 ms → **0.96**; 0.5B tok/s 390–440 → **389 best / ~325 typical**;
per-dispatch ~5 µs → **4.7–5.9**; launches 435 → **434**.

**The thesis holds: fixed cost dominates the 0.5B.** It is only **~1.5–1.8× the 1.5B's
tok/s despite 3× fewer parameters** — matching the in-repo WebGPU precedent (0.5B 166 vs
1.5B ~87–111, "only ~2× faster despite 3× smaller"). Bytes alone predict ~1000 tok/s
(360 MB @ 374 GB/s = 0.96 ms); we measure ~325–390. **The audit's sub-450 test is met.**

### One refinement to the audit's model
For the 0.5B, CPU issue and GPU span track within **0.02 ms across every run** (2.56/2.58,
2.62/2.64, 2.56/2.58). That is not coincidence — it is the launch-starvation signature: when
the CPU cannot get ahead, the device's event span **stretches to match the feed rate** and
the GPU idles between kernels. So the 0.5B's "glue = 1.35 ms (58% of GPU)" is an **upper
bound that includes idle**, not 1.35 ms of device work. The 1.5B's **glue = 1.77 ms (39% of
GPU)** IS real device work (CPU 3.98 < GPU 4.58 → the queue genuinely backs up).
Consequence: unchanged direction, stronger case — fusion cuts the launch count (CPU side)
**and** the glue (GPU side), and the 0.5B needs both.

Corollary: the 0.5B oscillates across the crossover run-to-run (321→390 tok/s), so it is
**jitter-exposed** — which is the audit's argument for graphs/de-hop as jitter immunity,
independent of mean throughput.

### Claim 1 verified in-tree
Production `cuda/resident.go` step() does `stream.Synchronize` + a full
`CopyDtoH(logitsHost, logits)` = 151936×4 B = **608 KB every token** (resident.go:236/239),
and **`argmax_reduce` is never loaded in the production path** (it is compiled into
glue.ptx — 10 occurrences — but only the e2e harness loads it). The 218.6 headline uses
on-device argmax; the shipped path pays the D2H. **Caveat the audit understates:** the
`ResidentForward` contract returns logits because the decoder needs them for sampling,
constrained decode, and logprobs — so the argmax kernel can only serve a *greedy fast path*,
not replace the D2H generally. That is an interface addition, not a drop-in.

### Launch diet (audit lever #2): 18→13 launches/layer — BUILT, parity-safe

Two same-math fusions, no new numerics:
- **`rope_kv`**: fuse rope(q) + rope(k) + kv_store(k) + kv_store(v) → **1 launch (−3)**. Safe
  because each thread rotates its own (d, d+half) pair and then stores exactly those
  elements — there is no cross-thread dependency the launch barriers were providing.
- **accumulate epilogue**: `accum` flag on both forward GEMVs (`dst[n] += result`) absorbs
  the two `residual` launches (**−2**). Bit-identical (same operands/rounding, minus a
  round-trip through a temp); race-free — only lane 0 of a row's warp touches `dst[n]`, and
  the GEMV's input activation is never `x`. Drops the `oO`/`dO` temps entirely.

**Parity: unchanged.** 1.5B 7/8 exact @ worst 0.087% (identical to pre-fusion), 0.5B **8/8
exact @ 0.000%**, 0 hard fails, via the production `decoder.Load(cuda)→BuildResident→Forward`
gate. Demo streams coherent code on the fused path.

| | before | after | |
|---|---|---|---|
| **1.5B** tok/s | 218.5 | **227.4** | **+4.1%** (audit predicted +3–5% ✓) |
| launches/token | 506 | **366** | 13/layer + 2 |
| glue (real device work) | 1.77 ms | **1.60 ms** | |
| **0.5B** tok/s | 321–390 (typ ~325) | **408–453 (typ ~410)** | **+26%** (audit predicted +10–15%) |
| launches/token | 434 | **314** | |
| CPU issue | 2.56 ms | **1.69–1.90 ms** | −30% |
| GPU exec | 2.58 ms (CPU-gated) | **2.15–2.16 ms** (honest) | |
| verdict | AT THE CROSSOVER (jittery) | **GPU-bound (stable)** | |

**The headline result: the launch diet broke the 0.5B's CPU/GPU crossover.** CPU issue now
sits 0.26–0.46 ms *under* GPU exec, so the device is the limit again — and the run-to-run
jitter collapsed with it (GPU span is now rock-stable at 2.15–2.16 ms vs the CPU-gated
2.58–2.64 spread). The 0.5B beat the audit's +10–15% estimate precisely because breaking the
crossover compounds: it removes launches (CPU side) *and* real glue (GPU side) at once.

**Consequence for the ranked plan:** the audit's 0.5B lever #2 (gocudrv de-hop / CUDA
Graphs, "+15–25% standalone… removes the CPU wall") is **no longer indicated** — the CPU wall
is already gone, with headroom to spare. The next lever is the 3-super-kernel fusion (§5.2)
cutting the remaining **real** glue: 1.19 ms of the 0.5B's 2.15 ms GPU time (55%) and 1.60 ms
of the 1.5B's 4.42 ms (36%) is still non-GEMV device work.

### Smaller levers (audit §1 + §5): f16 scales, greedy argmax, and a find the audit missed

**f16 int4 group scales.** Scales are exactly 20% of the int4 stream (K/8 scale bytes vs K/2
weight bytes at group-32); f16 halves that. 1.5B **1053→971 MB/token (−7.8%), 227.4→236.2
tok/s (+3.9%)** — the audit's byte math and +4–5% estimate both dead on. 0.5B 360→338 MB
(−6.1%), +1.5% only, because its GEMV is just 43% of GPU time (byte cuts can't touch its 57%
glue). Parity (the audit's lowest-confidence claim) **holds**: 1.5B 7/8 @ 0.087% unchanged;
0.5B 7/8 @ 0.939% (one flip, was 8/8) — far under the 3% gate, 0 hard fails.

**Greedy on-device argmax** (`decoder.ResidentGreedy`). Safety first, since it touches the
shared decode loop: the "can I be replaced by a raw argmax?" decision lives in
`Sampler.ArgmaxEquivalent()`, next to every logit-touching param, so a future param can't
silently invalidate it; the loop also requires `LogitProcessor == nil`. Gated by
`TestGreedyFastPathIdentical` — Generate's tokens must be identical with and without it
(token-identical over 24 tokens, both models). `GOINFER_NO_GREEDY_FASTPATH` is the escape
hatch.

**The find the audit missed: the D2H was pageable, not byte-bound.** The 594 KB logits
readback measured **0.47 ms = ~1.26 GB/s** — nowhere near PCIe. It was staging through a
driver bounce buffer because `logitsHost` was ordinary Go memory. Switching to page-locked
host memory (`gc.AllocHost` + `HostBuffer.Slice()`, a zero-copy view, so `Forward` still
returns with no extra copy) fixes it at the source:

| 0.5B | pageable | **pinned** |
|---|---|---|
| logits path (sampling) | 390.2 | **449.4 tok/s (+15.2%)** |
| D2H | 0.47 ms (1.26 GB/s) | **~0.13 ms (~4.6 GB/s)** |
| greedy fast path | 479.5 | 476.1 (unchanged — never paid the D2H) |
| greedy's remaining edge | +22.9% | **+6.0%** |

This matters more than the argmax lever it partly subsumes: **pinning speeds the general
sampling path** (`serve` at temperature > 0, constrained decode, logprobs) where the greedy
fast path cannot help by construction. The audit attributed the D2H cost to bytes; the real
cause was the allocation. 1.5B: logits path 232.8→236.2, greedy edge 3.2%→2.2%.

**Shipped-path scoreboard** (both levers + the launch diet, vs the pre-audit baseline):
| | pre-audit shipped | now (sampling) | now (greedy) |
|---|---|---|---|
| 1.5B | 232.8 | **236.2** | **241.5** |
| 0.5B | 390.2 | **449.4** | **476.1** |
