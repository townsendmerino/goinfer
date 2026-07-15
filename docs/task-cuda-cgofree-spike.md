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
