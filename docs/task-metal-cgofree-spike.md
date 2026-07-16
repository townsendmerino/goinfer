# Spike: cgo-free native Metal — can purego-Obj-C + hand MSL kernels give Apple Silicon the CUDA treatment?

> **Status: SPIKE, not a commitment. NOT SCHEDULED.** In the lineage of
> `task-cuda-cgofree-spike.md` (executed 2026-07-14: GO on viability, PAUSE on
> worth-it) and `gpu-vendor-coverage.md`, where Apple/Metal is "the one to watch —
> highest audience, worst cgo-free fit." Prompted by the existence proof that
> weakens the "worst fit" half: **ebitengine/purego ships an `objc` package**
> (dlopen `/usr/lib/libobjc.A.dylib`, typed `objc_msgSend`, class registration,
> blocks — cgo-free), and **Ebitengine's production Metal driver runs on it**
> (hajimehoshi/ebiten #3411 converted the Metal graphicsdriver from cgo to purego).
> purego is already in goinfer's dependency tree (v0.10.0, the foundation under
> gocudrv). The "gnarly purego-Obj-C dance" is a shipped technique, not a
> hypothetical — the gocudrv moment, second vendor.
>
> **Trigger (why not now):** the standing verdict (`gpu-vendor-coverage.md`) is
> "add a native vendor path only if a concrete adopter's demand clears the bar
> CUDA did." Fire this ONLY if a real adopter makes Mac GPU decode a priority, or
> the maintainer decides the Mac-local-LLM audience justifies opening the second
> native lane. Until then, watch item — this doc exists so the decision, when it
> comes, starts from a written go/no-go instead of re-derivation under excitement.

## The question (one sentence)

Can a **cgo-free** native-Metal path (purego-`objc` binding + runtime-compiled MSL
kernels ported from the proven CUDA set) decode a dense residency model
**meaningfully faster than goinfer's incumbents on Apple Silicon** — WebGPU-on-Metal
if it proves viable on the rig, CPU otherwise — while keeping `CGO_ENABLED=0` and
the single-binary property intact?

## Why it's worth asking (and why it might fail)

**The risk profile is the CUDA spike inverted.** There, Layer A (the driver
binding) was pre-solved by gocudrv and Layer B (the kernels) was the open risk.
Here Layer B is largely de-risked — the six dense kernels exist, are
cosine-1.0-validated and perf-tuned in CUDA C (`gemv_w4a8_fwd`/`gemv_w8a8_fwd` in
`cuda/realforward_test.go`, `glue.cu`, PTX in `cuda/testdata/`), and
MSL is a C++ dialect, so the port is semi-mechanical — while **Layer A (the
binding) is the open risk**: there is no gocudrv-for-Metal; we would hand-roll the
~20-selector compute surface on purego/objc, with only render-side (Ebitengine)
precedent. `github.com/gogpu/wgpu/hal/metal` claims a pure-Go Metal HAL — inspect
it (and ebiten's converted driver) for proven msgSend patterns before writing any;
maturity unknown, treat as reference, not foundation.

**What dissolves the old objection.** `gpu-vendor-coverage.md` reasoned "no clean
`dlopen libmetal` ⇒ likely reintroduces cgo." That was the wrong lever: you don't
dlopen a C API, you dlopen the **Obj-C runtime** plus `Metal.framework` and drive
everything through `objc_msgSend` — exactly what purego/objc packages and Ebitengine
proves in production. (`MTLCreateSystemDefaultDevice` is a plain C export in
Metal.framework — `purego.Dlopen` + `RegisterFunc`, same as `cuInit`.)

**Apple-only structural sweeteners (things the CUDA lane never had):**

1. **UMA zero-copy residency.** Weights already arrive mmap'd (the `.giw`
   read-only-image story, `ARCHITECTURE.md` §2). On unified memory,
   `newBufferWithBytesNoCopy:` wraps page-aligned host memory as a GPU buffer with
   **no copy** — "residency upload" becomes a pointer wrap. The 1.23 s cold-start
   and 87 MB footprint stories survive GPU-ification largely intact.
2. **The compiler ships in the OS.** `newLibraryWithSource:options:error:` compiles
   MSL at runtime — the NVRTC role with zero downloads, zero toolkit, nothing
   embedded but `.metal` source text. The no-toolchain story is *cleaner* than
   CUDA's (no ptxas wheel, no driver-JIT-of-PTX dance).
3. **The cgo-status flip.** On macOS today, GPU decode = `-tags gpu` = **cgo**
   (wgpu-native, quarantined). A purego-Metal backend would be the **first
   `CGO_ENABLED=0` GPU path on the platform where the pure-Go identity matters
   most** — the §A rig itself. CUDA's win was speed at cgo-parity-or-better;
   Metal's would be speed *plus* identity.

**Why it might fail anyway (the honest risks):**

1. **The binding is the 80% this time.** purego/objc is the runtime, not a Metal
   binding; Ebitengine proves render, not compute — nobody has demonstrably driven
   compute encoders/dispatch through purego at production quality. Probably the
   same selectors, but unproven is unproven.
2. **Obj-C memory management by hand.** No ARC: every per-token `commandBuffer` /
   `computeCommandEncoder` returned by msgSend needs explicit
   autoreleasepool/release discipline or the decode loop leaks. A binding-
   correctness failure mode the CUDA lane didn't have (gocudrv's handles are
   plain uintptrs).
3. **Per-call tax, Metal edition.** objc_msgSend-via-purego trampolines + command
   buffer commit/`waitUntilCompleted` per token is the channel-hop-tax analog. The
   CUDA lesson says the tax hides when GEMVs are large relative to it (~5 µs there);
   Metal's encoder/commit overheads are different animals — **measure, don't
   assume.**
4. **No dp4a in MSL.** The int8 dot is manual (`char4` load, widen, mad). The CUDA
   tuning arc showed ALU was never the bottleneck (43%→80% of peak was all memory
   access), so on a bandwidth-bound GEMV this *should* be free — but it's an
   assumption to verify on Apple's ALUs, not carry over.
5. **The prize might be small.** M1 Pro memory bandwidth is 200 GB/s (shared).
   Arithmetic ceilings, not measurements: 1.5B int8 at ~1.55 GB/tok ⇒ **~129
   tok/s**; int4 roughly double that. The CUDA arc's finding is that mature native
   engines sustain ~50–60% of ceiling end-to-end — so expect Ollama-Metal somewhere
   near ~65–80 int8-equivalent, and the honest question is what's left between
   WebGPU-on-Metal (never measured — could be respectable) and that. The delta
   native buys must be worth a second native backend's permanent maintenance.
6. **ANE stays out of reach regardless.** Apple's matrix/neural units are
   MLX/CoreML-only (`gpu-assessment.md`); hand MSL buys bandwidth-class wins, not
   ANE-class. Nobody should oversell a GO as "MLX-class on Mac."
7. **⚠️ `CGO_ENABLED=0` silently downgrades the MSL compiler — the landmine.** Go
   binaries built `CGO_ENABLED=0` on macOS omit the **`LC_BUILD_VERSION`** Mach-O
   load command (upstream **golang/go#77917**). Without that deployment-target
   metadata, Metal's runtime compiler (`newLibraryWithSource:options:error:`)
   **defaults `MTLCompileOptions.languageVersion` to MSL 2.4**, which **silently
   strips modern types (e.g. `bfloat16`)** — kernels "compile" and run but produce
   wrong/degenerate output. This spike's exact config (`CGO_ENABLED=0` + runtime MSL)
   **will** hit it. **Fix (bake in before the first real kernel):** explicitly
   allocate an `MTLCompileOptions`, set `.languageVersion` to **MSL 3.1+** (rig is
   macOS 26 — match it to the features used), pass it to `newLibraryWithSource:`;
   **never rely on the default**. Add a startup assertion that the compiled library
   reports the version set, so a regression is **loud, not silent**. (A linker-side
   deployment-target `-ldflags` mitigation may also help other macOS runtime checks —
   secondary; the in-code `languageVersion` set is the primary fix.) Corroboration
   the lane is real: `hybridgroup/yzma` (pure-Go, purego, `CGO_ENABLED=0`, llama.cpp
   Metal on macOS) hit the same `LC_BUILD_VERSION` issue — it binds llama.cpp's *C
   API* so it's trap-confirmation, **not** a source for compute-encoder selectors
   (get those from ebiten #3411 / `gogpu/wgpu/hal/metal`).

## Deliberately minimal scope (resist all sprawl)

**One chip (the M1 Pro rig), one family, the proven kernel set.** Explicitly NOT:
MoE/MLA/Mamba/vision, prefill, batching, iOS, Intel Macs, MPS/MPSGraph bindings
(Apple's cuBLAS-analog — it doesn't speak goinfer's W4A8/W8A8 packing; same
"cuBLAS would not save you" logic as the CUDA spike), MLX interop, multi-queue.
Dense **residency-eligible Qwen2 decode only**, same `DecodeRunnerEligible` swap,
same `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` checkpoint as the CUDA arc — so all
three backends (CPU, WebGPU, Metal) are compared on identical work.

Build it as `metal/` mirroring `cuda/`: own go.mod, `//go:build metal` opt-in,
not aikit. Layers:

**Layer A — the purego-objc Metal binding (the risk, ~20 selectors).**
`MTLCreateSystemDefaultDevice` → `newCommandQueue` →
`newLibraryWithSource:options:error:` → `newFunctionWithName:` →
`newComputePipelineStateWithFunction:error:` → `newBufferWithBytesNoCopy:` /
`newBufferWithLength:` → per token: `commandBuffer` → `computeCommandEncoder` →
`setComputePipelineState:`/`setBuffer:offset:atIndex:`/`dispatchThreadgroups:` →
`endEncoding` → `commit` → `waitUntilCompleted`, timed by **`GPUStartTime`/
`GPUEndTime`** (the CUDA-events analog), plus NSString/NSError glue and the
autoreleasepool discipline. Attribution follows the gocudrv precedent: NOTICE +
THIRD_PARTY_LICENSES entries for purego and any prior art consulted
(ebiten Apache-2.0, gogpu BSD).

**Layer B — the MSL kernel ports (de-risked by the CUDA arc).** The six proven
kernels, same packing, carrying the CUDA lessons over as requirements, not
rediscoveries: coalesced consecutive loads with ILP (the COAL3 structure —
Apple GPUs also want wide contiguous loads), `simd_sum` reductions, **f32 group
scales** (the correctness lesson), per-weight int4/int8 kind dispatch (q4_k_m is
mixed-precision), on-device aScale threading, kv_store, GQA online-softmax
attention, on-device argmax. Correctness gate: the committed **argmax-parity bar**
— token-identical vs CPU decode under the 3%-near-tie rule, `t.Fatalf` on breach,
same as `cuda/realforward_test.go`.

## Methodology (the measurement is the deliverable)

GPU-timestamp-timed, warm, best-of-N, per the standing rule: **no performance
claim without a device-timed run,** versioned (macOS + chip + goinfer commit +
purego version + pinned peer version).

**Step 0 — the baselines, all fresh on the SAME M1 Pro (these are new information
regardless of verdict):**

- **WebGPU-on-Metal residency decode** (`-tags gpu`) — the incumbent. Never
  measured; every §B GPU number is from the 2070 SUPER. Viability itself is an
  open question this step answers.
- **Ollama-Metal**, same GGUF, version pinned (the native peer / ceiling
  reference — the CUDA arc's discipline of re-anchoring the peer on-box, learned
  from the stale-147 episode).
- CPU decode: §A's ~36 tok/s (1.5B int8) as the pure-Go floor.

**Then report:**

- decode tok/s, equal quant, vs all three baselines;
- cold start: runtime MSL compile + zero-copy buffer wrap → first token (does the
  0.48 s/1.23 s story survive?);
- the per-token binding tax: msgSend/trampoline + commit/wait overhead, isolated
  the way the CUDA spike isolated the ~5 µs channel-hop;
- zero-copy verification: no weight bytes duplicated (footprint measured via
  `phys_footprint`, the §A convention);
- **`CGO_ENABLED=0` verified, not assumed**: `go version -m` + `otool -L` shows no
  Metal/libobjc linkage (dlopen'd at runtime; the darwin caveat that pure-Go
  binaries link libSystem is the analog of the Linux libdl nuance — state it
  honestly, the property is "cgo-free + OS-only," not fully static).

## Go / no-go bar

- **GO** ⇒ **≥1.3× the fresh WebGPU-on-Metal number at equal quant** (if step 0
  finds WebGPU-on-Metal non-viable or degenerate on the rig, the bar does NOT
  collapse to "beats CPU" — it becomes **≥85% of same-box Ollama-Metal**, the
  native peer), **AND** `CGO_ENABLED=0` holds end-to-end, **AND** the
  argmax-parity gate is green on the real checkpoint, **AND** cold start stays
  within ~2× the CPU story. Bonus signal: Ollama-Metal parity (the CUDA result —
  1.47× same-box Ollama — says a competent hand kernel can reach and pass the
  mature peer).
- **NO-GO** ⇒ the binding can't stay cgo-free in practice (ARC/blocks/dispatch
  force cgo), or it lands ≤ WebGPU-on-Metal, or parity is unreachable, or the
  tax eats the win. ⇒ close the item; `gpu-vendor-coverage.md`'s verdict stands,
  now **measured** rather than assumed — and the step-0 WebGPU-on-Metal number is
  banked either way.

Timebox: **a long weekend** (the CUDA precedent), and the CUDA arc says most of it
goes to Layer A this time. Not clearly GO by then = NO-GO.

## What a GO does — and does not — commit to

The CUDA spike's ending is this spike's opening assumption: even a clean GO lands
at **"measured, documented option, gated on demand"** — a parked capability, not a
shipping commitment. A GO here validates that the *cgo-free-native-Metal lane is
open* and prices it; whether to build it out remains the maintainer's worth-it
call against a second permanent kernel set. The asymmetry vs CUDA worth naming in
that call: Mac is the **largest local-LLM audience**, and this would be the **only
cgo-free GPU path on it** — the upside case is speed *plus* identity, not speed
alone. MoE/MLA/Mamba/vision Metal kernels remain separate decisions, each gated.

## Execution + provenance

Runs **on the M1 Pro rig** — the primary dev machine, so no box agent required
(unlike the CUDA spike). Findings get **appended here**, same discipline as
`task-cuda-cgofree-spike.md`: the tok/s table, the tax measurement, the cgo
verification transcript, GO/NO-GO — versioned (macOS, chip, commit, purego,
pinned Ollama).

## Relationship to the other GPU items

- **`gpu-vendor-coverage.md` remains the strategic answer** (coverage is done;
  native is a per-vendor treadmill; gate on demand). This doc doesn't reopen the
  treadmill — it converts the one flagged watch item into a written, gated
  go/no-go, exactly what `task-cuda-cgofree-spike.md` was for NVIDIA. Its Apple
  row's "likely reintroduces cgo" is corrected by the purego-objc existence proof;
  the pointer there now leads here.
- **AMD/HIP stays the cheapest port with no trigger; Intel stays lowest; Vulkan
  stays skipped.** Nothing here changes that ordering.
- If GO, the build-out shape is pre-decided by precedent: `metal/` submodule
  mirroring `cuda/` (opt-in build tag, own go.mod, backend-equivalence gate in the
  submodule, dense-qualified claims only, NOTICE discipline).

---

## Findings — Step 0 baselines (appended per the execution discipline)

**Rig / provenance:** Apple **M1 Pro** (200 GB/s shared), macOS **26.5.2 (25F84)**,
goinfer commit **`c51219c`**, purego v0.10.0. Model **qwen2.5-coder-1.5b-instruct-q4_k_m.gguf**,
greedy, warm, int8 runtime (`Quant: int8int8`). Device-timed harnesses:
`gpu.TestDecodeRealModel_throughput` (WebGPU/Metal, best-of-6 × 48-tok) and
`decoder.BenchmarkDecode` (CPU, 48×).

| Baseline (1.5B, int8) | tok/s | ms/token | note |
|---|---|---|---|
| **WebGPU-on-Metal** (resident, `-tags gpu`) | **31.9** | 31.3 | viable (goes GPU-resident) but **degenerate** |
| **CPU** (pure-Go SIMD) | **32.66** | 30.6 | the pure-Go floor |
| **Ollama-Metal** (native peer, q4_k_m/int4) | **83.3** | 1.5s/128tok | ollama 0.32.0, same GGUF, greedy, best-of-3 warm |
| bandwidth ceiling (arith) | ~129 | — | 200 GB/s ÷ ~1.55 GB/tok; int4 ≈ 2× |

**Result: WebGPU-on-Metal is a WASH with CPU on the M1 Pro (0.98×), and both are
2.6× slower than the native peer** — the Mac GPU path buys **nothing** today. New
information regardless of verdict: every prior §B GPU number was the 2070 SUPER;
WebGPU-on-Metal had never been measured, and it turns out **not** to be "respectable."

**Quant caveat (honest):** Ollama runs the GGUF at native **q4_k_m (int4)**; goinfer's
WebGPU/CPU numbers are **int8** (the `BenchmarkDecode`/resident harnesses hardcode
int8, and a separate goinfer-int4 CPU number wasn't taken — both goinfer paths sit at
~32 regardless, so the 2.6× gap holds). The native-Metal spike targets q4_k_m/int4,
matching the peer.

**Implications — GO bar is now set:**
- WebGPU-on-Metal degenerate (wash with CPU) ⇒ per §Go/no-go the bar is **≥85% of
  Ollama-Metal = ≥85% × 83.3 ≈ 71 tok/s** (int4). Aspirational: Ollama parity (~83+),
  which the CUDA arc showed a hand kernel can reach (it hit 1.47× same-box Ollama).
- The **32 → 83 (peer) → ~129 (ceiling) gap is the headroom.** A native cgo-free Metal
  backend at ~71–83+ is a **~2.2–2.6× win** over the WebGPU/CPU wash, *plus* the
  identity win (first `CGO_ENABLED=0` GPU path on the platform where pure-Go matters
  most; today Mac GPU = `-tags gpu` = cgo).
- Net: Step 0 **strengthens** the case — the incumbent Mac GPU story is a dud and the
  native peer proves ~83 tok/s is reachable on this rig, so the target is concrete.

**Next:** Layer A — the purego-objc ~20-selector compute binding (the open risk),
proven first by a single-MSL-kernel compile-and-run before the six kernel ports.

## Findings — Layer A, Phase 1: the binding reaches Metal + the compiler, cgo-free

**Rig / provenance:** Apple M1 Pro, macOS 26.5.2 (25F84), commit `0a3541e`, purego
v0.10.0. Module `metal/` (`//go:build darwin`, own go.mod on purego), test
`TestLayerA_deviceAndCompiler`.

The two riskiest binding pieces are **retired, cgo-free, on the real rig:**

1. **Reach Metal cgo-free.** `purego.Dlopen(Metal.framework)` +
   `MTLCreateSystemDefaultDevice` via `RegisterLibFunc`, then `objc.Send` for the
   rest → device name reads **`"Apple M1 Pro"`**. The purego-objc msgSend dance works
   for the Metal compute surface, not just Ebitengine's render side.
2. **Drive the runtime MSL compiler past the landmine (risk #7).** `CompileLibrary`
   allocates an explicit `MTLCompileOptions`, sets `languageVersion` = MSL 3.1, and
   **asserts the read-back**. The gate kernel uses **`bfloat`** — an MSL-≥3.1-only
   type — so it compiling is *positive proof* the fix works: had the LC_BUILD_VERSION
   landmine pinned us to the default 2.4, `bfloat` would fail to compile. It compiles.

**cgo-free VERIFIED (not assumed):** `go version -m` → `CGO_ENABLED=0`; `otool -L` on
the test binary shows **no Metal / libobjc / Foundation** in the link table (all
resolved via runtime `dlopen`), only `/usr/lib/libSystem.B.dylib`. Honest caveat as
predicted: the property is **"cgo-free + OS-only," not fully static** (a pure-Go macOS
binary links libSystem regardless).

3. **Full compute dispatch — correct result, cgo-free** (`TestLayerA_vectorAddDispatch`).
   The complete path runs end to end: `newCommandQueue` → `newBufferWithBytes` (shared
   UMA) → `commandBuffer` → `computeCommandEncoder` → `setComputePipelineState:` /
   `setBuffer:offset:atIndex:` → `dispatchThreads:threadsPerThreadgroup:` → `endEncoding`
   → `commit` → `waitUntilCompleted`, then read the shared buffer back. **Vector-add of
   4096 floats on the GPU returns the correct result** (`out[4095]=12285`). Two hazards
   the doc flagged, resolved: **`MTLSize` struct-by-value** — at 24 bytes (>16) it's
   passed *by reference* per AAPCS64, so `unsafe.Pointer(&sz)` lands the pointer in the
   arg register exactly as the ABI wants; and **manual autoreleasepool** (no ARC) — the
   per-dispatch commandBuffer/encoder are drained via an `NSAutoreleasePool` so the
   decode loop won't leak (risk #2).

**Per-token binding tax** (`TestLayerA_bindingTax`, best-of-50 warm, wall):
- per-command-buffer round-trip (1 dispatch, commit + `waitUntilCompleted`): **~161 µs**
- marginal per-encoded-dispatch (msgSend to encode one more): **~3.8 µs**

This is the "different animal" the doc predicted: Metal's commit/`waitUntilCompleted`
round-trip (~161 µs) dwarfs CUDA's ~5 µs channel-hop — **but it's a per-COMMAND-BUFFER
cost, not per-dispatch.** Encode a whole token's layer stack into ONE command buffer and
commit once, and at the ~71 tok/s target (~14 ms/token) the 161 µs is **~1%**. The
marginal encode (~3.8 µs, CUDA-channel-hop-class) × dispatches/token is the real budget
line — modest, and fusion (fewer dispatches/layer, the CUDA megakernel lesson) shrinks
it. **Architectural requirement banked: one command buffer per token, one commit/wait —
never per layer** (per-layer commit would multiply 161 µs by layer count and blow the
budget).

**Binding-risk verdict: RETIRED — the cgo-free native-Metal *lane is open*.** The "80%
this time" was *can compute go through purego-objc cgo-free at all, correctly, at a
tolerable tax* — yes on all three, proven on the real rig. The binding does not force
cgo (the NO-GO condition) — the opposite.

## Findings — Layer B: the full decode-layer kernel set ported to MSL, all bit-exact

Every kernel a dense Qwen2/Llama decode layer needs is ported and validated vs a CPU
reference on qwen-shaped data, cgo-free — **all bit-exact / near-exact** (the CUDA arc's
"MSL is the semi-mechanical C++ dialect" promise held completely; packing + math carry
over):

| Kernel (test) | Result |
|---|---|
| **W8A8 GEMV** (`gemvW8A8Parity`) | cosine 1.0000000, max-rel 0 |
| **W4A8 GEMV** (`gemvW4A8Parity`) — the **target int4 quant** (q4_k_m); goinfer's exact packing (group=32, nibble=q+8, word k/8 @ bit 4·(k%8)), f32 group scales | cosine 1.0000000, max-rel 0 |
| **RMSNorm+quant** (`rmsnormQuantParity`) — fused norm→int8; threadgroup reductions | cosine 1.0000000, int8 exact 1536/1536 |
| **RoPE** (`ropeParity`) — NeoX half-split | cosine 1.000000000 |
| **SwiGLU+quant** (`swigluQuantParity`) — silu(g)·u → int8 | cosine 1.0000000, int8 exact 8960/8960 |
| **GQA online-softmax attention** (`attentionParity`) — causal, streamed softmax over the KV cache, GQA head mapping | cosine 1.0000000, maxAbs 8.9e-08 |

Kernels are correct-first (naive one-row/one-head-per-thread); the CUDA lesson is that
tuning (coalescing/ILP to ~80% peak) follows parity and is all memory-access, not ALU.
MSL gotcha banked: `half` is a reserved type (f16) — can't be a variable name.

**Full decode-layer assembly — bit-exact, one command buffer** (`TestLayerB_fullLayerForward`).
All kernels (+ the two trivial ones, `kv_store` / `residual`) chained into a complete
dense decode layer — rmsnorm+quant → q/k/v GEMV → rope(q,k) → kv_store → GQA attention →
quant → o-proj → residual → rmsnorm+quant → gate/up GEMV → swiglu+quant → down → residual
— encoded as **17 dispatches in ONE command buffer** (the tax requirement), validated vs a
CPU reference that mirrors the exact quantized path: **cosine 1.0000000, maxAbs 0.00e+00**
(literally identical). The serial compute encoder's automatic inter-dispatch barriers
sequence the dependent kernels correctly (the WGSL storage-barrier analog). So the whole
layer assembles and runs correctly, cgo-free, in the shape the decode loop needs.

## Findings — real-model decode: CORRECT + cgo-free, but untuned perf ties WebGPU (NO-GO as measured)

`metal/model.go` (`BuildResident` + `Forward`) loads the real **qwen2.5-coder-1.5b** int8
weights out of the goinfer decoder (`w.Int8()`, direct — no repack), handles Qwen2 q/k/v
bias, and runs the full **28-layer stack + LM head per token in ONE command buffer**,
cgo-free. `TestRealModel_parityAndThroughput`, on the M1 Pro:

- **Correctness — PROVEN.** Token-by-token argmax vs the decoder's own CPU forward
  (`m.ForwardForTest`, same int8 weights): **23/24 match**, logit cosine **0.998** (the 1
  miss a near-tie). A real 1.5B model decodes correctly through cgo-free native Metal —
  the first `CGO_ENABLED=0` GPU decode of a real model on Mac.
- **Stability bug found + fixed.** Intermittent SIGSEGV in `waitUntilCompleted` was the
  **NSAutoreleasePool thread-migration hazard** (doc risk #2): Go migrates the goroutine
  between `begin()` (pool push) and `end()` (pool drain), but an autorelease pool is
  per-OS-thread → draining on a different thread is UB. Fix: `runtime.LockOSThread` around
  `Forward` (the CUDA backend's LockOSThread-executor discipline). Crash gone.
- **Perf — the honest number.** Naive one-thread-per-row GEMV was **4 tok/s** (250 ms/tok)
  — pathologically *uncoalesced* (adjacent threads read far-apart memory, ~6 GB/s of the
  chip's 200). One coalescing pass (simdgroup-per-row + `simd_sum`, `gemv_w8a8_coal`) →
  **~30 tok/s (int8), 7.5×** — but with real measurement variance (**~18–30 tok/s** warm,
  worse under concurrent CPU load). That's **~0.6–0.9× WebGPU-on-Metal (31.9, int8)** and
  **~0.3–0.4× the ~71 GO bar** (int4-anchored).

**Verdict — NO-GO as measured; viability + correctness GREEN; a real but *projected* path
to GO.** Per the spike's discipline (no perf claim without a measured run, "not clearly GO
= NO-GO"): the current coalesced-but-untuned int8 kernel **ties WebGPU-on-Metal, it does
not clear it** — nowhere near the ≥1.3× that would justify a second native backend, let
alone the ≥71 bar. What the spike *did* prove, all measured: the **cgo-free native-Metal
lane is open and correct** (binding retired, every kernel bit-exact, full real-model decode
matches CPU). What it did **not** prove: that it's *worth it* — reaching the bar needs (a)
the rest of the CUDA tuning arc (coalescing done → ILP/multi-row/register-tuning, the
43%→80%-of-peak work) and (b) **int4/W4A8** (the target quant, ~2× less bandwidth than the
int8 measured here). Both are known and bounded from the CUDA precedent — but they are
**projections, not measurements**, so the honest verdict stands at **GO-on-viability,
NO-GO-on-worth-it-as-measured**, exactly the CUDA arc's ending, one tuning-arc earlier.

## Findings — W4A8 + ILP re-measure: the bottleneck is DISPATCH COUNT, not bandwidth

Swapped the GEMV to **int4/W4A8** (the target quant, half the weight bytes — re-quantized
via the validated packer) with a **coalesced + ILP-unrolled** kernel (`gemv_w4a8_coal`,
simdgroup-per-row + `simd_sum`, 8-way-unrolled nibble inner). Re-measured on the M1 Pro:

- Correct (int4 is lossier: **20/24 argmax** vs CPU, cosine 0.989 — still coherent).
- **Perf: ~19.5 tok/s — NO improvement over int8** (~18–30). Halving the weight bytes did
  nothing. **The decode is not weight-bandwidth-bound.**

**Root cause, proven two ways:** (1) int4 (½ the bytes) gave zero speedup; (2) measured
~28–51 ms/token is **7–13× the bandwidth floor** (int4 ≈3.75 ms, int8 ≈7.5 ms at 200 GB/s).
A layer-count sweep nails it: **~1 ms/layer, near-linear** (nL=4→6.2 ms, 28→28.6 ms) =
**~50 µs × the ~19 dispatches/layer of launch + inter-dispatch-barrier overhead** — not
matmul compute. The decode is **DISPATCH-BOUND** (~530 tiny serialized dispatches/token),
the Metal analog of the WebGPU glue-serialization wall.

**So the lever is KERNEL FUSION, not quant or GEMV-ILP.** Collapsing the ~19 dispatches/
layer into ~3–5 fused super-kernels (the megakernel lesson — on Metal via threadgroup-local
fusion, no cooperative launch needed) would cut the ~50 µs × 19 overhead ~4–5×, projecting
~6–7 ms/token ≈ **~140 tok/s**, comfortably past the ~71 bar. **But that is a projection —
the fused kernels are not built or measured**, so the verdict is unchanged:

**Final verdict — GO-on-viability, NO-GO-on-worth-it (as measured), now precisely diagnosed.**
The cgo-free native-Metal lane is open and correct (all measured); the untuned per-kernel
decode is ~20–35 tok/s, dispatch-bound, tying (not beating) WebGPU-on-Metal. Reaching GO is
a **kernel-fusion** effort — well-scoped from this diagnosis, the CUDA arc's megakernel
precedent, and the fully-in-place scaffolding (binding, all bit-exact kernels, the
correctness gate) — but it is future work, not a measured result. The honest number today
does not clear the bar.

**If revisited:** fuse the layer's ~19 dispatches into ~3–5 threadgroup-fused super-kernels
(pre-attn / attn / ffn), keeping each stage's already-bit-exact math; re-measure the
dispatch-overhead reduction vs the ~71 bar. Not a rebuild — the kernels and gate exist.
