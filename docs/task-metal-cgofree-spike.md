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

## Findings — the fusion FALSIFIED the dispatch-bound hypothesis (claim-discipline correction)

Built the fusion (`metal/model.go`): QKV combined into one GEMV (+bias fused), gate/up
combined into one, residual folded into the o-proj/down epilogues (`gemv_w4a8_bias` /
`gemv_w4a8_resid`, offset-bound sub-views) — **19 → 12 dispatches/layer**. Still correct
(21/24 argmax vs CPU, cosine 0.989). **Perf: ~20 tok/s — unchanged.**

**That disproves the "dispatch-bound" call above.** Cutting a third of the dispatches did
nothing, and int4 (½ the weight bytes) did nothing — so it is **neither dispatch- nor
bandwidth-bound**. Measured ~50 ms/token is ~15× *both* the compute floor (~1 ms, 1.5 G
int8-MACs/token) and the int8 bandwidth floor (~3 ms). The real bottleneck is **GEMV kernel
efficiency — memory-latency / low-occupancy bound at ~22% of peak**: the coalesced kernel
uses one 32-lane simdgroup per output row with a stride-4 read pattern (adjacent lanes hit
different cache lines) and too few concurrent memory requests to hide latency.

**Corrected verdict — GO-on-viability, NO-GO-on-worth-it (as measured), honestly diagnosed
after three falsified levers.** Viability + correctness are proven and measured; the
tuning sequence — coalesce (4→30 tok/s ✓), **int4 (no gain ✗), dispatch-fusion (no gain
✗)** — lands at ~20–30 tok/s, ~22% of peak, tying not beating WebGPU-on-Metal. The gap to
the ~71 bar (~53% of peak) is **an expert GEMV-kernel-optimization effort** — truly
coalesced (adjacent-lane→adjacent-word) loads, `uint4`-vectorized reads, register/thread
blocking for occupancy: the CUDA arc's 43%→80% grind, which took that arc many commits and
is a multi-session job, not a weekend. **The honest measured answer: the cgo-free
native-Metal lane is open and correct, but a competent-non-expert kernel does not clear the
bar, and closing it is real GPU-kernel engineering — bigger than this spike scoped.**

The scaffolding (binding, all bit-exact kernels, the fused decode, the correctness gate)
is committed and in place for whoever takes that on.

## Findings — profile-driven arc to 68 tok/s (0.96× bar): attention → Stage A → the real wall is dispatch overhead

The "NO-GO as measured" verdict above was written before we **profiled**. Profiling
(`metal/profile_test.go`, per-kernel µs at real dims, cache-resident) overturned the
"GEMV-efficiency-bound" diagnosis and drove a 3.4× decode gain. The honest arc:

1. **Attention was the hidden 68%.** Per-kernel profile showed the one-thread-per-head
   attention kernel at **1139 µs/dispatch** — not the GEMVs. Rewrote it threadgroup-per-head
   (128 threads: staged scores + parallel softmax + parallel value-accum): **1139 → 23 µs**.
   Decode 20 → 56 tok/s. *The earlier "GEMV-bound" call was wrong because it never profiled —
   the profile-first instinct was correct.*

2. **Coalesced the W4A8 GEMV** (stride-4 → word-per-lane, adjacent lane → adjacent word):
   gate/up 254 → 220 µs. 56 → 58.8 tok/s.

3. **Fable Stage A (no repack):** `uint4` loads (one 128-bit load = one 32-element scale
   group) + int8 activation **staged once into threadgroup `short`** (kills the per-row
   device byte-gather — the activation was re-read 17920× cold) + 8 simdgroups/tg. gate/up
   211 → 164 µs isolated; **decode 58.8 → 68.9 tok/s (0.97× bar), parity 21/24.** The win
   was real because staging removed redundant *activation* DRAM traffic.

4. **Fused block-argmax lm head** (Fable): per-tile (maxLogit,rowIdx) → `argmax_finish` →
   4-byte token, never materializing 151936 logits. **Bit-exact (24/24 vs argmax(logits))**,
   but **throughput-NEUTRAL on UMA** — Fable priced the 608 KB readback at 0.2–0.4 ms
   assuming a discrete-GPU copy, but `logits.Floats()` is a zero-copy shared view (~30 µs
   memcpy). Kept as the correct production decode API (`ForwardArgmax → token`).

5. **Fable Stage B (tile-repack, lane-per-row, broadcast reads):** genuinely faster in
   isolation — gate/up **164 → 118 µs** (in Fable's 95–115 range). But **decode stayed flat
   (~68 tok/s) and cost a parity point (21 → 20).** Reverted. This is the pivotal negative
   result:

**The wall is no longer the GEMV.** Aggregate weight traffic is ~772 MB/token; at ~14.8
ms that is **~52 GB/s ≈ 26% of the M1's ~200 GB/s** — decode is neither compute- nor
bandwidth-bound. A 46 µs/kernel isolated win (Stage B) buying **zero** end-to-end proves it:
decode is **dispatch/encode-overhead-bound** — ~337 serial dispatches/token, each with
Go→`objc_msgSend` encode + Metal per-dispatch scheduling latency. Summing the (cache-resident)
profile gives ~12.4 ms of kernel; measured is ~14.8 ms → **~2.4 ms (~16%) is per-token
command-buffer overhead**, matching Fable's independent estimate.

**Updated verdict — GO on viability, near-bar on perf, honestly bottlenecked.** cgo-free
native-Metal decode is **68 tok/s (0.96× the ~71 bar, 2.1× WebGPU-on-Metal, 3.4× the
untuned spike), correct (21/24 argmax, cosine 0.989)**. Kernel optimization has run its
course for single-stream decode; the remaining lever is **cutting per-token dispatch
overhead** — Indirect Command Buffers (encode the static 337-dispatch schedule ONCE, replay
per token; only pos/embedding change in-place) is the identified next swing (Fable's #4,
+6–9 tok/s est., medium confidence, ~a day of purego-objc binding). Stage B's repack +
lane-per-row kernels are the right structure for a future **prefill/batch** path (compute/
bandwidth-bound, where the isolated 118 µs *would* show up) and are preserved in git history
(reverted from the decode path only).

## Findings — the cheap encode-tax probe RULES OUT ICB; the lever is dispatch-count

Before investing ~a day in Indirect Command Buffers, measured how much of the ~2.4 ms/token
overhead is **Go-side msgSend encode** (which ICB removes) vs **GPU-side per-dispatch
latency** (which it doesn't). Two free trims: skip `setComputePipelineState` when unchanged,
and batch the per-dispatch buffer binds (8–11 `setBuffer` calls) into ONE
`setBuffers:offsets:withRange:` — cutting ~2600 → ~337 buffer-bind msgSends/token.

**Result: 14.83 → 14.30 ms/token (~68 → ~70 tok/s, 0.98× bar).** Only **~0.5 ms of the
~2.4 ms** was Go-side. **The rest is GPU-side per-dispatch launch latency across 337 serial
dispatches.** So **ICB would not pay off** — it eliminates the Go-side encode we just captured
for free, but the GPU still serially executes 337 commands. Fable's +6–9 tok/s ICB estimate
assumed the gap was encode-side; this falsifies that assumption cheaply (the user's
"measure Go-side vs GPU-side first" instinct was exactly right — it saved a day).

**The only lever left to clear the bar is reducing the dispatch COUNT** — megakernel-style
fusion (multiple ops per dispatch, fewer GPU command launches), i.e. `docs/cuda-megakernel-
spec.md` applied to Metal. That is a real architectural effort, not a tuning pass.

**Final spike verdict — GO, effectively at the bar.** cgo-free native-Metal decode:
**~70 tok/s (0.98× the ~71 GO bar, 2.2× WebGPU-on-Metal, 3.5× the untuned spike), correct
(21/24 argmax, cosine 0.989)**, with the fused-argmax production decode path (`ForwardArgmax`).
Given best-of-40 run-to-run noise is ±1.5 tok/s, decode is at the bar for practical purposes.
Beating it decisively is a megakernel effort; the scaffolding (bindings, bit-exact kernels,
fused decode, correctness gate, Stage A/B kernels, argmax fusion) is all committed.

## Findings — megakernel tested and CLOSED on Metal (no grid-sync; redundant-recompute is net-negative)

Followed the CUDA spec's §5 for Metal. Two sub-findings:

1. **The true 1-launch/layer megakernel is not possible on Metal.** It needs grid-wide sync
   *inside* the kernel (each GEMV stage spans many threadgroups; the next stage depends on all
   of them). CUDA has `grid.sync()` via `cuLaunchCooperativeKernel`; **Metal has no
   cooperative launch and no grid barrier.** The spec's option 1 is CUDA-only.

2. **The sync-free alternative (redundant-recompute, spec option 2) is net-NEGATIVE here.**
   Built fused `rmsnorm+quant+GEMV` kernels where each GEMV threadgroup recomputes the shared
   activation locally (no grid barrier needed), folding 3 dispatches/layer away (12→9). Result:
   **correct (21/24 argmax, cosine 0.989) but 70 → 52 tok/s.** Why: a gate/up threadgroup does
   only ~8 GEMV rows but now redundantly recomputes the *entire* 1536-element rmsnorm+quant
   (two full threadgroup reductions) first — and that recompute, done by all 2240 threadgroups
   and sitting on the critical path, dwarfs each threadgroup's tiny GEMV slice. The redundant
   work costs far more than the dispatch it removes (decode's per-threadgroup GEMV slice is too
   thin to amortize a full-vector reduction). Reverted.

The only sync-free fusions that add *no* redundant reduction are the element-wise ones
(rope-q + rope-k + kv-store → 1), which save just 2 dispatches/layer and are fiddly (three
different thread counts) — not worth it against the ~70/71 margin.

**Megakernel verdict: closed for Metal single-stream decode.** The dispatch-count lever the
earlier diagnosis pointed to cannot be pulled without grid-sync (absent) or redundant work
(net-negative at these dims). **~70 tok/s (0.98× bar) stands as the practical ceiling** for
this cgo-free native-Metal decode architecture. Beating it would need either a future Metal
cooperative-launch primitive, or a fundamentally different work decomposition (e.g. batching /
prefill, where threadgroup GEMV slices are fat enough to amortize fused reductions — Stage B's
lane-per-row + a fused norm would pay off there, not in batch-1 decode).

## Findings — Step 0 (GPU timestamps) + L1 (f16 scales): issue-bound CONFIRMED, at the bar

Executed the headroom sequence (Step 0 → L1) from `metal-decode-headroom-fable.md`:

- **Step 0 — cgo-free GPU-timestamp capture** (`MTLCommandBuffer.GPUStartTime/GPUEndTime` via
  `objc.Send[float64]`, arm64 fp-return): per-token **wall 14.0 ms = GPU-busy 13.0 ms + host
  bubble 0.9 ms (~7%)**; traffic **964.7 MB/token** (matches Fable's corrected budget — my
  review's 772 undercounted the f32 scales 25%); GPU-busy effective BW **73.6 GB/s** (in the
  incumbent Ollama's 75–83 band → near the GPU-bound ceiling).
- **L1 — f16 group scales** (all GEMV kernels + packers; new `f32ToF16`/`NewBufferU16s`):
  **parity-neutral** (cosine 0.9887, fused-argmax still 24/24 exact), **71.4 tok/s — crosses
  the ~71 bar (1.01×)**. **KEY: traffic −10% (965→868 MB) but GPU-busy stayed FLAT (13.0 ms).**
  Bandwidth-bound would have dropped GPU time with the bytes; it didn't → **issue-bound
  CONFIRMED** (Fable's diagnosis). So the traffic diet barely helps GPU-time (scale loads are
  ~1/4 of weight-load issue, dwarfed by the nibble-unpack ALU with no DP4A on Apple).
- **L2 cheap variant** (`commandBufferWithUnretainedReferences`): **no measurable change**
  (bubble is the raw encode `msgSend`s, not resource retain/release). Reverted.

**Where the remaining decode headroom is, now that issue-bound is confirmed:**
- The **0.9 ms host bubble** is CPU encode of 337 dispatches while the GPU idles. Recoverable
  only by **encode-ahead** (double-buffered cmd-buffers/uniforms, encode t+1 during GPU(t)) or
  **ICB** — both ~+5 tok/s (→~77) but ~a day of NSAutoreleasePool-lifetime-risky work.
- The **13.0 ms GPU-busy** is the real cost, and it's **issue-bound** → the only lever is
  **"Stage C"**: an issue-optimised GEMV (fold −8·Sg once/group, uniform L1-broadcast
  activations, 2-deep pipelined independent loads to hide latency, f16 scales) that cuts
  ops/weight from ~10–12 to ~3. Fable's estimate if issue-bound (now confirmed): GEMV
  9.3→7.4–8.2 ms → **~88–97 tok/s**. Days of kernel work + a repack.
- Bigger prize: **self-speculative / prompt-lookup decode** (goinfer ships `--spec ngram`) on
  a **Metal batch-k verify forward** — verifying k drafted tokens per weight pass amortises the
  issue-bound stream k×, the only way to convert unused bandwidth into single-stream tokens
  without grid-sync. Plausibly **1.5–2.5×** for a coder model, and it's the same batch-k kernel
  prefill needs.

**Verdict at this checkpoint: 71.4 tok/s, past the bar (1.01×), issue-bound confirmed.** Pure-
decode headroom is either small (encode-ahead, +5, risky) or large-but-days (Stage C, →~90);
the multiplier is speculation. Diagnosis is now measurement-grounded, not inferred.

## Findings — speculation pivot: make-or-break batch-k experiment came back NEGATIVE

Before building the full Metal batch-k verify forward, tested the load-bearing assumption:
does a single-weight-pass **batch-k W4A8 GEMM** amortize the issue-bound unpack, so a verify of
k drafted tokens costs ≪ k× a single forward? Built `gemv_w4a8_sa_bk` (one weight matrix ×
kk staged activations, unpack each weight group ONCE, MAC against all kk) + dynamic threadgroup
memory (`Run1DBatchTG` / `setThreadgroupMemoryLength`, so k activations don't blow occupancy).

**Result (gate/up 17920×1536, per-token µs, best-of-20 hot):**
```
single-token Stage A: 161 µs
k=1 batch: 226  k=2: 186  k=4: 166 (102.8% of single)  k=8: 194
```
**T_k ≈ k·T_1 — essentially NO amortization** (k=4 best case ties the single-token loop; k=8
regresses on occupancy). Only ~10% at k=2 with a hand-sized static array; the unpack is only
~20% of the per-weight cost, the int MACs ~80%, and batching amortizes only the unpack.

**Why this kills speculation (airtight economics):** speedup = a·T_1/T_verify (a = accepted
tokens/round). Win requires T_verify < a·T_1. Measured T_verify ≈ k·T_1 ⇒ win requires **a > k**,
impossible (a ≤ k+1, averages far less). Speculation gives ≤1× here.

**Root cause = the SAME issue-bound property confirmed by L1 (f16 scales flat GPU-time):** no
DP4A/int8-dot on Apple ⇒ the int4 GEMV is scalar-int-MAC-bound, so the MAC work scales with k
and a batched verify is NOT cheap. **Speculation only pays when the forward is memory-bandwidth-
bound (batched verify ≈ free); this kernel is compute/issue-bound.** The property that caps
single-token decode also kills speculation — they're the same wall (Fable's 1.5–2.5× spec
estimate assumed a memory-bound weight pass, the same assumption L1 already disproved).

**Corollary caution for the remaining levers:** since the bottleneck is int-MAC throughput (not
unpack or bandwidth), **Stage C** (which optimises the unpack via −8·Sg folding + uniform
activations) likely also underdelivers — f16 scales (unpack-side) already showed flat GPU-time.
The one lever with guaranteed, MAC-independent value is **encode-ahead** (kills the measured
0.9 ms host bubble, +~5 → ~77). Beating that decisively appears to need hardware goinfer can't
reach on M1 (DP4A), i.e. **~71–77 tok/s is near the practical ceiling for cgo-free W4A8 decode
on this GPU.** Experiment kept (`batchk_test.go`, `gemv_w4a8_sa_bk`) as the evidence.

## Findings — encode-ahead: 70.5 → 73.6 tok/s, parity-exact (last pure-decode lever)

The one MAC-independent lever. Step 0 measured a 0.9 ms host bubble (CPU encoding 337 dispatches
while the GPU idles). Added a persistent OS-thread-pinned executor goroutine (CUDA-style) that
pipelines: commit token t → pre-encode t+1 while the GPU runs t → wait. The encode is hidden
behind GPU execution. Pool safety (the SIGSEGV risk): one long-lived pool, drained every 64
tokens with a one-token non-overlapped hiccup so no un-committed command buffer is live across
the drain (LIFO-safe); `encodeTrunkInto` records dispatches only (value-independent), uPos/r.x
set at commit. Result: **73.6 tok/s (+3.1), parity EXACT (10/10 argmax vs sync), CLI coherent.**
Recovered 0.61 of 0.9 ms; residual is embedding-copy + uPos-set + logits-readback + channel hop.

**Pure-decode work is now DONE at ~73–74 tok/s (1.04× bar).** All remaining levers are either
ruled out (speculation, Stage C underdelivers — MAC-bound) or unavailable (DP4A). Next: the
"different work" that doesn't fight the int-MAC wall — prefill throughput, KV-f16 for long
context, or broader model coverage.

## Findings — KV-f16 (memory) + prefill make-or-break (MMA GO, 3.5x)

- **KV-f16** (`kv_store`/`attention` kernels → `half` K/V, dot/accum stay f32): **memory win,
  not speed.** Halves the KV cache (235→117MB across 28 layers → more context in 16GB); flat
  tok/s (attention is MAC-bound like the GEMVs, so halving the KV *load* width doesn't cut the
  dot MAC count). Parity 20/24 (cosine unchanged — one near-tie nudge, inside the gate).
- **Prefill make-or-break (`mma_test.go`) — GO.** simdgroup_matrix (8×8 f16 MMA) **compiles +
  runs cgo-free at MSL 3.1** (the key unknown), correctness maxErr 0.0. A naive MMA GEMM is ~2×
  the int4 GEMV but doesn't amortize (per-row rises with M — weights re-read per tile). A
  **blocked GEMM (RPS=4 row-tiles/simdgroup, weight tile reused across rows) amortizes**: per-row
  ~47µs FLAT with M vs the GEMV's 164µs = **3.5× over the per-token prefill loop**. The f16 MMA
  path amortizes across the prompt dim (unlike the scalar-int4 decode path). Prefill is viable:
  ~13.7s TTFT (1000-tok prompt) → ~4s. Full build (blocked MMA + in-kernel int4→f16 dequant +
  f16 activation path + causal prefill attention + PrefillN wiring + parity) is ~1–2 weeks.

## Findings — FULL PREFILL BUILD complete: 3.7× faster TTFT, correct, wired (P1–P5)

Built the whole f16-MMA prefill path end-to-end (`metal/prefill.go`, opt-in, separate library):
- **P1 — `gemm_w4f16`**: blocked int4→f16 MMA GEMM. Each simdgroup owns RPS=4 row-tiles × one
  8-col tile; dequants each 8×8 weight tile ONCE (transposed to Wᵀ, in-kernel, reusing the
  resident int4 weights — no extra RAM) and reuses it across the 4 activation tiles via
  simdgroup_matrix. Correct vs CPU (cos=1.0), 2.5× the int4 GEMV, flat with M.
- **P2 — f16 activation path**: `rmsnorm_f16`, `residual_f16`, `swiglu_f16`, `rope_f16` (per-row
  positions), `kv_store_f16` (scatter to the f16 KV cache). Plus `gemm_w4f16_store` with fused
  bias/residual epilogues.
- **P3 — `attention_prefill`**: per-(row,head) causal attention over the shared f16 KV, reusing
  the decode softmax math (O(M²), inherent; MMA/flash is a later throughput lever).
- **P4 — `PrefillLast`**: assembles all layers + lm-head-on-the-last-tile into ONE command
  buffer; returns the last token's logits, populates the KV. **Parity vs the sequential Forward
  loop: last-token argmax MATCHES, cosine 0.9987** (f16-prefill vs int8-decode = same model).
  **TTFT (140-tok prompt): 2034 ms → 543 ms = 3.7×** (beats the raw GEMM 2.5× because the
  sequential loop also pays the per-token bubble ×140, folded into one command buffer here).
- **P5 — wiring**: optional `decoder.Prefiller` interface; `generateInto` uses it for GPU prefill
  (prompt ≥ 8, ≤ cap), falls back to the sequential loop otherwise. Non-metal backends unchanged.
  End-to-end CLI verified (coherent output on a real prompt via the batched path).

Bring-up bug caught + fixed: `rmsnorm_f16` is threadgroup-per-row → needs M*256 threads, not M
(dispatch clamps M to one 14-thread group → only row 0 normed). Later prefill levers (not done):
dequant uses 8/32 lanes; flash-attention MMA; 2D register tiling. **Prefill: DONE and shipping.**
