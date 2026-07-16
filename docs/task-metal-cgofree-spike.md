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

## Findings — Layer B, step 1: the hot GEMV ported to MSL, bit-exact

`TestLayerB_gemvW8A8Parity`. The bandwidth-bound bottleneck — goinfer's **W8A8 GEMV**
(int8×int8 → i32 accumulate → × activation-scale × per-row weight-scale) — ported to
MSL and validated against a CPU reference on real per-row-int8-quantized data (256×1536,
qwen-ish proj shape): **cosine = 1.0000000, max-rel = 0.00e+00 — bit-exact**, cgo-free.
Mirrors the CUDA arc's "B step 1 (real-weight GEMV parity)": the packing + integer-GEMV
math carry over exactly; MSL is the C++ dialect the doc promised. (This is the naive
one-row-per-thread kernel — *correct first*; the CUDA lesson is that tuning, coalescing/
ILP to ~80% peak, comes after parity, and was all memory-access, not ALU.)

**Remaining Layer B (the bulk, per the doc's scope):** port the rest of the six proven
kernels (rmsnorm+quant, rope, GQA online-softmax attention, swiglu+quant, residual, +
the W4A8 GEMV and kv_store), assemble a full dense decode layer into **one command
buffer per token** (the tax requirement above), stand up the **argmax-parity gate vs CPU
decode on the real qwen2.5-coder-1.5b checkpoint**, and measure device-timed
(`GPUStartTime`/`GPUEndTime`) decode **tok/s vs the ~71 GO bar** (aspirational ~83 Ollama
parity). Layer A being retired means this is now the *mechanical* half — real work, but
no open viability risk.
