# GPU binding migration: cogentcore/webgpu → go-webgpu — effort assessment

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


> **Scope:** estimation/analysis only. Assesses migrating goinfer's GPU backend
> from `github.com/cogentcore/webgpu` v0.23.0 (CGO, statically linked wgpu-native)
> to `github.com/go-webgpu/webgpu` **v0.5.2** (zero-CGO, runtime-loaded wgpu-native
> v29). No code was changed. Gaps verified against the fetched go-webgpu v0.5.2
> source at `$GOPATH/pkg/mod/github.com/go-webgpu/webgpu@v0.5.2/`.

## ⛔ Runtime feasibility BLOCKER (found 2026-06-17 — overrides the verdict below)

**Do not start this migration yet. go-webgpu v0.5.2 does not run on the project's
toolchain.** Before writing any code, the runtime was validated empirically on the
Linux GPU dev box (RTX 2070 SUPER, Vulkan, Go 1.26.3):

1. `cmd/setup` fetched `wgpu-native v29.0.0.0` (`libwgpu_native.so`) fine.
2. Running **go-webgpu's own `examples/compute`** (unmodified, with `WGPU_NATIVE_PATH`
   set and `wgpu.Init()` called) **aborts with SIGABRT** inside the native library:

   ```
   thread '<unnamed>' panicked at src/lib.rs:2801:43: invalid callback
   panic in a function that cannot unwind ... SIGABRT
     #3 wgpuInstanceRequestAdapter
   ```

   It dies at the **first async call (RequestAdapter)** — goffi's pure-Go FFI callback
   is rejected by wgpu-native as invalid. The GPU/Vulkan stack itself is healthy: the
   current cogentcore binding runs the full `-tags gpu` suite green on this same box.

**Not a goinfer bug, and not fixable from goinfer.** Reproduced with the upstream
example (zero goinfer code). Ruled out: goffi v0.5.2 → v0.5.5 (same crash),
`GODEBUG=asyncpreemptoff=1`, missing `wgpu.Init()`, nil-vs-options. Most likely a **Go
1.26 vs goffi callback-ABI incompatibility** (go-webgpu/goffi declare/test against Go
1.25; this repo is pinned to Go 1.26.3) or a callback-struct ABI mismatch with this
wgpu-native build. Either way it is an upstream defect in the go-webgpu/goffi stack on
this exact toolchain.

**Consequence:** the *code* migration below is still accurate and tractable, but it
**cannot be validated** here (the bit-exact parity gates — the whole point — can't run),
and worse, a migrated GPU binary would **crash at adapter acquisition** on the project's
own Linux box. The decision flips from "conditional GO on distribution" to:

> **NO-GO right now.** Revisit only after go-webgpu's own `examples/compute` runs green
> on Go 1.26 + wgpu-native v29 on Linux (verify upstream first — it is the gate; the
> goinfer migration is downstream of it). Track the upstream fix; do not invest the
> ~1 week until that smoke test passes. (Alternative paths if it's urgent: pin the GPU
> build to Go 1.25, or wait for a goffi release that re-establishes Go-1.26 callback
> support — neither confirmed to work.)

## ✅ Update (2026-06-19): the blocker is goffi, NOT wgpu-native v29 — and v29 has no decode penalty

Two things got conflated and are now separated by measurement:

1. **The runtime crash is the goffi/zero-CGO FFI path, specific to it.** Confirmed by
   running `github.com/oliverbestmann/webgpu` v1.33.4 — a **CGO** fork of cogentcore that
   statically links **wgpu-native v29** — on this exact box (Go 1.26.3, Vulkan, RTX 2070
   SUPER). It builds, links, acquires the adapter, and runs compute **fine**. So
   wgpu-native v29 itself runs here; only the **goffi pure-Go callback ABI** is broken on
   Go 1.26.

2. **There is no "v29 decode penalty."** Matched gemv-shaped micro-benches (cogentcore/v22
   vs oliverbestmann/v29, RTX 2070 SUPER, best-of-N throttle-free):
   - gemv [4096×4096] f32: v22 0.62 ms vs v29 0.64 ms/dispatch (~4%, within throttling noise)
   - per-dispatch cgo **record** cost: **identical**, 1.1 µs/dispatch on both
   - 256-dispatch decode-shaped (record→submit→poll): v22 147 ms vs v29 152 ms (**+3.5%**,
     attributable to the oliverbestmann fork's GC-finalizer layer, not v29)

   The earlier "−23% v29 decode loss" does **not** reproduce. If a zero-CGO distribution
   binding ever clears the Go-1.26 goffi gate, v29 perf is not a reason to avoid it.

**Not worth switching to oliverbestmann, though:** it's not a drop-in (its GC redesign
changed the API — single-return `CreateX`, `WGSLSource` vs `WGSLDescriptor` — so it's a
real port across the ~35 binding files), it's still CGO (no distribution win over
cogentcore), and it's marginally *slower* (the GC layer). Stay on cogentcore for zero
churn; revisit only if cogentcore goes unmaintained.

---

## Verdict (one paragraph)

**Tractable, ~1 engineer-week, mostly mechanical — but the go/no-go hinges on
distribution, not code.** All binding usage is isolated to the `./gpu` module (27
files; `gpu/doc.go:1` records it as the one place the binding is imported), and the
~600 symbol sites are dominated by three *mechanical, shim-able* gaps —
`CreateBufferInit` (74 sites), `ToBytes`/`FromBytes` (105 sites), and the
`ShaderModuleWGSLDescriptor` reshape (~16 sites). A thin `gpu/internal/wgpucompat`
shim (~200 LOC) keeps the large majority of call sites a near-textual swap. The only
genuinely *semantic* rewrites are concentrated in two functions: the buffer-readback
mapping in `gpu/device.go:245` (`MapAsync`+`Poll`+status → the blocking
`Buffer.Map(ctx)`) and adapter/device acquisition in `gpu/gpu.go:210` (`DefaultLimits`
is absent; `GetInfo` → `Info()` with renamed fields). The real decision is the
**distribution trade-off**: go-webgpu drops the CGO toolchain (enabling
`CGO_ENABLED=0` cross-compilation — strongly on-strategy for a pure-Go runtime) but
adds a *runtime* dependency on the wgpu-native shared library (`.so`/`.dylib`/`.dll`,
installed via `go run github.com/go-webgpu/webgpu/cmd/setup@latest`), replacing the
current self-contained static link. Recommendation: **conditional GO** — worth doing
for the cgo-free build upside, *if* the runtime-library packaging is owned (bundle in
release artifacts) and the bit-exact parity gates are re-run green.

## API gap verification (corrections to the brief)

Verified against go-webgpu v0.5.2 source. The brief was accurate except where noted:

| Gap (brief) | Verified in v0.5.2 | Remediation |
|---|---|---|
| `BufferInitDescriptor`/`CreateBufferInit` ABSENT | ✅ absent | shim `BufferInit()` over `CreateBuffer(MappedAtCreation:true)`+write+`Unmap`, or `CreateBuffer(COPY_DST)`+`Queue.WriteBuffer` |
| `ToBytes`/`FromBytes` ABSENT | ✅ absent | shim, trivial `unsafe.Slice` generics (~10 LOC) |
| `ShaderModuleWGSLDescriptor{Code}` shape differs | ✅ `Device.CreateShaderModuleWGSL(code string)` exists [shader.go:21]; or `ShaderSourceWGSL{Code: StringView}` | shim `CompileWGSL(dev,label,code)` |
| Buffer mapping differs | ✅ `Buffer.Map(ctx, mode, off, size) error` (blocking) [map_pending.go:151] replaces MapAsync+Poll+status; `GetMappedRange→unsafe.Pointer` [buffer.go:156]; `Unmap() error` [buffer.go:175] | rewrite `Readback`; shim `MapReadBlocking()` |
| `BufferMapAsyncStatus*` enum ABSENT | ✅ absent (go-webgpu has `MapAsyncStatus`, no `Unknown`) | falls out with the blocking-Map rewrite |
| `DefaultLimits` ABSENT | ✅ absent | shim `DefaultLimits()` returning an explicit `Limits` |
| `RequiredLimits`, `ProgrammableStageDescriptor`, `PowerPreferenceHighPerformance`, `FeatureNameTimestampQuery` PRESENT | ✅ present (note: `DeviceDescriptor.RequiredLimits` is `*Limits`, not `*RequiredLimits{Limits}`) | mechanical struct reshape |
| `NativeFeature` packed-int-dot-product (DP4A) constant | ⚠️ **CORRECTION: NOT present** in v0.5.2 `enums.go` (a `NativeFeature` type exists, but no `PACKED_INTEGER_DOT_PRODUCT`/DP4A member; `ShaderInt64/F64/I16`, `Subgroup` are present) | none needed — `dot4I8Packed` is a naga/WGSL builtin (no feature flag for correctness); the **HW fast-path is not unlockable by name here either**, so migrating does *not* advance the DP4A lever |
| `RequestAdapter`/`RequestDevice` sync, `(T,error)` | ✅ sync, return `(T,error)` [adapter.go, device.go:80] | mechanical (matches current style) |
| `Queue.Submit` returns `(uint64, error)` | ✅ [command.go:460] (25 call sites ignore the return) | mechanical `_, _ =` |
| `CreateInstance` | ⚠️ now returns `(*Instance, error)` [instance.go:62] — current code ignores error (`gpu.go:211`) | mechanical (handle/ignore error) |
| `Adapter.GetInfo()` / `AdapterInfo` | ⚠️ `Adapter.Info() (*AdapterInfoGo, error)` [adapter.go:489]; fields **renamed** (`Name→Device`, `VendorName→Vendor`, `DriverDescription→Description`) | semantic: rename at `gpu.go:284`, `spike_test.go:24` |
| `Adapter.GetLimits().Limits` | ⚠️ `Adapter.Limits()` returns `Limits` directly (no `.Limits` field) [adapter.go] | mechanical field-access change |
| Pass ops (`SetPipeline`/`SetBindGroup`/`Dispatch`) | ✅ void (errors via device callback) — matches current ignore-style | none |
| `wgpu.Init()` required | ✅ auto-called on first use, idempotent [wgpu.go:203] | none (or call once in `New`) |

## File-by-file

Sizing: **S** ≤ ~10 mechanical edits, **M** ~10–40, **L** > 40 or contains a semantic
rewrite. Change class is the *dominant* work; nearly every file also takes the import
swap + type-qualifier renames (mechanical).

### Production files (`./gpu`)

| File | Key symbols / sites | Class | Size | Notes |
|---|---|---|---|---|
| `gpu.go` | 133 wgpu refs; `New` acquisition `gpu.go:210-240`; `DefaultLimits()` `:225`; `RequiredLimits{Limits}` `:240`; `CreateInstance` `:211`; `GetInfo` `:284`; 5× CreateBufferInit | **SEMANTIC** + mechanical | **L** | Context struct = many `*wgpu.ShaderModule/ComputePipeline/BindGroupLayout` fields (pure type renames). Acquisition is the second semantic spot: build `Limits` explicitly (no `DefaultLimits`), `Info()` rename+error, `Adapter.Limits()` field change |
| `device.go` | 27 refs; `Readback` `:245-276` (MapAsync `:263`+Poll `:266`+status+GetMappedRange `:271`+Unmap `:272`); `UploadF32` `:112`; 3× CreateBufferInit | **SEMANTIC** (mapping) | **M** | The one true mapping rewrite — collapses to `Buffer.Map(ctx,...)`; `FromBytes` at `:271`. Concentrated, ~15 lines |
| `decoderunner.go` | 71 refs; mostly `CreateBuffer`/`CreateBindGroup`/`uni()`+`ToBytes`; 1× CreateBufferInit | MECHANICAL | **L** | High volume, low difficulty; shim collapses `uni`/`ToBytes`. The hot resident decode path — re-bench after |
| `moe.go` | 41 refs; 10× CreateBufferInit; 3× CreateShaderModule | MECHANICAL | **M** | Highest CreateBufferInit count |
| `mla.go` | 32 refs; 10× CreateBufferInit; 4× CreateShaderModule | MECHANICAL | **M** | (C4 levers from this session) |
| `gemv.go` | 53 refs; 7× CreateBufferInit | MECHANICAL | **M** | |
| `gemm.go` | 33 refs; 6× CreateBufferInit | MECHANICAL | **M** | |
| `quant.go` | 27 refs; 5× CreateBufferInit; 1× CreateShaderModule | MECHANICAL | **M** | |
| `decodetoken_fused.go` | 23 refs; 2× CreateBufferInit | MECHANICAL | **M** | |
| `gemv_w4a8.go` | 22 refs; 2× CreateBufferInit; 1× CreateShaderModule | MECHANICAL | **M** | |
| `vision.go` | 18 refs; 4× CreateBufferInit; 1× CreateShaderModule | MECHANICAL | **M** | |
| `decodelayer.go` | 16 refs; 3× CreateBufferInit | MECHANICAL | **S/M** | |
| `attention.go` | 13 refs; 7× CreateBufferInit | MECHANICAL | **S/M** | |
| `layer.go` | 12 refs; 2× CreateBufferInit; 1× CreateShaderModule | MECHANICAL | **S** | |
| `residency.go` | 11 refs | MECHANICAL | **S** | mostly `*wgpu.Buffer` type renames |
| `qknorm.go` | 4 refs; 1× CreateShaderModule | MECHANICAL | **S** | |
| `decodefuse.go` | 3 refs | MECHANICAL | **S** | |
| `vision_encoder.go` | 1 ref | MECHANICAL | **S** | |
| `backend.go` | import only (0 `wgpu.` refs) | MECHANICAL | **S** | import swap |
| `doc.go` | 0 refs (the isolation note) | DOC | **S** | update the binding name |

### Test/bench files

| File | Sites | Class | Size | Notes |
|---|---|---|---|---|
| `spike_test.go` | 5; `GetInfo` `:24` (field renames), `EnumerateFeatures`→`Features`, dot4I8Packed probe `:42-55` | SEMANTIC (renames) | **S** | the DP4A probe stays valid (naga builtin); only `Info()`/`Features()` renames |
| `graph_test.go` | 20; 1× CreateBufferInit | MECHANICAL | **M** | |
| `kv_i8_test.go` | 18; 3× CreateBufferInit | MECHANICAL | **M** | |
| `decode_instrument_test.go` | 12; 2× CreateShaderModule | MECHANICAL | **S/M** | |
| `gemv_w4a8_test.go` | 10; 3× CreateBufferInit | MECHANICAL | **S** | |
| `gpu_test.go` | 9 | MECHANICAL | **S** | |
| `decodetoken_bench_test.go` | 1 | MECHANICAL | **S** | |

**Touched files: 27** (20 production incl. doc/backend, 7 test/bench). No file outside
`./gpu` changes (the `decoder`↔`gpu` seam is the pure-Go `ResidencyBackend` interface;
`decoder` never imports the binding).

## The shim (recommended) and its size

A `gpu/internal/wgpucompat` package lets ~80% of sites change by import qualifier
only. Contents and rough size:

| Shim symbol | Replaces | LOC |
|---|---|---|
| `ToBytes[T](s []T) []byte` / `FromBytes[T]([]byte) []T` | the absent generics | ~12 |
| `BufferInit(dev, BufferInitDescriptor) (*Buffer, error)` | `CreateBufferInit` (74 sites) — `CreateBuffer(MappedAtCreation)`+copy+`Unmap` | ~30 |
| `MapReadBlocking(dev, buf, size) ([]byte, error)` | the `MapAsync`+`Poll`+status dance in `Readback` | ~25 |
| `CompileWGSL(dev, label, code) (*ShaderModule, error)` | the `ShaderModuleDescriptor{WGSLDescriptor:{Code}}` shape (~16 sites) | ~15 |
| `DefaultLimits() Limits` | the absent `DefaultLimits()` | ~25 |
| `type BufferInitDescriptor struct{Label string; Contents []byte; Usage}` | the absent descriptor | ~6 |
| thin type aliases (`Buffer`, `ShaderModule`, `BufferUsage*`, …) | re-export to minimize qualifier churn | ~40 |

**Shim ≈ 150–200 LOC.** It deliberately does *not* hide the two real semantic
changes (readback mapping model, acquisition/limits) — those are rewritten in place so
the control-flow change is explicit, not buried.

## Distribution: who actually pays the cost

The cost lands almost entirely on the *project's release pipeline*, not on users —
and not at all on the flagship "one file, download and run" lane:

- **CPU users (the download-and-run product) — unaffected.** The binding is
  `//go:build gpu` only, so the pure-Go static binary that the README promises has no
  webgpu dependency before or after. Download-and-run stays download-and-run.
- **GPU users (opt-in `-tags gpu`) — neutral to *easier*.** There are no prebuilt GPU
  binaries today; a GPU user already builds from source, and today that needs a **C
  toolchain** (cogentcore is cgo). After the switch the build needs **no C compiler**
  (`CGO_ENABLED=0`); the only new requirement is the wgpu-native shared library present
  at runtime. The build barrier drops; a runtime-file requirement appears.
- **The one place a user could see new friction** is running a GPU binary that can't
  find the lib. It is fully avoidable: go-webgpu's loader searches `./lib/<lib>`, so a
  release that **bundles the `.so`/`.dylib`/`.dll` beside the binary stays
  download-and-run** (unzip, run). Un-bundled, the user runs one `cmd/setup` step once.
  Either way the cost is the project's CI bundling the lib per platform — a one-time
  packaging task — not recurring user work.

Net: the migration trades a **build-time** C-toolchain requirement (today, GPU only)
for a **runtime** shared-library that the project can bundle away. It does not weaken
the CPU "single static binary" promise; it was never literally true for the cgo GPU
build (which links a native lib) and after the switch that native lib becomes a sibling
file the release archive carries.

## Distribution / CI / Go-version / test impact

- **Distribution (the crux).** cogentcore links wgpu-native *statically* via CGO → one
  self-contained binary. go-webgpu is CGO-free but `dlopen`s wgpu-native v29 at
  **runtime** (search: `WGPU_NATIVE_PATH` → `./lib/<lib>` → PATH), installed by
  `go run github.com/go-webgpu/webgpu/cmd/setup@latest`. Net: **lose** "single static
  binary"; **gain** `CGO_ENABLED=0` builds + trivial cross-compilation (no per-target C
  toolchain). For a `-tags gpu` release, ship the platform `.so`/`.dylib`/`.dll`
  alongside the binary (or document `setup`). The *default* (CPU) build is unaffected —
  the binding is `//go:build gpu` only, so non-GPU users see zero change.
- **CI.** GPU CI must run `cmd/setup` (or cache the lib) before `go test -tags gpu`.
  Upside: GPU cross-compile *builds* (not runs) need no C toolchain. The CPU CI matrix
  is untouched.
- **Go version.** go-webgpu needs go ≥ 1.25; goinfer is 1.26.3 — **OK**.
- **Tests.** The full `gpu/*_test.go` suite (~70 s this box) is the acceptance gate —
  every resident/parity test exercises the readback + buffer path that changes. The
  `spike_test.go` dot4I8Packed probe still reports correctly (the WGSL builtin compiles
  under naga regardless of binding); record whether wgpu-native v29's naga still
  accepts it. Bit-exact gates (W8A8 parity, the resident decode parity tests added this
  session) are the parity safety net — they must stay green unchanged.

## Recommended migration order

1. **Shim first** (`gpu/internal/wgpucompat`) + the runtime-lib setup in CI — land the
   substrate before touching call sites. ~0.5 day.
2. **Acquisition + readback** (`gpu.go` `New`, `device.go` `Readback`) — the two
   semantic rewrites; get one end-to-end op (e.g. `Attention`) green first as the
   vertical slice that proves the mapping model. ~0.5–1 day.
3. **Mechanical sweep** of the op files (gemv/gemm/moe/mla/quant/…/decoderunner) —
   import swap + `BufferInit`/`ToBytes`/`CompileWGSL`/`Submit`-return. sed-assisted,
   compile-driven. ~1.5–2 days.
4. **Tests/benches** — same mechanical sweep; then run the full `-tags gpu` suite +
   bit-exact parity gates; re-bench the resident decode hot path. ~1 day.
5. **Distribution** — release packaging for the runtime lib, README/CI docs. ~0.5–1 day.

## Effort estimate + go/no-go

- **Per-file:** ~2 L, ~10 M, ~15 S. The semantic work is **two functions** (`New`,
  `Readback`) + a handful of renames; everything else is volume.
- **Total:** **~4–7 engineer-days (~1 week)**, most of it the repetitive-but-low-risk
  mechanical sweep + the parity re-validation, not novel design.
- **Top 3 risks:**
  1. **Runtime shared-library distribution** — the static-link → runtime-`.so` change is
     the real cost and contradicts the "single static binary" line (though it *advances*
     the "cgo-free" one). Own the packaging or it bites users at deploy, not build.
  2. **Buffer-mapping / sync parity** — the `MapAsync`+`Poll` → blocking-`Map(ctx)`
     rewrite (`device.go:245`) is on the bit-exact decode path; a subtle ordering/sync
     regression would surface only in the parity gates. Mitigation: the existing gates
     are exactly the right net — run all of them.
  3. **wgpu-native v29 deltas** — different default limits, validation strictness, or
     naga shader-compilation behavior vs cogentcore's pinned wgpu-native could reject a
     currently-accepted shader or shift a limit (`maxBufferSize`, storage-binding cap).
     Plus: **no named DP4A feature** — migrating does *not* unlock the packed-int8
     hardware path (correct the optimistic read; it stays a separate, still-blocked lever).
- **Recommendation: conditional GO.** The code migration is well-isolated, low-novelty,
  and ~1 week. Do it **if** the cgo-free / `CGO_ENABLED=0` cross-compile is strategically
  valued (it aligns with goinfer's pure-Go positioning) **and** the runtime-library
  packaging is committed to. If "one self-contained static binary" is a hard product
  promise, **NO-GO** until go-webgpu (or a vendored wgpu-native) offers a static option —
  the binding swap itself is the easy part; the distribution model is the decision.
