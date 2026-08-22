# goinfer Metal spike — Phase 1: set the GO bar, then Layer A (with the MSL-version landmine pre-armed)

> **STATUS: DELIVERED — Metal shipped.** GPU residency on Metal was admitted end-to-end (G9/G10 in
> the v0.14.0 CHANGELOG) and the backend has been through several optimization campaigns since.
> The warning below still holds and matters more with age: **every peer number here is stale.**
> Read it as the spike's method record.


> **⚠ Peer numbers below predate the Ollama v0.32.5 re-anchor (2026-08-04).** Competitive figures
> in this doc (e.g. Ollama-CUDA ~149, Ollama-Metal 83.3, llama.cpp-CUDA 72.8, and any "×Ollama"
> multiple) were measured against **Ollama 0.5.7 (2025-01) / Ollama-Metal 0.32.0 / llama.cpp as of
> v0.5.0** — historical working records, not current claims. Current same-box numbers vs Ollama
> **v0.32.5** are in `docs/benchmarks.md` §B2 (CUDA) / §B3 (Metal).


> Continues `task-metal-cgofree-spike.md`. **Step 0 (phase 0) is done** — commit
> `4461ca5`, baselines on the M1 Pro rig. Read the spike doc's **Findings** and
> **Go/no-go** sections before starting; this prompt sequences the next moves and
> hands you one critical landmine that is **not in the doc yet**.

## Where Step 0 left us — the bar is not yet set

On the M1 Pro (macOS 26.5.2, commit `c51219c`, qwen2.5-coder-1.5b q4_k_m, int8):

- WebGPU-on-Metal resident = **31.9 tok/s**, CPU = **32.66** → the incumbent Mac GPU
  path is a **wash with CPU** (0.98×). The native lane's headroom is the
  **31.9 → ~129 tok/s** (bandwidth-ceiling) gap — large. Step 0 *strengthened* the case.
- **Ollama-Metal was NOT measured** (ollama not installed on the rig). Per the
  Go/no-go bar: because WebGPU-on-Metal is degenerate, **the GO bar is ≥85% of
  same-box Ollama-Metal** — so *you cannot even state the bar until Ollama-Metal is
  measured on this rig.*

Phase 1, in order:

### 0. Set the bar first (cheap, unblocks the whole verdict)
Install Ollama, **pin the version**, run the same GGUF greedy/warm, record decode
tok/s into the spike doc's baseline table. Hard prerequisite — the CUDA arc's
"stale-147" lesson is exactly this: anchor the native peer *on-box* before claiming
anything. Until this number exists, a "GO" is unprovable.

### 1. Layer A — the purego-objc compute binding (the open risk, the 80% this time)
Layer B (kernels) is de-risked by the CUDA arc; the binding is the risk. Two rules:

**(a) Recon before you hand-roll a single selector.** Read, for proven cgo-free
msgSend/encoder patterns: **ebiten's Metal driver post-#3411** (production
purego-Metal — render-side, but the msgSend/autorelease patterns transfer) and
**`github.com/gogpu/wgpu/hal/metal`** (claims a pure-Go Metal *compute* HAL — treat
as reference, verify maturity, don't build on it). Crib the command-buffer /
compute-encoder / dispatch selector sequences; don't reinvent them.

**(b) Arm the MSL-version landmine from line one** — next section. It bites exactly
at your `newLibraryWithSource:` call, and it fails **silently**.

Proceed to Layer B (the six proven kernel ports) **only once Layer A proves the
binding stays cgo-free in practice** (ARC/blocks/dispatch don't force cgo). If it
can't stay cgo-free, that is the NO-GO — stop there.

## ⚠️ The landmine: `CGO_ENABLED=0` silently downgrades your MSL compiler

New intel, **not yet in the spike doc — add it as risk #7 when you start:**

Go binaries built with `CGO_ENABLED=0` on macOS omit the **`LC_BUILD_VERSION`** load
command from the Mach-O header (filed upstream: **golang/go#77917**). Without that
deployment-target metadata, Metal's runtime compiler
(`newLibraryWithSource:options:error:`) **defaults `MTLCompileOptions.languageVersion`
to MSL 2.4**, which **silently strips modern types (e.g. `bfloat16`)** — kernels
"compile" fine and produce wrong/degenerate output. This is the *exact* config this
spike targets (`CGO_ENABLED=0` + runtime-compiled MSL), so you **will** hit it.

**Fix — bake in before the first real kernel:** explicitly allocate an
`MTLCompileOptions`, set `.languageVersion` to **MSL 3.1 or higher** (rig is macOS 26,
so anything ≥3.1 is safe — match it to the features you use), pass it to
`newLibraryWithSource:options:error:`. **Never rely on the default** — inheriting 2.4
from a build flag is the hazard, independent of whether you use bf16 today. Add a
startup assertion that the compiled library reports the version you set, so a
regression is **loud, not silent**. (Root cause is the missing load command; a
linker-side mitigation via a deployment-target `-ldflags` may also exist and could
matter for other macOS runtime checks — worth a look, but the in-code
`languageVersion` set is the reliable primary fix.)

**Why this is good news, not bad:** the trap is documented because **others walk this
exact path**. `hybridgroup/yzma` (pure-Go, purego, `CGO_ENABLED=0`, llama.cpp Metal on
macOS) hit the same LC_BUILD_VERSION issue — corroboration that the
cgo-free-purego-macOS lane is real and maintained by more than us. (yzma binds
llama.cpp's *C API*, so it's trap-confirmation, **not** a source for compute-encoder
selectors — get those from ebiten/gogpu.) Combined with the existence proofs already
in the doc (purego/objc, ebiten #3411), the binding path is de-risked as *technique*;
the remaining risk is purely our compute-encoder execution quality.

## Carry the discipline over (from the doc — don't re-derive)

- **No perf claim without a device-timed run** (`GPUStartTime`/`GPUEndTime`), warm,
  best-of-N, versioned (macOS + chip + commit + purego + pinned Ollama).
- **Prove cgo-free, don't assume it:** `go version -m` + `otool -L` show no
  Metal/libobjc linkage (dlopen'd at runtime; state the libSystem caveat honestly —
  "cgo-free + OS-only," not fully static).
- **Obj-C memory by hand (no ARC):** autoreleasepool + explicit release on every
  per-token commandBuffer/encoder or the decode loop leaks (doc risk #2). Isolate the
  per-token binding tax (msgSend trampoline + commit/wait) the way the CUDA spike
  isolated the ~5 µs channel-hop (risk #3).
- **Correctness gate:** argmax-parity vs CPU decode on the **real checkpoint**,
  `t.Fatalf` on breach, same as `cuda/realforward_test.go`.
- **Scope guard:** one chip, dense residency-eligible Qwen2 decode only, the six
  proven kernels, same GGUF. No MoE/MLA/Mamba/prefill/batching/iOS/MPS/ANE. Resist all
  sprawl.
- **Attribution as you go:** NOTICE + THIRD_PARTY_LICENSES entries for purego and any
  prior art you consult (ebiten Apache-2.0, gogpu BSD) — the gocudrv precedent.

## Deliverable
Append to `task-metal-cgofree-spike.md`: the Ollama-Metal number (→ the now-stated GO
bar), Layer A's cgo-free verification transcript, the per-token tax, and the risk-#7
landmine note. Timebox stands — a long weekend, most of it Layer A. Not clearly GO by
then = NO-GO, and the Ollama-Metal baseline is banked either way.
