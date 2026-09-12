# GPU vendor coverage: WebGPU covers everyone; native is per-vendor (and a treadmill)

> **Audience:** durable reference — the home for the recurring "what about AMD /
> Mac / Intel?" question. Bottom line up front: **goinfer already runs on every
> popular GPU** via WebGPU; native backends add *speed* on specific vendors, not
> *coverage*. Each native vendor is its own kernel language + driver binding +
> permanent maintenance commitment — the treadmill this doc weighs against demand.

## The two-tier model (the mental map)

**Tier 1 — cross-vendor "good," one codebase (shipped): WebGPU.** `goinfer/gpu`
(wgpu-native, `-tags gpu`) compiles the **same WGSL kernels** to the platform's
native API: Metal (Apple Silicon), Vulkan (AMD/Intel/NVIDIA on Linux), DX12 (all
vendors on Windows). AMD, Intel, and Mac GPUs run goinfer today on kernels
written once.

**Tier 2 — per-vendor "native/fast," separate codebase each: CUDA and Metal
(both shipped), then a treadmill.** Both are cgo-free (CUDA via `gocudrv`'s
`dlopen libcuda`; Metal via `purego`'s `objc` package, no `import "C"`
anywhere in either backend) and both sit on the same feature matrix,
[docs/hardware-matrix.md](hardware-matrix.md), same as WebGPU. Same-box, CUDA is
measurably faster than WebGPU on this box — WebGPU runs at **38 / 41 / 63%** of
goinfer's own CUDA at 0.5B / 1.5B / 7B (`docs/benchmarks.md` §B8, 2026-08-26; no
equivalent WebGPU-vs-Metal row exists yet). There is no shared native kernel
language across vendors, so each one is its own shader language *and* its own
cgo-free driver binding.

## Priority gradient (if a third vendor's demand ever clears the bar)

1. **AMD / HIP — best effort-to-reward.** HIP is a CUDA clone, so the CUDA
   kernels port semi-mechanically (`hipify`); the real new work is the cgo-free
   HIP binding (`libamdhip64` dlopen). Cheapest native vendor to add.
2. **Intel — lowest priority** (smallest GPU-LLM install base; oneAPI/SYCL or
   Level Zero, `libze` dlopen — full rewrite either way).
3. **Vulkan (cross-vendor NVIDIA+AMD+Intel, one SPIR-V codebase) — skip.** It
   buys ≈nothing over WebGPU: wgpu already compiles WGSL → SPIR-V/Vulkan at
   roughly the same ceiling. The native-speed leaps are the *vendor-specific*
   APIs, not the cross-vendor ones. Vulkan-native is WebGPU with extra steps.

## Verdict

- **Coverage is done.** WebGPU handles every popular card; nobody is unsupported.
- **Native speed is a per-vendor treadmill.** CUDA and Metal both cleared the
  bar (biggest single targets, both landed cgo-free); a third vendor needs a
  concrete adopter's demand to clear the same bar, not a speed argument alone —
  WebGPU is already the answer for whichever vendor hasn't.

**Correction (2026-09-12), retracting the 2026-07-15 verdict.** This doc
predicted Metal's Obj-C API would resist the cgo-free trick and "likely
reintroduce cgo," making it "the one to watch" rather than the one to ship. It
shipped cgo-free instead (`purego`'s `objc` package), and the treadmill cost
that verdict treated as hypothetical is now measurable, not predicted: **two**
native backends to keep on the same feature matrix as every family lands,
parity-gated, forever. That maintenance load — not whether a third vendor
*could* be done — is the real argument against adding one without demand to
justify it.
