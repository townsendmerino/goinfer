# GPU vendor coverage: WebGPU covers everyone; native is per-vendor (and a treadmill)

> **Audience:** durable reference — the home for the recurring "what about AMD /
> Mac / Intel?" question. Companion to `positioning-cuda-and-promotion.md` and the
> CUDA spike (`task-cuda-cgofree-spike.md`). Bottom line up front: **goinfer already
> runs on every popular GPU** via WebGPU; the CUDA backend added NVIDIA *speed*, not
> *coverage*. Native paths for other vendors are each a separate, multi-language
> kernel + driver-binding + maintenance commitment — the "treadmill" the positioning
> doc warns about, multiplied by vendors.

## The two-tier model (the mental map)

**Tier 1 — cross-vendor "good," one codebase (already shipped): WebGPU.**
`goinfer/gpu` (wgpu-native, `-tags gpu`) compiles the **same WGSL kernels** to the
platform's native API: **Metal** (Apple Silicon), **Vulkan** (AMD, Intel, NVIDIA on
Linux), **DX12** (all vendors on Windows). So **AMD, Intel, and Mac GPUs run goinfer
today** on kernels you already wrote — one codebase, all vendors, ~60–70% of native
speed (the WGSL "no single-dispatch megakernel" ceiling, `gpu-assessment.md`).

**Tier 2 — per-vendor "native/fast," separate codebase each: CUDA (done), then a
treadmill.** Native beats the WGSL ceiling (CUDA measured **1.96× WebGPU** on the
2070 SUPER) — but there is **no shared native kernel language**, so each vendor is
its own shader language *and* its own cgo-free driver binding.

## What each vendor would cost (Tier 2)

The dense residency path is ~**6 kernels** (W4A8/W8A8 GEMV, RMSNorm, RoPE, residual,
SwiGLU, attention). "Native support for vendor X" = those 6 rewritten in X's language
+ a cgo-free binding to X's driver + a parity gate + permanent maintenance.

| Vendor | Native API / kernel language | cgo-free binding | Port difficulty |
|---|---|---|---|
| **NVIDIA** | CUDA C → PTX | gocudrv (**done**) | — |
| **AMD** | **HIP** (ROCm) — a CUDA *clone* | `libamdhip64` dlopen (new) | **Easiest** — `hipify` translates the CUDA kernels semi-mechanically; same programming model |
| **Apple** | **Metal Shading Language** — unrelated to CUDA | purego-objc dance (**existence-proven** — see correction below) | **Full rewrite** of every kernel; binding via `objc_msgSend`, gated spike: `task-metal-cgofree-spike.md` |
| **Intel** | oneAPI/SYCL or Level Zero | `libze` dlopen (new) | Full rewrite, smallest audience |

So AMD + Apple + Intel ≈ **~18 kernel ports across 3 languages + 3 bindings**, for the
*dense* path alone. Extend to MoE/MLA/Mamba later → multiply again.

## Priority gradient (if ever)

1. **Apple / Metal — highest *value*, worst *fit*.** Mac is the dominant local-LLM
   platform, so the audience is biggest. But Metal's Obj-C API resists the
   `dlopen libcuda` trick that made NVIDIA cgo-free — a native Metal path likely
   **reintroduces cgo** (or a gnarly purego-Obj-C dance), which fights goinfer's
   whole identity. Highest audience, hardest to keep in-lane. The one to *watch*.
   > **Correction (2026-07-15):** the "likely reintroduces cgo" half is weaker than
   > written — **ebitengine/purego ships an `objc` package** (cgo-free
   > `objc_msgSend`, dlopen libobjc) and Ebitengine's production Metal driver runs
   > on it (ebiten #3411). The dance is a shipped technique. The watch item now has
   > a written, gated go/no-go: **`task-metal-cgofree-spike.md`** (NOT SCHEDULED —
   > same demand gate as ever). The audience point and the treadmill verdict stand.
2. **AMD / HIP — best effort-to-reward.** HIP is a CUDA clone, so the kernels port
   semi-mechanically (`hipify`); the real new work is the cgo-free HIP binding.
   Cheapest native vendor to add.
3. **Intel — lowest priority** (smallest GPU-LLM install base).
4. **Vulkan (cross-vendor NVIDIA+AMD+Intel, one SPIR-V codebase) — skip.** It buys
   ≈nothing over WebGPU: wgpu *already* compiles WGSL → SPIR-V/Vulkan at roughly the
   same ceiling. The native-speed leaps are the *vendor-specific* APIs (CUDA, Metal),
   not the cross-vendor ones. Vulkan-native is WebGPU with extra steps.

## Verdict

- **Coverage is done.** WebGPU handles every popular card today; nobody is unsupported.
- **Native speed is a per-vendor treadmill** — a full multi-language kernel set +
  binding + parity + maintenance per vendor, for a shrinking slice of audience. The
  WebGPU backend exists *precisely so goinfer doesn't have to do this*.
- **NVIDIA-CUDA was the one justified exception** — biggest single target **and**
  gocudrv made it cgo-free-cheap. That combination is what cleared the bar; it does
  **not** generalize to the other vendors (Metal fights cgo-free; AMD/Intel are
  smaller).
- **Add a native vendor path only if a concrete adopter's demand clears the same high
  bar CUDA did.** Absent that, WebGPU is the answer for AMD/Intel/Mac. Watch Metal
  (Mac audience); accept it's the hardest to keep cgo-free.

Same discipline as `positioning-cuda-and-promotion.md`: the intersection (portable +
cgo-free + one binary) is the moat; per-vendor native kernels are a treadmill that
trades the moat for throughput on ever-smaller audience slices. Gate hard on demand.
