# WebGPU / WGSL primer — the ecosystem and how goinfer's `gpu/` sits in it

> Orientation doc for anyone touching the `gpu/` module. **What kind of work is
> this?** Not C. It's **Go for everything structural + WGSL for the ~6 hot kernels
> + a Rust dependency you never open.** This doc maps the ecosystem so you know
> which repo answers which question, and where goinfer plugs in.

## TL;DR

- **WebGPU** is a cross-platform GPU API (the modern successor to WebGL); **WGSL**
  (WebGPU Shading Language) is its shader language — a Rust/C-flavored language for
  compute and graphics kernels.
- goinfer's GPU backend is **pure Go orchestration calling WGSL kernels**. The cgo
  is quarantined inside the `cogentcore/webgpu` dependency (behind `-tags gpu`); the
  default goinfer build stays pure Go.
- The stack under you, top to bottom: **your Go (`gpu/`) → `cogentcore/webgpu` (Go
  binding) → `wgpu-native` (C API) → `wgpu` (Rust) → `naga` (WGSL→native shader
  compiler) → Vulkan/Metal/DX12 → the GPU.**
- Three repos answer three questions: **gpuweb/gpuweb** = "is this legal WGSL?",
  **gfx-rs/wgpu** = "did the feature land, on which backend?", **cogentcore/webgpu**
  = "what's the Go API I write against?"
- What you actually write: Go (buffers, dispatches, decode-graph wiring, tests) and
  WGSL string literals (the kernels). You never write C++/Rust — that's downstream
  of the dependency boundary.

---

## 1. What WebGPU and WGSL are

**WebGPU** is the modern cross-platform GPU API — designed for the browser but
equally usable natively. It replaces the WebGL/OpenGL generation with an explicit,
Vulkan-style model: you allocate buffers, build pipelines, record command buffers,
and submit them to a queue. The same API runs on top of **Vulkan** (Linux/Windows/
Android), **Metal** (Apple), and **D3D12** (Windows). That "one API, three native
backends" property is exactly why goinfer chose it: one shader set, every platform,
no driver install, no CUDA toolkit (`gpu-assessment.md` §4).

**WGSL** (WebGPU Shading Language) is WebGPU's shader language — the code that runs
*on* the GPU. Its syntax is influenced by Rust, with strong static validation and
explicit resource binding. A compute kernel looks like this (the `dot4I8Packed`
probe from `gpu/spike_test.go`):

```wgsl
@group(0) @binding(0) var<storage, read_write> out: array<i32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = dot4I8Packed(0x01020304u, 0x05060708u);
}
```

Note what's WGSL-specific: `@group/@binding` (resource binding), `@compute
@workgroup_size(...)` (this is a compute kernel with N threads per workgroup),
`var<storage, ...>` (memory space). You think in **workgroups** (thread blocks) and
**lanes** (threads) — the warp-per-head attention in `gpu/attention.go` is one
workgroup per attention head, 128 lanes doing a tree-reduced softmax. That mental
model — workgroups, lanes, reductions, keeping intermediates off the critical-path
"spine," avoiding barriers — is the actual skill of GPU kernel work here.

It is **not C**. It's closer to a restricted GLSL/Rust hybrid: no pointers, no
recursion, no dynamic allocation, bounds-checked, designed to compile safely on
every backend.

---

## 2. The ecosystem, in four layers

### Layer 1 — the spec

- **[gpuweb/gpuweb](https://github.com/gpuweb/gpuweb)** — the W3C working group's
  repo. Holds the **WebGPU API spec** and the **WGSL language spec**. This is ground
  truth for *what WGSL guarantees*. Example: `dot4I8Packed` (packed 4×int8 dot, the
  DP4A builtin) was debated here in [issue #2677](https://github.com/gpuweb/gpuweb/issues/2677)
  before any implementation shipped it. **Read this when** you need to know whether
  something is legal/portable WGSL, not just whether one backend happens to support it.

### Layer 2 — the two reference implementations

Every native WebGPU program runs through one of these. They are the "browsers' GPU
engines," usable standalone:

- **[gfx-rs/wgpu](https://github.com/gfx-rs/wgpu)** — pure-Rust, cross-platform
  (Vulkan/Metal/DX12/GL). Ships a C API, **`wgpu-native`**, for non-Rust callers.
  **This is the implementation under goinfer.** When the DP4A blocker cleared, it
  was this repo's PRs [#7494](https://github.com/gfx-rs/wgpu/pull/7494)
  (`dot4I8Packed`/`dot4U8Packed` builtins) and
  [#7595](https://github.com/gfx-rs/wgpu/pull/7595)
  (`NATIVE_PACKED_INTEGER_DOT_PRODUCT` → `VK_KHR_shader_integer_dot_product`, the
  2070's TU104 DP4A units).
- **[google/dawn](https://github.com/google/dawn)** — C++, Chromium's WebGPU
  implementation. Same role, different language/owner. Not in goinfer's path, but
  it's the other half of "WebGPU support" you'll see referenced (e.g. React Native's
  WebGPU binding wraps Dawn).

### Layer 3 — the shader compilers (WGSL → native)

Each reference impl has its own WGSL compiler that translates your WGSL into the
platform's real shader language (SPIR-V for Vulkan, MSL for Metal, HLSL for DX12):

- **naga** — in the wgpu repo (Rust). The one under goinfer. It's *why* a WGSL
  builtin like `dot4I8Packed` actually becomes a hardware DP4A instruction on a
  given backend — feature availability is ultimately a naga + driver question, which
  is what `gpu/spike_test.go` probes at runtime.
- **Tint** — in Dawn (C++). The Chromium-side equivalent.

You don't touch these directly, but they're the layer where "WGSL feature X" turns
into "runs on my GPU" — or doesn't, hence the runtime capability probe.

- **[wgslrunner](https://github.com/hanawatson/wgslrunner)** — a handy tool that
  feeds the *same* WGSL into both Tint and naga and validates outputs. Exactly the
  kind of differential harness for catching cross-backend numerics drift that
  `gpu-assessment.md` §3 warns about (Metal vs Vulkan vs DX12 fma/rounding).

### Layer 4 — the Go bindings (where goinfer works)

- **[cogentcore/webgpu](https://github.com/cogentcore/webgpu)** — **goinfer's
  dependency** (`gpu/go.mod`, currently `v0.23.0`). A Go binding over `wgpu-native`.
  This is the API the `gpu/` module calls: `New()`, `device.CreateShaderModule`,
  `CreateComputePipeline`, `Queue.Submit`, buffer allocation, etc. It uses cgo —
  but the cgo is *inside this dependency*, sealed behind goinfer's `-tags gpu`
  submodule, so the default goinfer build is pure Go. It can also target **browser
  WebGPU under wasm without cgo**, which is what makes the `demo/gemma-web`
  client-side story possible.
- **[gogpu/wgpu](https://github.com/gogpu/wgpu)** — a *pure-Go* WebGPU
  implementation (no cgo at all). Philosophically aligned with goinfer's no-cgo
  ethos, but almost certainly far less mature than `wgpu-native`. Worth watching,
  not switching to.

---

## 3. How goinfer's `gpu/` plugs in

The stack, top to bottom:

```
goinfer  gpu/*.go            ← Go you write: buffers, dispatches, decode graph, tests
   │     (WGSL kernels as Go string literals: quant.go, gemv.go, attention.go, ...)
   ▼
cogentcore/webgpu            ← Go binding (the API you call)   [cgo lives here]
   ▼
wgpu-native (C API) → wgpu (Rust)        ← reference implementation
   ▼
naga: WGSL → SPIR-V / MSL / HLSL         ← shader compiler
   ▼
Vulkan / Metal / D3D12 → the GPU
```

**What you actually edit**, by file (live as of this writing):

- `gpu/backend.go`, `gpu/gpu.go`, `gpu/device.go` — Go: device/adapter setup,
  buffer arena, the `Backend` abstraction.
- `gpu/decoderunner.go`, `gpu/decodelayer.go` — Go: the resident decode graph
  (M=1 token forward, `Run(x, pos)`), how kernels chain per layer.
- `gpu/quant.go`, `gpu/gemv.go`, `gpu/gemm.go`, `gpu/gemv_w4a8.go`,
  `gpu/attention.go` — **the WGSL kernels** (W8A8/W4A8 GEMV, prefill GEMM,
  warp-per-head attention), embedded as Go strings, plus their Go launchers.
- `gpu/spike_test.go` — the runtime capability probe (adapter name, limits,
  `dot4I8Packed` support, timestamp queries).
- `decoder/residency.go` — the decoder-side seam (`ResidentForward`,
  `ResidencyBackend`, the eligibility gate) connecting `decoder.Generate` to the GPU.

**Practical map of "which repo answers my question":**

| Question | Repo |
|---|---|
| Is this legal / portable WGSL? | gpuweb/gpuweb |
| Did feature X land, and on which native backend? | gfx-rs/wgpu (PRs/releases) |
| Does my actual GPU expose it at runtime? | `gpu/spike_test.go` (probes naga + driver) |
| What's the Go API / method signature? | cogentcore/webgpu |
| Cross-backend shader output differs — why? | naga vs Tint (wgslrunner to diff) |

---

## 4. The cgo / pure-Go boundary (the architectural rule)

goinfer's whole position is that the default binary is **pure Go, no cgo**. The GPU
backend uses cgo — but only *transitively*, through `cogentcore/webgpu`, and only
under the `-tags gpu` build tag in a separate submodule (`gpu/go.mod`). You never
write C or Rust; that's all downstream of the binding. CI guards the core graph so
the cgo can't leak into the default build. This quarantine is **non-negotiable**
(`gpu-assessment.md` §4) — it's why goinfer can ship one pure-Go static binary that
runs CPU-only everywhere and opt into GPU without a CUDA toolkit or driver install.

---

## 5. Where to start reading WGSL

If you've written Go and can read a C-like language, learn the WGSL here from the
kernels already in the repo — in increasing order of complexity:

1. `gpu/spike_test.go` — a 4-line WGSL kernel; the minimal shape.
2. `gpu/quant.go` — the W8A8 GEMV (int8×int8 matmul, the hand-unpacked int8 dot —
   the thing `dot4I8Packed` would replace).
3. `gpu/gemv_w4a8.go` — int4 group-wise GEMV (nibble unpack + per-group scales).
4. `gpu/attention.go` — warp-per-head attention (one workgroup/head, tree-reduced
   softmax) — the kernel that landed the 3.1× decode win (`gpu-assessment.md` §0.0)
   and the one a batched-verify (spec-decode) `ForwardN` would extend to M=K.

For the broader concepts (workgroups, bindings, memory spaces, barriers), the WGSL
spec in gpuweb/gpuweb is the reference, but the four kernels above are the fastest
way in for this codebase specifically.

---

## Sources

- In-repo: `gpu/spike_test.go`, `gpu/quant.go`, `gpu/attention.go`,
  `gpu/decoderunner.go`, `decoder/residency.go`, `gpu/go.mod` (cogentcore/webgpu
  v0.23.0); `docs/gpu-assessment.md` (§3 cross-backend numerics, §4 cgo quarantine),
  `docs/gpu-next-levers-assessment.md`.
- Spec: [gpuweb/gpuweb](https://github.com/gpuweb/gpuweb),
  [WGSL — Wikipedia](https://en.wikipedia.org/wiki/WebGPU_Shading_Language).
- Implementations: [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) (PRs
  [#7494](https://github.com/gfx-rs/wgpu/pull/7494),
  [#7595](https://github.com/gfx-rs/wgpu/pull/7595)),
  [google/dawn](https://github.com/google/dawn).
- Go bindings: [cogentcore/webgpu](https://github.com/cogentcore/webgpu),
  [gogpu/wgpu (pure Go)](https://github.com/gogpu/wgpu).
- Tooling: [wgslrunner](https://github.com/hanawatson/wgslrunner).
