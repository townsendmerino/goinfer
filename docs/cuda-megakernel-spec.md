# CUDA cgo-free megakernel — engineering spec (Phase-1a of the spike)

> **⚠ Peer numbers below predate the Ollama v0.32.5 re-anchor (2026-08-04).** Competitive figures
> in this doc (e.g. Ollama-CUDA ~149, Ollama-Metal 83.3, llama.cpp-CUDA 72.8, and any "×Ollama"
> multiple) were measured against **Ollama 0.5.7 (2025-01) / Ollama-Metal 0.32.0 / llama.cpp as of
> v0.5.0** — historical working records, not current claims. Current same-box numbers vs Ollama
> **v0.32.5** are in `docs/benchmarks.md` §B2 (CUDA) / §B3 (Metal).


> **Companion to `task-cuda-cgofree-spike.md`.** That doc is the *decision* (go/no-go,
> triggers, scope); this is the *"implement exactly this"* map for the one fused
> decode-layer kernel the spike writes. Derived from a full read of the WebGPU
> resident decode path (`gpu/decoderunner.go` + friends). Dense **residency-eligible
> Qwen2/Llama decode only** — no MoE / MLA / Mamba / vision / prefill / QK-norm.
>
> **Status: CLOSED — the spike this belongs to has completed. Read as a record, not a
> task (2026-09-02).** `docs/completed/task-cuda-cgofree-spike.md` carries the full
> Phase-2 log: production backend, real-checkpoint parity, W4A8 coalescing to 80% of
> peak, the launch diet (18→13→8/layer), and §5.2's fusion itself — **K1 and K3a shipped
> as `cuda/fused_qkv.cu`** (behind `fuseQKV` / `GOINFER_CUDA_NO_FUSE`), **K2 built,
> measured at ~0%, and reverted.** `cuda/megakernel.cu` is the original scaffold and is
> now dead: the work landed in `fused_qkv.cu` instead, and nothing outside tests
> references it. §8's "box fills" list is therefore historical.
>
> The WGSL counterpart question — can §5.2's K1/K2/K3 grouping be rebuilt as 3 WebGPU
> dispatches? — was asked and answered **NO** in `docs/QUEUE.md` G35.

## 0. What must be reproduced (the correctness bar — read first)

The gate is **argmax-match to the pinned CPU golden**, NOT bit-identical logits. GPU
attention/RoPE run f32 while the CPU oracle accumulates in f64 → logits are cosine
≈1.0, not bit-exact (`gpu/attention.go:18`). The existing `TestWebGPU_forwardParity`
(`gpu/forward_parity_test.go:36`) asserts the greedy next-token **argmax equals
`testdata/gemma_forward_golden.json` `Argmax`**. The CUDA backend reuses that harness
with `Backend:"cuda"` and must clear the same token-identical bar. **Match the token,
don't chase raw bits** — but DO match the quant packing exactly (§3), because a
packing mismatch corrupts the token, not just low bits.

## 1. The seam (what the CUDA backend implements)

Drop-in on the same interfaces the WebGPU `DecodeRunner` already satisfies — nothing
in `decoder/` changes:

- `decoder.Backend` — `Name() string`, `MatmulBT(a,b,dst []float32, M,K,N int)`, `Close() error`
- `decoder.ResidencyBackend` — `BuildResident(m *Model) (ResidentForward, ok bool, err error)`
- `decoder.ResidentForward` — `Forward(embedding []float32, pos int) ([]float32, error)`,
  `ForwardN([][]float32, startPos int) ([][]float32, error)`, `UploadKV(layer int, keys, vals []float32) error`,
  `TruncateTo(pos int)` (no-op — positional cache), `Reset()`, `Close() error`
- registered via `decoder.RegisterBackend("cuda", factory)` on `init()` under `//go:build cuda`,
  keeping `libcuda`/gocudrv out of the core dep graph exactly like `gpu` keeps webgpu out.

`BuildResident` uploads weights **once**; `Forward` rewrites only the input embedding
+ the position-dependent uniforms (rope pos, KV-store base `pos*kvDim`, `nKeys=pos+1`)
per token — the same static-plan/per-token-delta split as `gpu/decoderunner.go:912`.

## 2. The per-layer op-DAG (what the megakernel fuses)

One dense layer today = **~13 WGSL dispatches** (`gpu/decoderunner.go:807-822`). Ordered
stages (H=hidden, I=intermediate, hd=headDim, nH/nKV heads, kvDim=nKV·hd, half=rotaryDim/2):

| # | stage | in→out | notes |
|---|-------|--------|-------|
| 1 | RMSNorm(x) → **per-row int8 quant** | x[H] → aq[int8], as[1] | fused (`decodefuse.go`) |
| 2 | Q/K/V projections (quant GEMV) | aq × Wqkv → q[nH·hd], k[kvDim], v[kvDim] | +optional Qwen2 bias |
| 3 | RoPE(q) + RoPE-store(k) + store(v) | q rotated; k→kCache[pos·kvDim], v→vCache[pos·kvDim] | fused `qkvFinalize` |
| 4 | attention (QKᵀ, online softmax, ·V) | q × kv[0..pos] → ctx[nH·hd] | GQA `kvh=qh/group`, `nKeys=pos+1` |
| 5 | quant(ctx) → int8 | ctx[nH·hd] → cq, cs | |
| 6 | O-proj **+ residual** | cq × Wo → x[H] += | fused epilogue |
| 7 | RMSNorm(x) → **int8 quant** | x[H] → mq, ms | fused |
| 8 | gate + up projections | mq × Wgate→g[I], mq × Wup→u[I] | |
| 9 | SwiGLU **+ int8 quant** | (g/(1+e⁻ᵍ))·u → dq, ds | fused |
| 10 | down-proj **+ residual** | dq × Wdown → x[H] += | fused epilogue |

End of stack, once: RMSNorm→quant → LM-head GEMV → logits[vocab].

## 3. Quant packing — MUST match bit-for-bit

**W8A8** (`gpu/quant.go`, `gpu/gemv.go:41`): per-row symmetric int8, `scale[i]=maxabs(row)/127`,
row padded K→mult-16, 16 int8 per coalesced `vec4<u32>` load. Math: int8×int8 → i32
accumulate, then `×aScale×bScale[n]`. **Activation** = per-token int8, one scale =
maxabs/127 (produced inside the fused rms/swiglu kernels).

**W4A8** (`gpu/gemv_w4a8.go`): **group=32**, per-group `scale=maxabs/7` stored **f16**,
nibble = `q+8 ∈ [1,15]` (8=zero). Packed layout: element k → u32 word `k/8`, nibble
`4*(k%8)`; row padded to mult-32; a 32-nibble group = one coalesced `vec4<u32>`. Math:
`Σ(nibble−8)·int8act → i32`, `×f16 group-scale`, `×aScale`. The decoder's int4 storage
is **byte-identical** to this packing when K%32==0 (`TestInt4LayoutMatch`) → weights
upload with **no repack** (`UploadW4A8Packed`).

## 4. KV cache on device (spike = f32 only)

Flat `array<f32>` length `ctxCap·kvDim`, element layout **`[pos·kvDim + head·hd + d]`**.
K stored **post-RoPE**; V raw. Attention reads `s ∈ [0, nKeys)`, `nKeys=pos+1`. RoPE:
`θ = pos·invFreq[d]`, rotate pair `(vec[off+d], vec[off+half+d])` for `d<half`; partial
RoPE leaves trailing `hd−rotaryDim` dims untouched; per-layer YaRN mscale applied.
(f16/int8 KV variants exist in WGSL — out of spike scope; match the f32 default.)

## 5. The launch-structure decision (the key CUDA-specific finding)

The megakernel's premise — collapse ~13 dispatches → ~1/layer — needs **grid-wide sync
*inside* the kernel** (stages are data-dependent, and each stage spans many blocks:
GEMV = N/tile blocks, attention = one block/head). On CUDA that's `grid.sync()` via
**`cuLaunchCooperativeKernel`**.

**gocudrv (as of the ten-weekends writeup) does NOT expose `cuLaunchCooperativeKernel`.**
Three ways forward, in order of preference:

1. **Add cooperative launch to gocudrv** (recommended). It's one more dlopen'd driver
   symbol — stays `CGO_ENABLED=0`, ~a day. Then a true 1-launch/layer megakernel with
   `cg::this_grid().sync()` between stages. Occupancy cap: total blocks ≤ SM-count ×
   blocks/SM (fine at decode's small grids).
2. **Decompose into ~3 fused super-kernels/layer** at the natural cross-block sync
   boundaries — no cooperative launch needed:
   - **K1 (pre-attn):** RMSNorm+quant → QKV GEMV → RoPE+KV-store
   - **K2 (attn):** attention → quant(ctx) → O-proj + residual
   - **K3 (ffn):** RMSNorm+quant → gate/up GEMV → SwiGLU+quant → down + residual
   Still ≪ 13 dispatches, still amortizes the ~16µs/call channel-hop, and each
   super-kernel fuses its intra-block stages in shared memory. **This is the low-risk
   spike path** — it tests the megakernel thesis without the gocudrv gap.
3. Persistent-threads / single-block-per-layer — bounded by H and shared memory; tight
   at 1.5B (H=1536). Not recommended for the first cut.

**Spike plan: start with (2) — 3 super-kernels — and only invest in (1) if (2) already
clears the WGSL wall and the remaining launch overhead is the bottleneck.**

## 6. Layer-A plumbing → gocudrv mapping (the easy 20%, already covered)

`cuda.Init` · `Device` + primary context · `Alloc[T]` (weight upload at BuildResident;
per-token embedding H2D) · `Memcpy` H2D/D2H · `ModuleLoadData`(embedded PTX) → `Function`
· `LaunchKernel` · `Stream` · `Event`(the CUDA-event timing — mandatory per gocudrv's
own 18× lesson: `time.Since` measures enqueue, not kernel). Calls run on gocudrv's
pinned per-context executor goroutine (the channel-hop). **Missing from the 106 syms:
cooperative launch (§5).**

## 7. Measurement (the deliverable) + go/no-go

Per `task-cuda-cgofree-spike.md`: CUDA-event-timed, warm, ≥3 runs, equal quant, same
2070 SUPER. **First step on the box: re-measure current *coalesced* WebGPU decode tok/s**
— the `benchmarks.md §B` baseline (89.7 int8 / 51.7 int4) predates the coalescing win
(`f8ef42b`/`5c3777f`) and is stale (**§B was RETIRED 2026-08-27** and now lives in
`benchmarks-archive.md`; §B8 has current WebGPU-vs-own-CUDA rows); the 1.3× GO bar keys off the fresh number.
Report: CUDA tok/s vs fresh WebGPU + llama.cpp-CUDA (72.8, 7B int4); JIT cold-start;
channel-hop tax at the layer's launch count; confirm `CGO_ENABLED=0` + single static
binary on a driver-only box. **GO = ≥1.3× fresh-WebGPU, cgo-free holds, cold-start OK.**

## 8. What the skeleton (`cuda/`) provides now vs the box fills

- **Provided (compiles under `-tags cuda`, no GPU):** the module, the three interface
  impls (`Backend`/`ResidencyBackend`/`ResidentForward`), the internal `driver` seam +
  a not-wired stub, the per-token Forward flow (embedding H2D → launch → logits D2H),
  the `.cu` scaffold with the 3 super-kernel entry points, the `go:generate` nvcc→PTX
  hook. `BuildResident` returns `ok=false` (clean fallback) until the box wires it.
- **Box fills (Phase 2):** the real `.cu` math (GEMV/RoPE/attn/SwiGLU per §2–§4), the
  gocudrv-backed `driver` impl (+ cooperative launch if going route §5.1), weight
  extraction in `BuildResident`, `nvcc` compile, and the measurement run.
