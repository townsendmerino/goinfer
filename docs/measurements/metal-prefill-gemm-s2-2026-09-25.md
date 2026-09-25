# Metal prefill GEMM redesign — S2 (R16): prior-art read (2026-09-25)

`docs/tasks/red-october.md` R16 pre-registers this item (ship ≥ 2.85× / park 1.8–2.85× / kill < 1.8× on the
in-sequence GEMM category at K=512, 1.5B) and requires the prior-art read before any kernel is written. This is that
read. **No prototype exists yet; nothing here is a measurement.** The design section is a proposal the prototype will
be measured against R16's band.

## Sources read, and which of them ran

- **llama.cpp / ggml 0.22.0**, `kernel_mul_mm`, read from the Metal source **embedded in the Homebrew
  `libggml-metal.so`** (`/opt/homebrew/Cellar/ggml/0.22.0/libexec/`) — the exact build S1b measured
  (`metal-gemm-ceiling-2026-09-25.md`), not whatever is on GitHub today. The source has two versions: a Metal-4
  `mpp::tensor_ops::matmul2d` path under `GGML_METAL_HAS_TENSOR`, and the classic simdgroup-matrix path. **On this M1
  Pro the classic path ran**: S1b's own init log reads "tensor API disabled for pre-M5 and pre-A19 devices … has tensor
  = false". So the ≥ 2.96 TFLOPS S1b measured comes from the classic kernel, and that is the one compared below.
- **MLX 0.32.0**, `qmm_t_impl` + `QuantizedBlockLoader` (`mlx/backend/metal/kernels/quantized.h`, Homebrew). Not
  measured in S1; read for its structure.
- **goinfer**, `gemm_w4f16_store` (`metal/prefill.go:37-106`) at `4d1083b1`.

## The three kernels side by side

| | goinfer `gemm_w4f16_store` | llama.cpp `kernel_mul_mm` (classic) | MLX `qmm_t` |
|---|---|---|---|
| threads / threadgroup | 256 (8 simdgroups) | 128 (4 simdgroups) | 128 (WM×WN = 2×2) |
| output tile per threadgroup | none shared — each simdgroup owns 32 tokens × 32 features on its own | **64 features × 32 tokens** | **32 × 32** |
| per-simdgroup tile | 32 × 32 (16 accumulators) | 32 × 16 (8 accumulators) | 16 × 16 |
| K step per iteration | **8** | **32** | **32** |
| weights staged in threadgroup memory | per simdgroup, private (`wscr + sgid*…`) — not shared | once per threadgroup, shared by all 4 | once per threadgroup, shared |
| activations staged in threadgroup memory | **no** — each simdgroup `simdgroup_load`s its A tiles straight from device memory, stride K, inside the MMA loop | yes, cooperatively (8 values per thread) | yes, cooperatively (`BlockLoader`) |
| bank-conflict layout | none needed (private 8×8 tiles) | blocked 8×8 tiles (`ib` index) | padding (`BK_padded = BK + 16/sizeof(T)`) |
| barriers | 2 simdgroup barriers per 8-K step | 2 threadgroup barriers per 32-K slab | 2 threadgroup barriers per 32-K slab |
| inner loop, per simdgroup, per 32 of K | 64 MMAs, **16 device loads**, 16 TG loads, 8 barriers | 32 MMAs, **0 device loads**, 24 TG loads | (BlockMMA over 32 of K) |
| weight reads | each lane reads one `uint32` + one scale from a **different weight row** per 8-K step (32 rows per simdgroup) | each thread dequantizes one 16-weight chunk per slab | `n_reads` packed words per thread per slab |
| epilogue | via TG float scratch, fused bias / residual per element | direct `simdgroup_store` of f32 (no fusion) | `store_result` |

(Counts for goinfer are from the source: per 8-K step, 4 A loads from device, 4 B loads from TG, 16 MMAs, 2
barriers — so per 32 of K, 16 / 16 / 64 / 8.)

## What separates ~0.75 from ≥ 2.96 TFLOPS — the read's answer, to be tested

Both peers share four things goinfer's kernel lacks:

1. **Both operand tiles are staged once per threadgroup and shared by 4 simdgroups.** goinfer stages nothing
   between simdgroups; every simdgroup fetches its own activations from device memory.
2. **The device loads leave the MMA loop.** In the peers the inner loop reads only threadgroup memory; device traffic
   happens cooperatively, coalesced, once per slab. In goinfer every 8-K step waits on four strided 8×8 device loads
   before its MMAs can run, so the loop is exposed to memory latency with little to hide it.
3. **K advances 32 at a time**, so barrier round trips are 4× fewer per unit of work.
4. **The staged layout is designed for conflict-free matrix loads** (llama.cpp's blocked 8×8 index, MLX's padding).

The read suggests (2) is the one that matters most, because it would also fit S0's unexplained burst: a
latency-bound loop is sensitive to memory-system state, and would run faster right after idle while a compute-bound
kernel does not — which is what S1a found for MPS. **That is a hypothesis, not a finding**; the prototype's
sustained-versus-after-idle timing, which R16 requires anyway, is where it gets tested.

## Design proposal for the prototype

A classic-`kernel_mul_mm`-shaped kernel adapted to goinfer's data, test-only until R16's band is met:

- **Layout fit.** goinfer's weights are already the peers' "transposed right operand": W is [N features × K],
  contiguous in K, 8 nibbles per `uint32`, one f16 scale per 32 weights (`WS[n*K/32 + k/32]`). Activations A are
  [Mpad × K] f16 row-major. Output C [Mpad × N] f16.
- **Tile.** 128 threads (4 simdgroups); threadgroup output 64 features × 32 tokens; each simdgroup 32 × 16 (8 f32
  accumulators); **K slab 32 = exactly one scale group**, so each staged weight row needs one scale per slab.
- **Staging per slab.** Weights: 64 rows × 32 = 2048 values, 16 per thread (2 `uint32` words + 1 scale),
  dequantized with the current kernel's exact formula (`half(float(int(nib) − 8) * sc)`) into a blocked 8×8 layout.
  Activations: 32 tokens × 32 = 1024 halves, 8 per thread (one 16-byte load). 6 KB of threadgroup memory per slab
  (4 KB weights + 2 KB activations), well under the M1's 32 KB, leaving room to double-buffer.
- **Epilogue.** Keep goinfer's fused modes (plain / +bias for QKV / +residual for o and down) — a peer kernel writes
  f32 and fuses nothing, so this part is goinfer's own — through a threadgroup scratch once per tile.
- **Fidelity, possibly bit-identical.** Each output element still accumulates K in 8-wide chunks, in order, into an
  f32 `simdgroup_float8x8` via the same `simdgroup_multiply_accumulate`, from identically dequantized f16 weights. If
  that holds end to end, the prototype's output is bit-identical to today's kernel and the §3.2 gate is met
  trivially. It has to be checked, not assumed (the MMA's internal 8-term order is hardware-defined, but the same
  instruction on the same operands).
- **Arms, per R16.** Current kernel (the do-nothing arm) and the prototype, same session, in sequence (leave-one-out),
  sustained and after 2 s idle, 1.5B and 7B shapes, K=512 deciding; then double-buffered staging as a second
  prototype if the first lands in the park band.

## Not settled by the read

- Whether 64 × 32 is the right threadgroup tile for goinfer's shapes, where gate/up's N = 17,920 gives 280 tiles
  across and M = 512 gives 16 down (llama.cpp's shapes are the same models, so probably, but not measured).
- The qkv and o shapes (N = 2048 / 1536) have fewer tiles; occupancy there is untested.
- The 7B (H=3584, I=18944).
