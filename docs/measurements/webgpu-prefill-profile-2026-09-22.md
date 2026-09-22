# R10 prefill profile: WebGPU batched prefill is 81–94% one kernel — the tiled W8A8 GEMM at ~1 TFLOPS, ~11% of f32 peak

`docs/tasks/red-october.md` R10's prefill investigation: "per-class time ... so the GEMM, the attention, the batched norms/rope and the KV write each carry a number. Then one arithmetic line: GFLOPS achieved by the GEMM class against the card's naive-f32 and its shared-memory-tiled expectations. The write-up registers the prefill band and names the kernel; the build is a separate step." RTX 2070 SUPER, driver 595.91.07, `int8int8`, resident WebGPU path, `PrefillLastW8A8`.

## Instrument

`gpu/prefill_prof.go` + `TestPrefill_dispatchProfile` (`gpu/prefill_dispatch_profile_test.go`): per-category wall-time accumulation wired into `PrefillLastW8A8` itself, nil by default (one nil-check per boundary when off — parity gates re-run after the edit: `TestResidentPrefillLast_parity` 0 diff, `TestLocalize_BiasEpilogue`, batched rms/rope/attn parity all green). Same method and same accepted trade-off as `cuda/prefill_decomp_test.go`: category boundaries are syncs, so the category sum runs near-but-under wall (the layer loop is profiled; LM head, final norm, input upload and the logits readback are outside it — category sum is 94–97% of wall in every cell). Decode's ablation-by-omission harness (`TestDecode_dispatchProfile`) does not transfer: `PrefillLastW8A8` is not a replayable step list, and skipping a class cascades into shape errors downstream.

**Two instrument defects were caught by built-in self-checks before any number was trusted, and both are now assertions in the test:**
1. `device.Poll(true)` waits only for SUBMITTED work; `PrefillLastW8A8` records dispatches into an encoder and submits every 32 passes, so the first version attributed time by flush cadence, not by category — attention read 544 ms at P=256 and 55 ms at P=512, impossible for an O(P²) kernel. Fixed: each boundary flushes the encoder before polling (profiling-on only). Self-check: attention must grow with P.
2. The first prefill in a process pays one-time driver JIT per WGSL pipeline, booked into whichever category dispatches each first — `normsRope` read 496 ms at P=256 vs 50 ms at P=512. Fixed: a discarded warm-up call, then best-of-2 per cell. Self-check: `normsRope` (O(P) elementwise) must not shrink with P.

## Decomposition (warmed, best of 2, category ms and share of category sum)

| model | P | wall ms | gemm | attn | normsRope | kvWrite | GEMM GFLOPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5B (h=1536, 28L, inter 8960) | 256 | 770 | 688 (**93.5%**) | 18.5 (2.5%) | 27.4 (3.7%) | 1.8 (0.2%) | 974.5 |
| 1.5B | 512 | 1544 | 1367 (**91.9%**) | 62.2 (4.2%) | 56.0 (3.8%) | 3.0 (0.2%) | 981.5 |
| 0.5B (h=896, 24L, inter 4864) | 256 | 264 | 210 (**84.9%**) | 13.2 (5.3%) | 22.5 (9.1%) | 1.7 (0.7%) | 873.4 |
| 0.5B | 1024 | 1015 | 790 (**81.1%**) | 146.6 (15.0%) | 35.7 (3.7%) | 2.0 (0.2%) | 927.7 |

Logs: `webgpu-prefill-profile-2026-09-22-1.5b.log`, `-0.5b.log`. The 1.5B has no P=1024 cell: `PrefillLastW8A8` does not chunk (unlike CUDA's `prefillChunked`) and its O(M·inter) scratch OOMs at M=1024 on this 8 GB card — a real limit of the shipped path, recorded, not silently narrowed; the doc's own prior P=1024 reference (`prefill-batched-ttft-2026-09-13.md`) was also the 0.5B.

Attention scales as expected (1.5B 18.5→62.2 ms for 2× P; 0.5B 13.2→146.6 ms for 4× P) and is a **small** share at these depths — the opposite of CUDA's pre-L2 situation. R5's attention-tile lever does not transfer to WebGPU prefill; the same "an analogy transfers its assumptions" lesson G36 recorded for decode.

## The arithmetic line, and the kernel named

GEMM FLOPs per token per layer = 2·hidden·(qDim + 2·kvDim + qDim + 3·inter) (int8 MAC = 2 FLOP). Achieved: **~975 GFLOPS on the 1.5B, ~900 on the 0.5B, flat across P** — the signature of a kernel-side ceiling, not a depth effect.

Card (repo-cited figures, `prefill-l2l3-phase0-2026-09-05.md`, `ollama-chase.md`): 2560 cores × 1770 MHz → **9.06 TFLOPS f32 FMA peak**; DP4A on Turing ≈ 4× → **~36 TOPS**. So the GEMM class runs at **~10.8% of f32 peak and ~2.7% of DP4A peak**. A naive untiled f32 GEMM on this card is memory-bound in the ~1–2 TFLOPS range; a shared-memory-tiled f32 GEMM is expected at 50–70% of peak (4.5–6.3 TFLOPS); a DP4A-tiled int8 kernel should exceed f32 peak. **The shipped kernel is at the naive-f32 level despite being tiled and despite DP4A being active.**

The kernel: `matmulTiledW8A8KernelWGSL` (`gpu/gemm.go`), instantiated as the DP4A variant (`hasDP4A=true` on this box — confirmed empirically via `probeDP4A`, so the `dot4I8Packed` path IS what runs here now, superseding the 09-13 "no dot4 symbol" note for this adapter). Shape: **16×16 tile (`TS=16`), `@workgroup_size(16,16)`, `As`/`Bs` shared tiles of 256 packed u32 each, `acc: i32` over the whole K reduction, scales applied once at the end** (`f32(acc) * aScales[row] * bScales[col]`). A 16×16 output tile with a K-strip of 16 packed words is a very small tile for this card: each thread computes one output element, so arithmetic intensity per shared-memory load is low and the 40 SMs see 256-thread workgroups with little register-level reuse — the shape a well-tiled kernel (register blocking, 2–4 outputs per thread, wider K strips) is built to fix. That is the diagnosis of the gap, from the kernel's own source; it is a reading, not a measured attribution, and the build should re-verify it with a register-blocked variant rather than assume it.

**Bit-identity is available by construction for any retile.** The entire K reduction is an exact `i32` sum (int8×int8 products, K ≤ 8960 → |acc| ≤ 8960·127² ≈ 1.4·10⁸, far under 2³¹) with the f32 scales applied exactly once at the end — integer addition is associative and exact, so reordering or re-blocking the K loop cannot change a single bit. A GEMM rewrite here needs no fidelity gate beyond `math.Float32bits` equality against the shipped kernel. (This is the W8A8 per-row-scale case; the int4 per-group case discussed in `task-moe-streaming.md`'s "int32-per-group" note is different and is NOT what this profile measured.)

## Pre-registered band for the build (R10's own requirement: "a build follows only against a registered band")

- **Lever:** a register-blocked / larger-tile rewrite of `matmulTiledW8A8KernelWGSL` (DP4A variant). Nothing else in this profile is worth a build: attention, norms/rope and the KV write together are 6–19%.
- **Cell:** this instrument, 1.5B at P=512 (GEMM = 91.9% of the category sum), warmed, best-of-2, arms in separate processes alternating; the shipped kernel is the do-nothing arm. Confirmation: 0.5B at P=1024. Wall-clock: `TestResidentPrefillLast_TTFT` on the same cells.
- **Ships:** ≥ **2.0×** on the GEMM class (≈ 1.85× whole prefill by Amdahl on the 91.9% share; → ~1.9 TFLOPS, still only ~21% of f32 peak — a modest target for a tiled kernel) AND bit-identical (`Float32bits`) to the shipped kernel on the parity fixtures AND no regression >3% on attn/normsRope/kvWrite. **Parked:** 1.3–2.0× on the class. **Killed:** <1.3× (the tile shape was not the wall; profile again before another attempt). **Ambiguous:** any ship-side result within its last 5% is parked, not shipped.
- Not started in this pass.

## Not done, stated

R10's Record line also asks for `benchmarks.md` §B8/§B10 WebGPU rows (decode; nothing here changes them) and for the stale 18.8 s WebGPU vision-tower figure in Table 1 to be re-measured or struck — **not done here**; it is a vision-tower re-measure, a separate instrument. Left open, named.
