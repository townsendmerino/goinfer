# Task (goinfer + aikit): the DeltaNet recurrence — the last unvisited ~19% of the 35B token

> **For:** Claude Code, in `~/tmcode/goinfer` + `~/tmcode/aikit`, on the M1 Pro. Written
> 2026-08-25. **Prior art (read all three before any code — this section is mandatory now):**
> `docs/queue-performance.md`'s quantization entry already names this exact lever ("the
> DeltaNet recurrence is scalar Go" under "Still open"; the kernel question "belongs to
> aikit"); `docs/deltanet-residency-plan.md` proved the compute parallelizes (WebGPU resident
> at 11.4-12.2x CPU decode, and the head_dim-256 wall is documented there); the 35B diagnostic
> (`docs/task-zeno-compare.md`) measured DeltaNet at **~19% of the decode token**, the largest
> component no perf campaign has ever visited. This brief funds the CPU side only.

## Gate D0 — the sixth split, plus the invariant enumeration (diagnosis before design)

1. **Split the ~19% at the real 35B shapes**: the big projections (`in_proj_qkv` [10240,5120]
   etc. — already W4A8, presumably fast) vs the delta-rule recurrence proper
   (`gatedDeltaNetStep`, `decoder/deltanet.go` — the "3. Gated delta-rule recurrence, per
   value head" block) vs conv window, gating, and `l2normScaled`. Same env-gated-stub method,
   same sums-to-~100% bar. Do not assume the recurrence dominates — measure it; the
   projections were the surprise last time anyone assumed.
2. **Rate check on the recurrence**: ns per state-element update vs the ~1.4 ns/op
   serial-chain signature. If it runs at scalar-chain speed (it is written as plain Go loops),
   the A1 playbook transfers directly.
3. **Enumerate the exactness surface before designing**: which gates are exact — the DeltaNet
   golden and qwen3_5_moe forward parity are documented as bit-unchanged under the
   quantization change, so treat them as exact until proven tolerance-based. Note
   `deltanet.go`'s own `matvec` comment already flags SIMD-reassociation as a known
   consideration — find out what constraint the original author was honoring.

## D1 — the ladder, cheapest-invariant-preserving first (A1's playbook, new substrate)

The recurrence is sequential **across tokens**; within one step, the state matrix's elements
are updated independently — the same independent-axes-never-the-reduction structure A1
exploited (`task-decode-splitkv-attention.md:36`). So:

- **(a) SIMD across independent state elements** — vectorize the per-step update across the
  state's value dimension with each element's own fold order unchanged: bit-identical by
  construction, zero golden churn, the default plan. NEON f32 4-lane on contiguous state rows;
  the Sᵀk dot products interleave across independent outputs the way A1(b) did.
- **(b) Thread across value heads** — each head's state is independent; per-head chunks at
  real shapes are the fan-out candidates, subject to the same worker-scratch discipline as
  A1(a). Measure the chunk size before assuming it clears fork-join overhead.
- **(c) Reassociating variants** (wider lanes, fused forms) only if (a)+(b) measure short —
  behind the cosine re-gate + regold, priced as such, never silently.

Kernels land in `aikit/linalg` per the queue entry's own placement call; goinfer wires them.
Exact-equality tests against the current scalar step (`==`, every shape residue of the lane
width) before any perf claim — the house bar.

## Acceptance

- Derive the projection band in the doc before building: fresh Amdahl from D0's split and the
  current 35B ms/token, stated, then held to. Working target: **DeltaNet component ≥3x** at
  the real shapes if D0 confirms scalar-chain behavior; end-to-end whatever the band says.
- All exact gates green with zero golden churn for (a)+(b); `-race` on the threaded path.
- Quiet box; `b.Run` shapes; order-alternated; measured cells recorded in the campaign doc
  with the split refilled as the after.
- **Cross-family note in the writeup**: every Gated-DeltaNet hybrid (qwen3_5_moe /
  qwen3_next / Qwen3.8) shares this code path — state which families the win applies to and
  re-run the cheap goldens for each.

## Not in scope

- CUDA/WebGPU DeltaNet residency — its own track per the queue; this brief is CPU-only.
- The qwen35-family dense GGUF loader gap (separate queue entry), the chunked-scan batched
  prefill (roadmap item), anything MoE/paging/kind-4 (that arc is closed), attention, W4A8.
- Quantizing the recurrence's activations or state — numerics surface, not this pass.
