# PRE-REGISTERED — R5 phase 1: a 32-row query tile for `attn_fused` (P24)

**Written 2026-09-21 BEFORE the kernel variant exists. Not edited after any result was seen.** The record goes in `attn-fused-tile-2026-09-21.md`; this file stays as written.
Context: `docs/tasks/red-october.md` R5; baseline restored by `prefill-chunk-demotion-2026-09-21.md` (K=3900, 1.5B: attention 804 ms of a 1423 ms prefill, GEMM 529 ms).

## The question
`attn_fused` at K=3900 runs a 12 x 61 grid of 128-thread blocks (BM=64, BN=64) at ~12.6% occupancy and 1.7% of tensor peak; blocks are unequal (causal: block j does j+1 key tiles) and the tail of the grid runs on few SMs.
Does a smaller query tile (more, cheaper blocks; causal work per block halves) buy the attention category >= 1.25x, with the same numerics?

## Arms (one new module `attn_fused_bm.cu`, `template <HD, BM, BN>`; the shipped `attn_fused.cu`/PTX is not touched)
- **A0 = 64x64**: the shipped kernel (do-nothing arm, also the control for drift).
- **A1 = 32x64** (BM=32, 64 threads/block, BN unchanged): the R5 "BM=32 grid arm". Shared memory per block is unchanged, so it can raise block count but NOT resident-thread occupancy; it is the arm R5 names.
- **A2 = 32x32** (BM=32, BN=32): halves shared memory per block too, the arm that can actually raise occupancy. Added at registration because A1 alone cannot test the occupancy mechanism. Not on R5's list; it is judged by the same rule.
- A K-split arm is NOT built in phase 1 (R5 says only "if BM=32 disappoints"; the decision below covers that).

## Expected numerics (registered so a surprise is visible)
- A1 is **bit-identical** to A0 for window=0 (each row sees the same key tiles in the same order; extra fully-masked tiles are exact no-ops). Gate: `math.Float32bits` equal on random and real K/V, several M, startPos > 0, GQA. With a sliding window the tile grouping depends on the tile's first row, so bit-identity is NOT expected there and is tested against `_vsF16Reference` only.
- A2 changes the reduction order (BN=32): NOT bit-identical; it must pass `TestAttnFused_vsF16Reference` at the same bar as A0 and the §3 fidelity gate.

## Measurement
`TestPrefillDecomp` attention category (the gemv category is the control: it must not move beyond ±3%), K in {512, 1024, 2048, 3900}, 1.5B int4, best of 3, arms interleaved in one session (A0,A1,A2,A0,A1,A2...), idle box, same binary (arm chosen by an env knob read at backend setup, default = A0). Also 0.5B at 3900 (hd64 is the second instantiation) and a gemma3-1b-class window model is NOT measured in phase 1 (recorded as unmeasured).

## Decision rule (R5's band, applied to the better of A1/A2 by the attention category at K=3900; A0 re-measured in-session is the denominator)
- **Ships** (offered as default, pending the fidelity gate for a non-bit-identical arm): >= 1.25x AND K=512 attention no slower than A0 by more than 3% AND the gemv control within ±3%.
- **Parked**: 1.05-1.25x, recorded, no claim. **Below 1.05x**: parked-as-null.
- **Phase 2 (FA-style 64-128 row tile, R5's >= 2.5x band) is not started** unless phase 1 ships. A phase-1 park means the tile-size lever is not the mechanism and R5 needs an ncu diagnosis first, not a bigger kernel.
- An A0-vs-A0 floor from the interleave is reported; effects inside it are not read.
- If A1 ships it is bit-identical and needs no new fidelity gate beyond the bit test; if only A2 ships, the §3 gate (S at K=3900, and floor K=512) must pass before it is default.

---

## ADDENDUM — written 2026-09-21 AFTER the phase-1 timing, BEFORE any further run

**Phase-1 result (read under the rule above, no further interpretation): A1 32x64 = 2.86x SLOWER than A0 at K=3900 (2307 vs 807 ms attention); A2 32x32 = 1.19x slower (963 ms); all K, all 3 interleaved rounds (spread < 0.5%); gemv control within ±0.5%.
Neither arm is >= 1.05x, so phase 1 is a NULL/regression, the BM=32 lever is dead, and phase 2 (FA-style tile, shipping claim) is NOT started.**

**Mechanism hypothesis (not established; a reading of the sign):** the kernel is bound by K/V STAGING (f32 global load -> f16 -> shared, once per block per key tile), which is per-block cost, so halving BM doubles it per query row (and 64-thread blocks hide its latency worse). If that is the mechanism, **a TALLER tile amortises it and should be FASTER**, the opposite of the R5 brief's direction.

**Diagnostic arm A3 = 128x64 (BM=128, 256 threads, same shared memory, same registers per thread; bit-identical to A0 for window=0 by the same argument as A1).** It is a DIAGNOSTIC of the staging hypothesis, not phase 2 (phase 2's ship band applies only to a designed FA tile), and it is judged only as follows, fixed now:
- Confirms the staging mechanism if attention at K=3900 is **<= 0.85x of A0** (>= 1.18x faster) with the gemv control unmoved; A3 must also pass the same bit-identity test and `TestAttnFused_vsF16Reference`.
- Refutes it if A3 is >= 0.97x of A0 (no faster, or slower): then the 32x32-vs-32x64 gap is a shared-memory/occupancy effect and staging is not the wall.
- 0.85-0.97x: ambiguous, recorded, no claim.
A confirmation does not by itself ship anything: it names where R5's real lever is (staging), and a shipping kernel is a fresh pre-registered item against R5's own bands.
