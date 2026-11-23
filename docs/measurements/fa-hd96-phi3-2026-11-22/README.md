# The CUDA flash-decode lane at head dim 96 (Phi-3 mini): registration (2026-11-22)

**Status: REGISTERED 2026-11-22, APPROVED (Francis, 2026-11-22, about 15:40 PST: "Yes build it", relayed by the coordinator).** Gates in order, each passing before the next, fixed N, no early stop; if it ships it lands behind an opt-in, and the default is Francis's separate call.

**Question.** Does a head-dim-96 variant of CUDA's flash-decode lane decode Phi-3 mini (`q4k`, f16 KV) as well as the path it
decodes on today, and faster at depth? Today `faEligible` admits hd 64, 128 and 256 only, so Phi-3's 32 layers (hd 96, MHA, G = 1)
decode on the exact attention path at every depth; split-KV stays off for it (Francis, 2026-11-19, the split-KV record). Review
finding 1(b) and [`splitkv-phi3-2026-11-19/record.md`](../splitkv-phi3-2026-11-19/record.md) name this lane as the next lever: about
half the 16.1 ms token at depth 3,900 is depth-dependent, and Phi-3 is BEHIND both peers on CUDA (0.79× Ollama at 3,900).

**The change.** `fa_partial_96` and `fa_partial_rows_96_g1` (and its ragged twin) in `decode_fa.cu`: the score dot is unchanged (NQ = 6
float4 steps), and the P·V step takes its own lane mapping, since 96 dims over 32 lanes is 3 a lane (the template's VEC = 2 gives a
non-integer NCH). `faEligible` admits hd 96. The partial buffer is already sized to the model's largest head dim and allocated for
Phi-3, so the device bytes do not change; the PTX modules grow by the new kernels.

**Gates, in order. Each must pass before the next runs.**
1. **Tier 0, correctness:** `fa_partial_96` (+ `fa_combine`) and the rows kernel against the float64 reference
   (`TestAttnSoftcap_flashDecodeAgainstFloat64`'s pattern: cosine ≥ 0.9999, max abs ≤ 1e-2; windows 0 and 100; 4 and 16 splits;
   verify batches 1, 5 and 16), with a plant (a lane dimension dropped) going red. `TestAttnKernelDigests` (G1-K): every existing
   kernel's digest unchanged.
2. **Fidelity, registered now:** `TestFlashDecodeGateVsReference` with a new `P3M` cell, `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`,
   `q4k` on CUDA, K = 3,900, held-out set B, against a new `TestPrefillGateReference` `P3M` row (CPU f32 from the same GGUF, as G22B).
   The criteria, preconditions and ambiguous band of
   [`attn-decode-fa-fidelity-PREREGISTERED.md`](../attn-decode-fa-fidelity-PREREGISTERED.md), unchanged: (a) hard flips ≤ exact +
   2√exact; (b) agreement ≥ exact − 1.0 pt and lane ≥ exact on at least half the prompts; (c) mean KL ≤ 1.10 × exact's.
   AMBIGUOUS → PARKED: (b) within its last 0.2 pt, or (c) in 1.05–1.10×. The GGUF declares no `sliding_window`, so goinfer attends
   the full context (`phi3Architecture`), and the lane's 2,048-key floor is crossed from depth 2,048 on.
3. **Speed (TE5(b), then TE5(a)):** an in-process whole-token A/B, `BenchmarkResidentDecode` with serve's f16 KV, lane on against off
   (`GOINFER_CUDA_FLASH_DECODE=0`) in one binary, A B B A × 5 blocks (10 pairs) at depths 3,900 and 128. Then a served two-arm A/B (`goinfer` against
   `goinfer_old`, one binary, the flag on one arm), depths 128 and 3,900, `BENCH_RUNS=2`, `BENCH_ABBA=1`.
   **SHIP:** the served paired median ≥ 1.02 at 3,900 and ≥ 0.98 at 128. **REGRESSION:** < 0.98 at 3,900. **Between:** parked.
   An in-process result that does not resolve its direction (pairs straddle 1) still goes to the served cell.

**Stopping rule:** fixed N at every gate; no early stop. A failed gate stops the sequence there.

**What each outcome means.**
- All three pass and SHIP: the lane takes hd 96 behind an opt-in until Francis decides the default; the PR states device and host
  bytes and the longest context that fits.
- Gate 1 fails: fixed before gate 2 is spent (gate 1 is a defect detector, not a decision).
- Gate 2 FAILs: hd 96 stays off the lane; the record names the criterion, and the kernel is not merged.
- Gate 3 parks or regresses: the kernel is not merged; the record says which.

**Cost:** gate 1 about 15 min of build and card tests. Gate 2's reference about 60–80 min of CPU (the 2B took 37 min at K = 3,900) and
the gate about 5 min under the exclusive lock. Gate 3 about 25 min in-process and 10 min served. By day; nothing past 21:30 PST.

## Gate 1, 2026-11-22 17:18–17:23 PST (nobara, RTX 2070 SUPER, driver 595.112.07): PASS

- **The kernels' code:** `fa_partial_96`, `fa_partial_rows_96_g1` to `g8` and `fa_partial_ragged_96`, at `0d4843a18`. The PTX was rebuilt at
  the ambient NVRTC 12.9.86 (`build_ptx.sh decode_fa`).
- **`TestAttnKernelDigests` (G1-K):** 2,152 digests equal base's on this card and driver. Every existing kernel is unchanged
  ([`gate1/gate1.log`](gate1/gate1.log)).
- **`TestFlashDecodeHD96_againstFloat64`:** all 288 cases are inside the bar (cosine ≥ 0.9999, max abs ≤ 1e-2), at f32 and f16 KV:
  - the M = 1 lane at 4 and 16 splits;
  - 704 verify rows, each also bit for bit against the M = 1 lane at its depth;
  - 32 ragged rows, bit for bit against the M = 1 lane.
- **The plant:** the P·V load dropped one of each lane's three dims (`v[c] = 0` at c = 2). 896 comparisons fell outside the bar
  (cosine about 0.77; `gate1/plant-red.log.gz`). The bit-identity checks stay green under it, as they should, since both sides carry
  it. Restored, the PTX rebuilt to the committed bytes (sha256 `74627d34…`).
