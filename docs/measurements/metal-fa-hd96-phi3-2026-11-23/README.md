# Metal's attention_fa at head dim 96 (Phi-3 mini): registration (2026-11-23)

**Status: REGISTERED 2026-11-23, before any gate reading.** Owner's go: Francis, 2026-11-23 ~08:45 PST ("let's finish the
performance and webgpu parts" of the fresh-eyes review ledger), relayed by the coordinator. Perf review finding 10. Gates in order,
each passing before the next, fixed N, no early stop. If it ships it lands behind the existing opt-in `--flash-decode-hd96`
(`Options.FlashDecodeHD96`, today CUDA only); the default is Francis's separate call. The template is CUDA's hd-96 lane,
[`fa-hd96-phi3-2026-11-22/README.md`](../fa-hd96-phi3-2026-11-22/README.md): the same gates and bars, on Metal.

**Question.** Does a head-dim-96 twin of Metal's split-KV decode attention decode Phi-3 mini (default `q4k`, f16 KV) as well as the
exact attention kernel it decodes on today, and faster at depth? Today `attnFAHeadDimOK` admits hd 128, and 64 at G = 7, so
Phi-3's 32 layers (hd 96, MHA, G = 1) decode on the per-query-head `attention` kernel at every depth.

**The change.** `attention_fa_blk96<1>` (and its batched-step rows form): `attention_fa_blk64`'s shape at 3 dims a lane
(`packed_half3` K/V loads, `float3` q and accumulators), the same grid, split count (`attnFABlkSplit`, 16) and partial layout, so
`attention_fa_combine` merges it unchanged. `attnFAHeadDimOK` admits hd 96 at G = 1 only under the option. Metal's own key floor
applies (`attnFADepthFloor`, 1,024; CUDA's lane floor is 2,048). Device bytes: with the option off, none; with it on, the partial
buffer, 32 heads × 98 floats × 32 splits × 4 B = 401 KB, plus the batched step's per-row copy.

**Gates, in order.**
1. **Tier 0, correctness** (the Mac): `TestAttnFABlk96MatchesFloat64` (cosine ≥ 0.9999 and max abs ≤ 1e-2 per head against float64;
   4 and 32 KV heads; 4 and 16 splits; 1,024, 1,025, 3,900 and 4,100 keys; near-uniform and peaked), with
   `TestAttnFABlk96_plantGoesRed` (one lane dimension dropped from P·V) red under the same bar; `TestAttnFAHeadDimOK_hd96` and
   `TestFlashDecodeHD96_metalSelection` (off: no partial buffer, no lane; on: the block split and every layer past the floor);
   `TestAttnDigest_mc3RowsOtherGeometries/hd96-G1` (the rows form bit-identical to the single-row path through the batched step);
   and every existing attention kernel's digest unchanged against `origin/main` (`-attn-digest`, the same box).
2. **Fidelity:** `TestR2_decodeFidelityGate` with `GOINFER_METAL_GATE_MODEL=~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`,
   `GOINFER_METAL_GATE_CELL=P3M`, `GOINFER_METAL_GATE_QUANT=q4k`, `GOINFER_METAL_GATE_HD96=1`, K = 3,900, held-out set B. The
   reference is CUDA gate 2's `TestPrefillGateReference` `P3M` rows (CPU f32 from the same GGUF, 2026-11-22), copied from nobara
   with their sha256 recorded. Both arms prefill on the exact kernel (`decodeAttnFA` off) and the harness fails unless every layer
   takes the lane in the candidate arm. **Criteria, CUDA's, unchanged** ([`attn-decode-fa-fidelity-PREREGISTERED.md`](../attn-decode-fa-fidelity-PREREGISTERED.md)),
   graded from the harness's per-prompt lines: (a) hard flips ≤ exact + 2√exact; (b) mean agreement ≥ exact − 1.0 pt and lane ≥
   exact on at least half the prompts; (c) mean KL ≤ 1.10 × exact's. AMBIGUOUS → PARKED: (b) within its last 0.2 pt, or (c) in
   1.05–1.10×. The harness's own printed verdicts are reported beside, not deciding. VOID: a prompt failing the reference-identity
   check (seed-row KL > 1.0).
3. **Speed, served:** one serve binary for both arms through `scripts/bench_peer.py`, `goinfer` with `-flash-decode-hd96`,
   `goinfer_old` with `-flash-decode-hd96=false`, `BENCH_BACKENDS=metal`, Phi-3 mini at its default, depths 128 and 3,900,
   `BENCH_RUNS=2`, `BENCH_ABBA=1`, under the Mac's exclusive lock, fit-checked first. Graded as CUDA's `grade_cell.py`:
   **SHIP:** median lane ÷ exact ≥ 1.02 at 3,900 and ≥ 0.98 at 128. **REGRESSION:** < 0.98 at 3,900. **Between:** parked.

**Stopping rule:** fixed N at every gate; no early stop. A failed gate stops the sequence there.

**What each outcome means.** All three pass and SHIP: the lane takes hd 96 on Metal behind `--flash-decode-hd96`, the PR states the
bytes above, and the default goes to Francis. Gate 1 fails: fixed before gate 2 is spent. Gate 2 fails: hd 96 stays off the lane on
Metal and the kernel is not merged. Gate 3 parks or regresses: not merged; the record says which.

**Cost:** gate 1 about 10 min on the Mac. Gate 2 about 20 sequential q4k prefills of 3,900 positions plus 1,280 decode steps,
roughly 30–45 min under the exclusive lock. Gate 3 about 15 min. By day; nothing on the Mac past 21:00 PST.

## Addendum 2026-11-23 08:45 PST, before any gate reading: gate 3's arm check and scripts

- Metal has no per-op cost report, so CUDA's launch-count probe cannot run here. [`probe_lane_metal.py`](probe_lane_metal.py) takes
  its place: each arm serves the depth-2,048 prompt with `GOINFER_ATTNFA_DEBUG=1`, and the lane arm must log `attention_fa` dispatches
  at hd 96 while the exact arm logs none. The cell does not run otherwise.
- [`run_gate3_served.sh`](run_gate3_served.sh) is CUDA's served script on Metal (macOS settle under load 2.0, the Mac's idle gate);
  it grades with CUDA's [`grade_cell.py`](../fa-hd96-phi3-2026-11-22/grade_cell.py) unchanged.
- Unchanged: the gates, cells and bars above.
