# PRE-REGISTERED — multi-row flash-decode lane for speculative verify (R6 spec-decode story, option B)

**Written 2026-09-21 BEFORE any multi-row kernel exists. Not edited after any result was seen.** The record goes in
`attn-decode-fa-verify-2026-*.md`; this file stays as written. Context: `spec-decode-lane-2026-09-21.md` (why), the fidelity record
`attn-decode-fa-fidelity-2026-09-20.md` (what the single-row lane already passed).

## The question

Option A (shipped) makes speculative generations run the EXACT attention tree for decode and verify, so speculation and the lane coexist but
speculation gets no lane speed. In a speculative round on the 1.5B at ~2100 keys, verify attention (`attn_batched`, one block per head x row, each
re-reading K/V) was **47.1%** of GPU time (399 us/layer; ncu, one window). Can a multi-row lane kernel, whose per-row results are
BIT-IDENTICAL to the single-row (M=1) lane, replace it — so verify rows equal what plain lane decode would compute, and speculative output equals
plain LANE greedy — and how much does that buy?

## Why bit-identity is the chosen route, and what it buys

If every verify row is bit-identical to the M=1 lane at the same position, then (1) speculative output equals plain lane greedy exactly (lossless on the
lane's tree, the same guarantee speculation has on the exact tree today); (2) **no new fidelity gate is needed for the lane's numerics**: the rows'
numerics are the M=1 lane's, already gated on held-out set B (which is spent; a re-run is a re-roll). The M=1 lane kernel is NOT modified.

## Design (fixed here; deviations are recorded, not silent)

- New kernels `fa_partial_rows_{64,128,256}` beside `fa_partial_*` in `cuda/decode_fa.cu`; the M=1 kernels and `fa_combine` maths are untouched
  (`fa_combine` gains a row index). A CTA handles (kv head, split, group of R consecutive verify rows), R fixed at compile time; the 8 warps are the
  same 8 "units" as M=1 and each keeps, per row, the SAME (m, l, acc) recurrence over the SAME blocks in the SAME order as the M=1 warp with that unit
  index — so per-row partials are equal bit for bit. K and V rows of a block are loaded once and used for all R rows.
- **Partition:** each row keeps its own M=1 partition, `per = ceil(span/S)`, `lo = winStart + sp*per`, with `span = nKeys_row - winStart_row`. Rows of a
  batch are grouped into maximal runs of consecutive rows with equal (`per`, `winStart`); a run shares a launch. Within a run, rows differ only in how far
  the last chunk extends (causal masking): keys beyond a row's `hi` are masked (score -1e30, p = 0), which is exactly neutral in the online softmax
  (alpha = exp(0) = 1, l*1+0 = l, acc*1 + 0*v = acc for finite v), so processing them equals not visiting them.
- **Per-row eligibility:** a row uses the lane iff the M=1 rule says so at its own position: lane enabled, no exact-attention scope held, layer eligible
  (hd in {64,128,256}, GQA <= 8, no sink), and its attended span >= `GOINFER_CUDA_FLASH_DECODE_MIN_KEYS`. A batch that straddles the floor is split into an
  exact run (`attn_batched` over that row range) and a lane run, so every row does exactly what M=1 decode would.
- **Routing:** only all-rows verify tails (`tailAllLogits`, `tailAllArgmax`) take the lane; prefill (`tailLastLogits`) is unchanged. Batches above the
  kernel's row cap are processed in chunks of at most that many rows (attention only). When the multi-row kernels are unavailable, or
  `GOINFER_CUDA_FLASH_DECODE_VERIFY=0`, speculative generations fall back to option A's exact-attention scope.

## Gates (all must hold before this is offered)

- **G1 cross-M bit-identity.** For every row of a batch, the multi-row output equals the M=1 lane output at that position, `math.Float32bits` equal, on (a)
  random K/V at hd 64/128/256 with the GQA groups of the 0.5B (7), 1.5B (6), gemma3-1b (4) and D7 (7); S in {2,4,8,16}; M in {1..16}; batches placed to straddle
  a `per` change and (with a sliding window) a `winStart` change and an attended-span floor; and (b) the real K/V of D7 at 8000 and the 1.5B at 3900. **Any
  differing bit is a defect.** The test is mutation-checked (a changed rounding / block order must fail it).
- **G2 speculative losslessness on the lane's tree.** With the lane on and NO exact-attention scope, n-gram, two-model and block speculative output equals plain
  LANE greedy token for token on the real 1.5B copy-heavy prompt and on a non-copy prompt, and the lane is actually launched inside verify rounds (launch
  count, not a flag). Mutation-checked as above.
- **G3 M=1 unchanged.** Plain lane decode is bit-identical before and after this change (the M=1 kernel is not edited; checked by the existing
  `TestFlashDecodeVsF64` and a golden of 24 decode steps' logits).
- **G4 verify path parity elsewhere.** The exact path's spec-decode losslessness gates still pass with the lane OFF.

## Speed decision rule (served, greedy, `serve --spec ngram`, fresh serve per arm, alternating order, idle box, paired)

Arms per cell: exact + spec (today's option A behaviour), lane + spec with B, and (context) plain exact and plain lane. Cells: 1.5B on a copy-heavy code
prompt at ~2100, ~3900 and ~7500 tokens; D7 on the same prompt at ~3900 and ~7500. Ratio R = tok/s(lane+spec with B) / tok/s(exact+spec).
- **Ships (offered opt-in, spec+lane speedup claimed):** R >= 1.15 on the 1.5B at ~3900 AND on D7 at ~7500, and no cell below 0.98.
- **Parked:** R in [1.05, 1.15) on either of those two cells (ambiguous band: no claim, recorded).
- **Killed:** R < 1.05 on either, or any cell < 0.98 (a regression). A kill is recorded like a win and the code stays out of the default path.
- An A/A floor (both arms exact + spec) is run in the same procedure; effects inside it are not read.
- **Also recorded, not gating:** acceptance stats (accepted/round) per arm, so a speed change is not misread as an acceptance change; the ncu share of
  attention in a verify round before/after.

## What this does not claim

Not a fidelity result (G1 makes the numerics those of the gated M=1 lane); not a claim for non-copy traffic (n-gram acceptance is prompt-dependent, so the
prompts and their acceptance are stated); not the LM head or weight GEMVs (separate levers); not default-on (the lane stays opt-in; this removes only its
cost to speculation).
