# Metal audit program — 2026-10

**Status: proposed 2026-10-01, not started.** This is the execution plan for `docs/audit-metal-2026-09-30.md`: the
order the audit's §10 items run in, re-tagged for the run-budget rules and this Mac's limits. Each item keeps the
definition, band, probe and kill line its §10 row gives; this doc does not restate them, so read the row before
starting an item. Item IDs are the audit's. Decisions marked **O-** are the audit's Track 4 owner decisions, not the
Metal int8 task's O1.

## Where things stand

- **Fixed:** both Criticals, A-C01 ≡ F-C01 and F-C02 (`de1c7f17`, 2026-09-30).
- **Half settled:** T1.11's served K=3900 TTFT cell (LEVEL, 0.983) and the stale rows A-D01 and B-D01 named. Still
  owed: the 0.5B decode cells at 2048 and 3900 keys, and the short-K prefill rows.
- **In flight, not an audit finding:** native int8 on Metal (`docs/tasks/task-metal-int8-2026-10.md`). Its gates run
  tonight. Its slice 2 (int8 prefill GEMM) should reuse A-P01's tile selector, and its slice 3 (int8 in the MC3 step)
  meets E-P01, which is int4-only until then.
- Nothing else under `metal/` has changed since the audited snapshot.

## Constraints that set the order

- **Run budget.** By day: code, unit tests, and identity or bit-equality checks on tiny fixtures and the 0.5B. No
  timed or graded run on the Mac by day. The audit's `[day]` tags predate the 2026-09-28 rule, so its timing probes
  (T1.3, T1.4, T1.10's timing half, T1.12's microbenchmark) move to the night queue here.
- **One night queue** on the Mac: jobs of 3 h or less, none starting past the 06:30 deadline, the owner starts it.
  Tonight's queue already holds 3 h 45 min, so the first audit batch is tomorrow night at the earliest.
- **Memory.** 16 GB, with about 5 GB free by day. M26 is off-limits on this Mac without the owner's say-so (a
  monitored night run at most). That gates T1.8, T1.9, C-P01's probe and D-P01's measurement. The gpt-oss 20B
  checkpoint T1.13 needs is not on the Mac, and G20 is off-limits here too, so T1.13 runs on nobara.
- **Gates before speed.** F-G01 lands before any prefill-attention change, and D-G01 before any MoE prefill build.
- **CI does not run the batched-step gates** (E-G01): `.github/workflows/ci.yml` exports neither `GOINFER_METAL_MC3`
  nor `GOINFER_HEAVY_TESTS`. The default-run fixture in phase 1 is the fix.

## Phase 1 — guards, gates and text (by day, no numeric change)

In this order, one commit each:

1. **T0.2's remainder (F-G03).** The prefill run at 4097 keys the audit asked for exists since `de1c7f17`
   (`TestPrefill_exactAttentionDeclinesPast4096Keys`). What the ceiling comment in `metal/model.go` still claims without
   a test is that the decode attention kernel tiles correctly past 4096 keys. So: replace
   `TestMetalCtxCapWithinKernelBound` with a decode-attention run past 4096 keys against a float64 reference, on the
   tiny fixture `metal/prefill_criticals_test.go` already builds, and correct the comment.
2. **T0.3.** Enumerate registry architectures with head dim 128 and nH/nKV > 8, then add the G ≤ 8 guard (F-C03); a
   gpt-oss decline test in the expert-major branch (D-C01); a finite check on `PrefillLast` logits that declines to
   the sequential path (A-C02). The `Encoder.Dispatch` length check (C-N01) is in aikit; it goes in with the next aikit
   release.
3. **T0.4 (C-C01).** The unit test first: injected `hostRAMAvailable` and a 131072-window config, red today. Then the
   Metal ceiling and f16 KV pricing in `decoder/fitplan.go` and `decoder/fitguard.go`, and an auto-pin Metal clamps
   rather than refuses. A decoder core edit, so the parity refresh follows.
4. **Track 3 gates.** F-G01 first: the default-run float64-reference test for steel, to the audit's spec. Then A-G01,
   C-G01, D-G02, F-G04. Then the default-run identity fixture for E-G01 and F-G02: a generated hd=128 random-weight
   model with two slots, so the MC3 step, chunked prefill and spec-verify identity gates run in CI.
5. **T0.5.** The docs and comments pass.
6. **Identity checks by day:** T1.10's byte comparison (E-P01's premise: zero differing K/V and logits through the
   8-row step), T1.6 (resident set after load, 1 against 4 slots), T1.12's tile-waste count (no timing).
7. **The test-only hooks the night probes need:** T1.2's floor override, next to `attnFASplitOverride`.

## Phase 2 — night probes on the Mac

Each batch gets a `run-*.sh` with pinned binaries, an estimate made when it is written, and its §10 reading
committed before it runs.

- **Batch A** (estimate about 1.5 h, refined when written): T1.1, T1.2, T1.3, T1.4, T1.10's timing, T1.5.
- **Batch B** (estimate about 2.5–3 h): T1.7, T1.11's 0.5B cells via `bench_peer.py`, and T1.14, the same-session
  goinfer, mlx-lm and Ollama run that replaces the cross-session 0.82× and 0.75× (mlx-lm 0.31.3 is installed and
  `bench_peer.py` drives it).
- **Owner-gated:** T1.8 and T1.9, both on M26.
- **On nobara:** T1.13, the CPU-side requantisation-error script on the real gpt-oss checkpoint.

## Phase 3 — builds, in this order

Each item ships behind its own pre-registered band and kill line from §10. "Gated" means the pooled or set-B fidelity
gate runs at night before it ships.

1. **E-P01:** short prompts through the 8-row step. Bit-identical, so no fidelity gate. Needs T1.10 first.
2. **Decode attention, one campaign on one harness:** B-P03 (needs T1.2; re-bakes the snapshot golden that straddles
   the floor), B-P02, then B-P01. All gated.
3. **Small-M prefill:** A-P01 (bit-identical), then A-P02 (gated). Then Metal int8 slice 2 on the same tile selector.
4. **Decode GEMV residue:** B-P04 (bit-identical), B-P06, and B-P05 only after O4, with the MC3 down kernel moved in
   the same change.
5. **C-B01:** the on-device token chain (bit-identical), with C-P02 as its sibling.
6. **MoE and hybrids:** D-G01's gate first. Then D-B01 on the Qwen3.5-9B hybrid, if the fit guard admits it resident
   at night; then D-B02, D-P04, D-P03 and D-B04. D-P01 needs M26 and so the owner's OK.
7. **The batched step:** E-P03, E-P02, E-P05, E-P06, E-P08, E-P07. Metal int8 slice 3 joins here.
8. **Memory:** C-P01 (M26, owner-gated), E-P09 (after T1.6), F-D02.

Not planned until a probe says otherwise: the "not worth a probe" list at the end of §10, and B-P08 until T1.7.

## Owner decisions

None blocks phase 1 or 2. Each is needed only when its build comes up.

| ID | Question | Needed before | Recommendation |
|---|---|---|---|
| O1 (A-P04) | Embeddings: exact or batched | A-P04 | Pipeline first: bit-identical, 1.05–1.5×. Decide batched after measuring it |
| O2 (D-B05) | `delta_rule`'s summation order | D-B05 (after D-B01) | Decide once D-B01 shows the recurrence share is large |
| O3 (B-N01) | An int4 head on Metal | none in this plan | Defer: lossy, about 2.3 points top-1, and its quality re-eval is parked |
| O4 (B-P05) | Move decode's down projection off the per-word chain | phase 3 step 4 | Decide after B-P04 and B-P06, with the fidelity gate's result |
| O5 (D-B03) | Native MXFP4 for gpt-oss | after T1.13 | Decide on T1.13's error numbers |
| O6 (E-P04) | Device top-k/top-p/min-p | after phase 3 | Decide after the batched-step items |
| O7 (F-B01) | Keep P in f32 in steel | after F-G01 | A fidelity question; run the pooled gate first |
| — | M26 night runs (T1.8, T1.9, C-P01, D-P01) | phase 2's owner-gated row | A monitored run with the night queue's kill-watch, or not at all |

## Log

- 2026-10-01: proposed.
