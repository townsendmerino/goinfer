# Metal audit program — 2026-10

**Status: phase 1 in progress (started 2026-10-01 on the local branch `metal-audit`, not pushed until the program is done).** This is the execution plan for `docs/audit-metal-2026-09-30.md`: the
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
- **CI runs no Metal device test** (E-G01's premise, stronger than the audit could see): the metal job builds, vets and
  runs one device-free test, `TestParity_NaNCosineFailsTheGate`. So "default-run" in this plan means a plain
  `go test ./metal/` on the Mac, with no environment variable, which is what `cmd/gate gpu` and every local check run.
  The batched-step identity gates need `GOINFER_METAL_MC3` and a real checkpoint today; the fixture in phase 1 fixes
  that.

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
   model with two slots, so the MC3 step, chunked prefill and spec-verify identity gates run by default on the Mac.
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
- 2026-10-01: started on the local branch `metal-audit` (worktree `~/tmcode/goinfer-metal-audit`), owner decision: no
  push until the program is done. **Phase 1 item 1 (F-G03) done:** `TestAttentionKernelsPastTileBound` runs the shipped
  `attention`, `attention_f32` and `attention_i8` at 4096–12289 keys, with windows and a sink, at head dims 64 and 128:
  46 cases within 5e-5 of a float64 reference. Each multi-tile case plants a dominant key in its last tile and checks
  that a first-tile-only answer would miss the bar; a mutation that skips every other tile failed all 34 multi-tile
  cases. The constants-only test is removed, and the ceiling comments in `metal/model.go`, `metal/kernels.go` and
  `metal/backend.go` are corrected.
- 2026-10-01: the worktree had none of the main checkout's gitignored fixtures, so its suite skipped tests (125 s
  against 158 s); the 23 ignored fixture paths are now symlinked into it (excluded in `info/exclude`). **T0.3 done**
  except C-N01 (aikit, with its next release):
  - F-C03: `attnFAGroupOK` (G ≤ `attnFAMaxG`, tied to `ATTN_FA_MAXG` by `TestAttnFAMaxG_matchesKernel`) in the build
    loop, `canUseAttnFA` and `canUseAttnFAAt`. The registry cannot list the architectures it protects: the group size
    comes from each checkpoint (Llama 3.1 405B has 16).
  - D-C01: an explicit gpt-oss term in `prefillOK` and its own `PrefillPath` reason. Today gpt-oss is also kept out by
    two features the prefill map lacks (`attn-sink`, `out-bias`); the test admits both, so the guard alone holds it.
  - A-C02: `PrefillLast` declines on a NaN or ±Inf logit, and the decoder re-runs the prompt sequentially.
  - Mutations: removing each guard fails its test (4, 1 and 1 failures).
- 2026-10-01: **T0.4 (C-C01) done**, except that the fit guards are unchanged:
  - Before the fix (each change reverted in turn, against the new tests): an auto-pin of 20000 allocated 20000 resident
    positions, and one of 40000 was refused, which moves the whole forward to the CPU. `Plan("metal")` planned an
    unpinned 8192 at 8192 and accepted 32769. On a 131072-window config with memory for 40000 f32 positions,
    `guardGIWFit` pins 40000 on a Metal load, so the refusal half was reachable and the probe's kill line did not fire.
  - The fix: `resolveMetalCtxCap` treats an auto-pin as a ceiling (`min(pin, 4096)`; an explicit `-ctx` is honoured up
    to 32768 as before). `Plan("metal")` has Metal's ceiling, from new `decoder.MetalCtxDefault` and `MetalCtxCeiling`,
    which `metal/model.go` now takes its constants from. Plan already priced Metal's KV at f16 (`ResidentKVBytes`).
  - Not done: f16 pricing in the guards. They run before the backend decides whether it can host the model, and a
    model Metal declines runs on the CPU, whose KV grows to the request (R13), so f16 pricing there would admit a load
    the CPU fallback cannot hold. The cost: a machine too tight for the 2048-position floor at f32 is refused a load
    Metal could run at f16.
  - Also fixed: the floor refusal printed the whole window's need under "even at the 2048-token floor" (about 32 GB
    for a 128k-window 7B); it now prints the floor's.
  - Gates: the three changes' mutations fail their tests (2, 2 and 1 failures). The parity manifest stays fresh
    (neither file is hashed), so no refresh. Default metal suite: 208 pass, 51 skip, 125 s.
  - The citation re-point found 25 citations that the earlier rounds' `--update` had moved onto a neighbouring
    citation's line, with the lint green. All are fixed against their originals on main, checked by line content.
- 2026-10-01: **Phase 1 item 4 begun; F-G01 done.** `TestAttentionPrefillSteelMatchesFloat64` (default-run, no tag)
  runs the shipped `attention_prefill_steel` as `PrefillLast` launches it, over the audit's grid: M in {1, 7, 31, 33,
  100}, startPos in {0, 5, 16, 1000}, window in {0, 64}, nH/nKV in {12/2, 14/2, 32/8}, each with uniform and peaked
  (Q × 8) scores, against a float64 reference over the same f16 Q, K and V. 240 cases in about 3 s:
  - Worst row-head cosine 0.9999999, worst max abs 4.1e-4, no NaN (the output starts as f16 NaN, so an unwritten
    element fails). The bars: cosine 0.9999 (the audit's) and max abs 1e-3 (the audit's 0.05, tightened to 2.5× what
    the kernel measures).
  - Built-in check that it can fail: in the 210 cases whose first query row sees at most 64 keys, one more key per row
    moves the float64 answer by at least 0.039, past the bar.
  - Mutations of the kernel: causal limit +1 key fails 234 of 240 cases, −1 fails 239; no online-softmax rescale fails
    187 (94 peaked, 93 uniform); window start +1 fails 48, all window cases whose rows run past the window.
  - Also: the `metal/backend.go` comment that said the decoder always passes startPos 0 (it passes a reused prefix's
    length and each chunk's offset); a pointer in `prefill_startpos_test.go`, whose head-dim-16 fixture runs the fused
    kernel; a dated note under the 2026-09-12 audit's G-08 closure; and the F-G03 test now releases each case's
    buffers (about 300 MB by its buffer sizes, held until the process ended).
- 2026-10-01: **A-G01 done.** Its kernel half is F-G01's test (steel with window, ragged M and startPos, against
  float64 rather than the exact kernel). `TestPrefillFloor` (default-run) covers the floor: the parser ("" → 64, "0"
  → 0, "128" → 128; "-1", "x" and " 32" → 64), and the decision through `PrefillLast` on the tiny resident with the
  knob set per model: 63 positions decline and 64 admit; cached positions count, so 56 + 7 declines and 56 + 8 and
  1000 + 8 admit; "0" admits 8, "128" declines 100 and admits 128; `PrefillPath` names the floor. Mutations: dropping
  `startPos` from the predicate fails 2 cases, `<` for `<=` 3, a parser that accepts negatives 1.
- 2026-10-01: **C-G01 done.** `TestMetal_CloseFreesMemory` (heavy: `GOINFER_HEAVY_TESTS=1`, the 0.5B GGUF in
  `~/models`, about 10 s) now also reads `CurrentAllocatedSize` through its own probe Device and fails if it ends more
  than 16 MiB above the post-warm-up value. Measured: 655,360 bytes before and after every cycle; 401 MB with a
  resident built (the test's comment said about 0.7 GB; corrected). With `Close` made to skip `ReleaseAll`, the new
  check fails at +1,602,355,200 bytes, exactly 4 × one resident. The RSS check failed too, but it saw +631 MB and
  +1022 MB of the 1.6 GB in two runs.
- 2026-10-01: **D-G02 done.** `TestQGateKernels_cpuParity` (tag `goinfer_testhooks`, beside the DeltaNet chain gate)
  runs `delta_qsplit` then `delta_attn_gate` as `encodeLayer` launches them, at nH × hd of 16 × 256, 4 × 128 and 3 × 40,
  with gate values out to ±100, against the CPU's own split and gate. Those were inline in both CPU paths; they are now
  `splitQGate` and `qGateContext` in `decoder/forward_qwen35.go`, called by the per-token and the batched path and
  exported as test hooks. The refactor moves no number: the forward goldens of all four families that own the file
  (OlmoHybrid, Qwen35, Qwen3_5, Qwen3Next) ran green before the `deps_hash` refresh (39 passed, 24 skipped, 0 failed),
  and `TestPromptHidden_batchedMatchesSequential` still bounds the batched path at relative L2 about 1e-7.
  - Measured: the split matches exactly and writes nothing past its length; the gate factor is within 2.1e-7 of the
    CPU's (bar 1e-6).
  - `TestQGateKernels_mutations`: swapped halves (all 9456 elements differ), two blocks instead of per-head
    interleaving (8608), no sigmoid (gate error 100) and the sigmoid's sign flipped (1) each fail.
  - The chain gate's comment said both kernels and the DeltaNet reset had no wiring; it now says where each is checked
    (the reset by the replay in `qwen35_resident_parity_test.go`).
