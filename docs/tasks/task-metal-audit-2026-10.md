# Metal audit program — 2026-10

**Status: phase 1 done; phase 2 Batch A and the M26 job graded 2026-10-02; C-P01 built, its A/B queued (started 2026-10-01 on the local branch `metal-audit`, not pushed until the program is done).** This is the execution plan for `docs/audit-metal-2026-09-30.md`: the
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

### Batch A: pre-registration (written 2026-10-01, before any graded run)

The probes are `metal/audit_batch_a_test.go`, run by `docs/measurements/metal-audit-2026-10/run-batch-a.sh` from a test
binary built at a pinned commit. T1.5 is not in it: its 1.5B re-run and its default-run fixture were done by day on
2026-10-01 (identity, not timing). Smoke runs at one repetition checked the harness only; none of their numbers is a
result. They ran before this table was written: its thresholds are the audit's, set on 2026-09-30, and what this table
adds (the parked bands, T1.2's two-model reading, T1.4's control) is structure, not chosen from the smoke numbers. Models are int4 from `~/models`: the 1.5B and the 0.5B are qwen2.5-coder-1.5b/0.5b-instruct q4_k_m `.gguf`; the 7B
is qwen2.5-7b-instruct, loaded from its `.int4.metal.giw` sidecar as R18b did (file-backed, so lighter on this Mac's
memory than the `.gguf`). Estimate: about 15 minutes of work and 7 model loads; queued at 20.

| Probe | Models, sample | Reading | Rule |
|---|---|---|---|
| T1.1 (A-P03) | 1.5B; one `PrefillLast` pass at startPos 64, 2048, 8000 × C 16, 32, 48, 64; 7 reps, cells interleaved in a rotating order | paired delta, wall time, (startPos 2048) − (startPos 64) at C = 32, median of the 7 pairs | ≤ 10 ms: A-P03 closes. ≥ 25 ms: build the BQ=16 steel variant (Phase 3). Between: parked. |
| T1.2 (B-P03) | 1.5B and 7B; 20 decode tokens per arm from 256, 384, 512, 768, 1024, 1280, 1535 keys; 5 reps, arms alternated; GPU time per token | legacy ÷ block kernel, per depth and model | Below 1.05 at 768 keys on both models: the floor stays (killed). Otherwise the candidate floor is the smallest depth with ≥ 1.10 on one model and ≥ 1.00 on the other, and it goes to the fidelity gate before anything ships. No such depth: parked. |
| T1.3 (C-B01) | 1.5B and 0.5B; 200 greedy tokens through the pipelined `Forward` | GPU-idle gap GPUStart(t+1) − GPUEnd(t), median | Under 0.15 ms on both models: C-B01 is killed. At or above it on both: C-B01 stands, and the median replaces its projected band. Mixed: reported per model, parked. |
| T1.4 (B-P04) | standalone `gemv_w4a8_sa_rows4` at ~15 MB of weights, K 1536, 2048, 3072; 7 reps, arms interleaved, weight copies rotated past the cache | GPU time per byte, K=1536 ÷ K=2048, with K=3072 ÷ K=2048 as the no-tail control | ≥ 1.10 with the control within ±5%: the idle tail is confirmed and B-P04's bit-identical build goes ahead. Under 1.05: B-P04's premise is killed. Otherwise parked. |
| T1.10 timing (E-P01) | 1.5B and 7B; a fresh K = 8, 16, 32, 64 prompt three ways (sequential loop, the batched step in 8-row pieces, the batched pass with its floor off); 7 reps, arms interleaved | step ÷ sequential at K = 32, wall time; step ÷ pass at K ≥ 16, reported | Above 0.6 on either model: E-P01 is killed. At or under 0.6 on both: E-P01's route goes ahead (Phase 3 item 1), limited to the K where step is faster than the pass. |

Two deviations from the audit's wording, both decided before the run. T1.3's gap uses GPUStartTime and GPUEndTime: the
audit wrote kernStart/kernEnd, but Metal's kernelStart/EndTime are the CPU's scheduling window for a command buffer, not
its execution (the probe prints that gap too, for reference). T1.2's two-model rule is spelled out above; the audit's
"on either model" is read as "the floor stays unless at least one model gains 1.05×".

### The M26 job: pre-registration (written 2026-10-01; owner-approved the same evening, to run last in the queue)

T1.8 and T1.9 on the Gemma 4 26B, `~/models/gemma4-26b-int4-v14st.metal.giw` (the only M26 on local disk: a v14
metal-target bundle, which carries f16 scales in its mapping as a v15 one does; the pager builds its scale cache the
same way on both). Probe: `TestAuditM26_pagedProbe` (`metal/audit_m26_test.go`), one configuration per process,
loaded through the production Metal path with every guard on, 40 timed greedy tokens after a warm one. Each process
runs under `scripts/swap_killwatch.sh` (1 s polls; kill on two consecutive ticks of more than 80 MB of swap growth, or
1 GB over the run's baseline). Nothing forks from the probe: `vmmap` and `footprint` are taken from outside while it
holds at token 32. Run script: `docs/measurements/metal-audit-2026-10/run-m26.sh`. Estimate about 5 minutes; queued at
15, priority 90 so it runs after everything else.

T1.12 is not in this job, and cannot run on M26: Metal's batched prefill excludes Gemma 4's MoE (`prefillOK` requires
`!HasGemma4MoEResident()`), so M26's prompt runs token by token and never reaches the expert-major code D-B02 is
about. M35 is excluded too (DeltaNet, and paged). T1.12 needs a fully resident generic MoE.

| Run | Configuration | Reading | Rule |
|---|---|---|---|
| T1.8 (C-P01) | 8 slots per layer, held at token 32 | the pager's f16 scale cache (computed from the pool: experts × per-expert scale words × 2 bytes) ÷ the Go heap in use at token 32; `vmmap -summary` and `footprint` for the region breakdown | ≥ 0.50: C-P01 is confirmed, and serving the scales from the mapping (as the nibbles already are) goes to Phase 3. Under 0.20: C-P01's premise is killed, and the vmmap table names what the heap is. Between: parked. |
| T1.9a (D-P02, M-11) | 8 slots, `GOINFER_MOE_PROF_SPLIT=1` | per-command-buffer round trip, (submit+wait − GPU-busy) per token ÷ the token's time, with the commit/wait split | Under 0.05: M-11's shared-event design is closed for M26, as D-P02 projects. ≥ 0.10: reopened, and C-B03's fence microbenchmark (aikit) is the next step. Between: parked. |
| T1.9b (R11(c)) | 64 slots, last, and only if neither earlier run tripped the kill-watch | whether it completes without the kill-watch firing, and its peak swap growth; tok/s against the 8-slot run, reported only (one run each, separate processes) | Completes with swap growth under 256 MB: R11(c)'s "lower default" conclusion is withdrawn (its spirals predate the 2026-09-24 fork fix). The kill-watch fires: R11(c) stands at 64 slots. |

### Batch A: results (night of 2026-10-01, graded 2026-10-02)

Ran 01:30–01:33 PDT under the night runner, binary `metal-cfce51e1.test` (sha256 `ed34f2636802abc5`), load 1.44 at
the start. Logs and the collected RESULT lines: `docs/measurements/metal-audit-2026-10/batch-a-2026-10-01/`.

| Probe | Measured | Verdict |
|---|---|---|
| T1.1 (A-P03) | paired delta at C = 32, median of 7: **15.5 ms** (pairs 15.1–15.7) | **Parked** (between 10 and 25 ms) |
| T1.2 (B-P03) | legacy ÷ block at 768 keys: 1.5B **1.091**, 7B **1.028**. At 1024: 1.167 and 1.056; at 1535: 1.320 and 1.121. Below 768, both models are 0.97–1.03 | Not killed (the 1.5B clears 1.05). **Candidate floor 1024**, the first depth with ≥ 1.10 on one model (1.5B) and ≥ 1.00 on the other; 768 is 1.091 / 1.028. It goes to the fidelity gate before anything ships |
| T1.3 (C-B01) | GPU-idle gap, median over 199: 0.5B **0.583 ms** (p10 0.572, p90 0.650), 1.5B **0.663 ms** (p10 0.596, p90 0.683) | **C-B01 stands** on both models; these medians replace its projected band |
| T1.4 (B-P04) | time per byte K=1536 ÷ K=2048 = **1.163**; control K=3072 ÷ K=2048 = **1.033** | **Confirmed** (≥ 1.10, control within ±5%): B-P04's bit-identical build goes ahead |
| T1.10 timing (E-P01) | step ÷ sequential at K = 32: 1.5B **0.216**, 7B **0.254**. Step ÷ pass at K = 16, 32, 64: 0.408 / 0.802 / 1.565 (1.5B), 0.398 / 0.792 / 1.564 (7B) | **E-P01 proceeds**, limited to K ≤ 32, where the step beats the pass on both models |

### The M26 job: results (night of 2026-10-01, graded 2026-10-02)

Ran 01:33–01:35 PDT, binary `metal-tagged-f56b40ec.test` (sha256 `e82447acb289f9d8`). Swap was 494 MB before the
first process and after the last, and no kill-watch fired. Logs, `vmmap` and `footprint`:
`docs/measurements/metal-audit-2026-10/m26-2026-10-01/`.

| Run | Measured | Verdict |
|---|---|---|
| T1.8 (C-P01) | scale cache **1361.2 MB**, Go heap in use at token 32 **3004.6 MB**: **0.453**. `footprint`: 4.26 GB, of it 3.19 GB untagged (the Go heap) and 1.05 GB IOAccelerator; 2.18 GB of mapped file, clean | **Parked** by the rule (between 0.20 and 0.50). Promoted by the owner on 2026-10-02 on its absolute size; see below |
| T1.9a (D-P02, M-11) | round trip (submit+wait − GPU-busy) **17.7 ms of 99.8 ms per token = 0.177**; commit 0.5 ms, waitUntilCompleted 58.2 ms, GPU idle inside the wait 17.1 ms per token. T1.8's run, without the split, read 0.244 | **M-11 reopened** (≥ 0.10). C-B03's fence microbenchmark (aikit) is next |
| T1.9b (R11(c)) | did not run on Metal. The resident guard declined the 64-slot build: 7.72 GB against a 6.65 GB budget, set by live-available memory 30 s after the previous M26 process exited. The probe failed on the CPU fallback, as it should. No swap growth | **Not graded.** Re-queued alone, first in a night, on the same binary (owner, 2026-10-02) |

For the record, tok/s from 40 tokens on 8 slots: 8.73 (T1.8's process, which held at token 32) and 10.02 (T1.9a's).
One run each, so they say nothing about noise beyond a 15% spread between two processes of the same configuration.

### C-P01: built, pending its A/B (owner, 2026-10-02)

The owner promoted C-P01 on the absolute saving, not on T1.8's ratio, and asked for an A/B to make sure the change
does not slow decode much. Both pagers (`buildGemma4MoELayer` and the generic `buildMoELayer`) now stage each expert's
f16 scales from its own WeightMat (`Int4ScalesF16`), which a v14 metal or v15 `.giw` aliases from the mapping. The
heap cache is gone, with `int4DirectBytes`, its only builder. Why the outputs cannot move:

- The cache was the WeightMat's binary16 scales widened to f32 and narrowed back. `TestF16ScaleRoundTrip_exhaustive`
  (default-run) checks that round trip returns the same bits for all 63,490 non-NaN binary16 values, so the cache and
  the mapping agree on every checkpoint, M26 included.
- `TestGemma4PagingPread_matchesNonPaged` (new, tagged): the tiny Gemma 4 MoE transcoded to a metal `.giw`, paged at 2
  and 3 slots by pread and by byte-copy, exact logits against non-paged. It asserts that all 16 expert scale arrays
  lie in the mapping and that the slices the pager stages are those bytes; with the stage function returning a copy,
  it fails ("4 of 8 staged scale slices are the mapping's bytes").
- The existing paging gates pass: `TestGemma4Paging_bitExact`, `TestMoEPaging_matchesNonPaged`,
  `TestMoEPagingPread_matchesByteCopy` (the generic pread path, which now uploads gate's and up's scales to the two
  halves of the slot's range), and the expert-pool tests.

The fit and resident guards never priced the cache (paged experts are counted as streaming, with no host copy), so the
guard's number does not change; what changes is that the number is now true.

**The A/B, pre-registered 2026-10-02 before any timed run.** Run script
`docs/measurements/metal-audit-2026-10/run-cp01-ab.sh`, on the Mac night queue after T1.9b.

- Arms: **old** = `metal-tagged-f56b40ec.test`, the binary T1.8 ran (the cache); **new** = the same probe built at
  the C-P01 commit, `2838b7de` (`metal-tagged-2838b7de.test`). Both run `TestAuditM26_pagedProbe` on M26 at 8 slots, 128 timed tokens after a warm one, one
  process each, under the kill-watch. Both hold at token 32 (the script removes the hold file on sight, so the pause
  is symmetric and outside the timed window) and print the Go heap in use there.
- Order: one discarded warm-up process (old), then 8 pairs alternated ABBA (old, new, new, old, ...), 15 s between
  processes.
- Reading: per pair, new tok/s ÷ old tok/s; the median of the 8 and the count of pairs below 1. The median heap in
  use at token 32 per arm.
- Rule:
  - **Ship** (it stays on the branch, default, with no knob): median ratio ≥ 0.97, and the heap drops by ≥ 1.0 GB.
  - **To the owner with the numbers:** 0.93 ≤ median < 0.97.
  - **Not shipped as is:** median < 0.93. The next build preads the scale span with the nibbles and re-runs this A/B.
  - **Not shipped, whatever the speed:** the heap drops by less than 1.0 GB, because then the saving did not happen.
  - The kill-watch fires: the A/B stops there, and that is recorded.
- Estimate: 17 processes at about 45 s each (about 11 s to build, 129 tokens at about 0.1 s, 15 s settle): about 13
  minutes; queued at 20.

## Phase 3 — builds, in this order

Each item ships behind its own pre-registered band and kill line from §10. "Gated" means the pooled or set-B fidelity
gate runs at night before it ships.

1. **E-P01:** short prompts through the 8-row step. Bit-identical, so no fidelity gate. T1.10 cleared it for K ≤ 32.
2. **Decode attention, one campaign on one harness:** B-P03 (T1.2's candidate floor is 1024; re-bakes the snapshot golden that
   straddles the floor), B-P02, then B-P01. All gated.
3. **Small-M prefill:** A-P01 (bit-identical), then A-P02 (gated). Then Metal int8 slice 2 on the same tile selector.
4. **Decode GEMV residue:** B-P04 (bit-identical; T1.4 confirmed its idle tail), B-P06, and B-P05 only after O4, with the MC3 down kernel moved in
   the same change.
5. **C-B01:** the on-device token chain (bit-identical), with C-P02 as its sibling. T1.3: a 0.58–0.66 ms GPU-idle gap
   per token.
6. **MoE and hybrids:** D-G01's gate first. Then D-B01 on the Qwen3.5-9B hybrid, if the fit guard admits it resident
   at night; then D-B02, D-P04, D-P03 and D-B04. D-P01 needs M26 and so the owner's OK.
7. **The batched step:** E-P03, E-P02, E-P05, E-P06, E-P08, E-P07. Metal int8 slice 3 joins here.
8. **Memory:** C-P01 (built 2026-10-02 on the owner's word, pending its A/B, above), E-P09 (after T1.6), F-D02.

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
- 2026-10-01: **F-G04 answered** (no code). `TestBatchedVerifyKernelParity` was deleted with the kernels it measured in
  `b51b846d` (2026-09-24): the batched-verify experiment's `gemv_w4a8_bvk_*` kernels were a NO-GO and never in
  `allKernels` or a dispatch path. The shipped verify path (`--spec ngram`, ForwardN on the MC3 step kernels) is gated by
  `TestSpecVerify_forwardNMatchesForward` and `TestMC3Verify_sameSlotRowsBitIdentical`, which need `GOINFER_METAL_MC3=1`
  and a real checkpoint. `cmd/gate` never sets that variable, so `gate gpu` skips them: F-G02's gap, and the reason for
  the two-slot fixture next.
- 2026-10-01: **E-G01 and F-G02 done: the identity fixture.** `metal/mc3_fixture_test.go` writes a qwen2 with
  Qwen2.5-1.5B's attention geometry (12 query heads of 128 over 2 KV heads, so attention_fa's G=6 block kernel and the
  steel prefill kernel engage) and small everything else (hidden 256, 2 layers, vocabulary 20480 so the tests' token
  ids stay valid), loaded at int4; `TestMC3Fixture_isTheBatchedPath` pins that it builds the batched step, steel and
  the block kernel. With `GOINFER_METAL_MC3` unset, `mc3Resident` and the new `mc3PrefillResident` return it, so seven
  identity checks run by default (16 s together); the timings call `mc3RealResident` and still need the variable.
  `TestMC5_prefillChunkInvariance` and `TestSpecVerify_forwardNMatchesForward` left `goinfer_testhooks`: their only use
  of it was `ResidentForwardForTest`, and the backend wraps a resident exactly as the struct literal does.
  - Default suite: 209 pass / 51 skip → 223 pass / 41 skip, 142 s.
  - Mutations on the fixture: a one-ulp change to `mc3_bt`'s group scale fails both step checks (518,252 logits); the
    steel causal limit +1 fails chunk invariance; `ForwardBatch` attending one key short fails spec verify. Two
    attempts showed nothing and are not evidence either way: re-associating `mc3_bt`'s partial sums (no bits moved,
    so presumably the same machine code), and any batched-kernel change for spec verify, whose `ForwardN` runs
    `ForwardBatch`'s per-token kernels, not the step kernels.
  - Found on the way: chunk invariance at sizes aligned to the 32-row tiles cannot see a tile-edge bug. With the steel
    causal limit moved one key, C = 64, 128, 256 and 384 all still matched bit for bit, and only the control failed.
    The test now adds 100 and 77 (off the tiles, as a reuse turn's start is) and 512 (serve's default); on the shipped
    kernel all are bit-identical, on the fixture and on the 1.5B.
  - The 1.5B re-run on the steel build: 0 of 14,336,000 K/V elements and 0 of 151,936 logits differ at all seven
    sizes; appended to `docs/measurements/chunked-prefill-2026-09-27.md` with its log. `TestMC3Step_bitIdentical` and
    `TestSpecVerify_forwardNMatchesForward` also pass on the 1.5B through the restructured helpers.
- 2026-10-01: the branch's tiny-fixture prefill tests (`TestPrefill_declinesNonFiniteLogits`, `TestPrefillFloor`,
  `TestPrefill_gptossFailsClosed`) now load int4 rather than int8int8. Once `nativeInt8` is on (the int8 task's flip,
  on main), an int8int8 load of the dense tiny fixture runs native int8 (every projection's columns divide by 4), and
  batched prefill declines it before any of these guards is reached. A load with no Backend keeps canonical int4
  (`wantsCanonicalInt4`), so int4 needs no re-quantization; the T0.3 comment that said otherwise is corrected. Main's
  own `prefill_criticals_test.go` has the same dependency and is the flip's to fix. Each guard's mutation still fails
  its test under int4 (1, 2 and 1 failures).
- 2026-10-01: **T0.5, the code-comment half.**
  - A-D02: `PrefillLast`'s cancellation, default-on and floor comments, and the floor values in `prefill_ttft_test.go`
    and `prefill_gate_test.go` (the floor is 64 since R3).
  - B-D02: the attention block is 8 dispatches at attention_fa depths (the combine).
  - C-D01: item 1, the `.giw` guard comment now says the paged scale cache and the rest of the heap are anonymous and
    unpriced; item 3, `attnFACoreCount = 14` is right (this Mac's M1 Pro has 14 GPU cores by `system_profiler`; the
    audit read 16), and its comment now says it sets only the legacy kernel's split (E-D01); item 5, `ForwardArgmax` is
    called by tests and gates only.
  - D-D01 7–10: MoE prefill is expert-major by default, with the row loop behind `GOINFER_MOE_EXPERT_MAJOR=0`
    (`metal/model.go`, `metal/backend.go`, the MoE parity test); `FeatAttnSink` is declared and wired; the DeltaNet
    unroll comment cites `4090dc45`'s measurement instead of calling itself a candidate.
  - E-D01: `--spec`'s help names Metal as well as the CPU; `EmbedInt4`'s comment gives both defaults (off in a zero
    `Options`, on in the CLIs since 2026-09-28).
  - F-D02: `alias.go`'s header says what the S6 record says (the footprint gate read NOT MET, the memory-hog arm ran
    on the 7B only, M26 decode was not resolvable); the banner no longer says a v15 bundle carries no f16 scales, and
    its test follows. F-D02's code options (bind v15 scales directly, a v15 non-metal fixture assertion) are not done.
  - F-D03: `HiddenLast` matches decode's kernels, not the CPU bit for bit; the comment gives the measured cosines.
  - Citations: 13 pointed into rewritten comments or had already drifted on main (A-D02's table row cited the
    `PrefillLast` signature, a `ctx.Err` check and a brace; the C-09 `recordExecErr` site was not that line). They are
    fixed by content, and five index rows were re-keyed to the new text.
  - The parity manifest went stale for 38 families (`decoder/model.go`'s comments); 39 goldens green, `deps_hash`
    refreshed.
- 2026-10-01: **T0.5 done** (the docs half; the comment half is above), except F-D02's code options.
  - `docs/benchmarks.md`: B-D01 (R18b's "AHEAD in every cell" is all six 1.5B and 7B cells; the 0.5B's latest
    reading at depth is still 0.75× / 0.58×), F-D01 (callouts on the §B3 banner and its 2026-08 verdict, and N-40: the
    LM head is int8), D-D01 4 and 5 (M35 and M26 have run here since "off-limits on any path"; gpt-oss is resident on
    Metal), A-D01 (the floor tables predate R16 and R19).
  - `docs/tasks/red-october.md`: D-D01 1–3 (the 3–4 tok/s ceiling is a 35B count; R11(c)'s "lower default" awaits a
    pager re-run after the 2026-09-24 fork fix; Metal's expert-major prefill is not the lane R11(b) found
    bit-identical).
  - `docs/gpu-residency-coverage.md`: F-D04 (what the two tests cover; no Metal test runs a Qwen-VL model).
  - `docs/audit-metal-2026-09-12.md`: D-D01 6 (`NewBufferNoCopy` superseded by S6) and C-D01 4 (N-20's f32 premise).
  - `docs/tasks/task-never-swap-2026-09.md`: C-D01 2, under "What this doc does not claim".
  - Two findings did not hold: A-D03's record is in the tree (`fdd990ba`; the audit's snapshot lacked it), and C-D01
    item 3's 16 cores (`system_profiler`: 14, as the constant says).
- 2026-10-01: **Phase 1 item 7: T1.2's hook.** `attnFAFloorOverride` (test-only, zero in production) moves the key
  count where attention_fa takes over, read through `attnFAFloor()` by `attnPlanFor`, `canUseAttnFA` and the batched
  step's `canUseAttnFAAt`. The audit named only the first two; the third is needed because a batched step that planned
  from the constant would no longer match the single-token step. `TestAttnFAFloorOverride_stepsPlanAlike` (default-run,
  on the fixture) checks the three plans move together and that a step straddling a floor of 256 stays bit-identical
  (0 logits differ over 4 sequences × 12 steps); with the batched check left on the constant, 149,563 logits differ.
  Production's path is unchanged (the override is zero). The legacy-against-blk timing arms are night work (Phase 2).
  Three citations of `gemvRowsFor`'s guard had pointed at `attnPlanFor` since before the branch; re-pointed.
- 2026-10-01: **Phase 1 item 6.**
  - T1.10's byte comparison: `TestMC3Step_promptInRowsBitIdentical` (default-run on the fixture; with
    `GOINFER_METAL_MC3=1`, the 1.5B) feeds K = 8/16/32/64 tokens to the batched step in 8-row pieces from position 0;
    every K/V element and every row's logits match the sequential loop on both. A one-ulp change to `mc3_bt`'s scale
    fails it. The timing half is night work.
  - T1.6 on the 1.5B (`TestMC3_kvSlotFootprintProbe`, a probe, each arm in its own process: in one process the fit
    guard refused the third load, correctly, because the first two had not handed back their memory). Untouched KV
    pages do not count at allocation (4 slots and 1 slot: footprint 2809 vs 2805/2811 MB); after one token every slot
    does (IOAccelerator +449 MB at 4 slots, +114 at 1). The 3 extra default slots cost ~335 MB on the 1.5B from the
    first token, ~705 MB on the 7B by the same bytes. Recorded under E-P09; lowering the default is the owner's call.
    Log: `docs/measurements/metal-audit-2026-10/t16-kv-slot-footprint-2026-10-01.log`.
  - T1.12 not done: it needs a real MoE's routing, and the only MoEs on this Mac's local disk are M26 and M35
    (owner-gated). It joins the M26 night items.
- 2026-10-01: **Phase 2, Batch A written and pre-registered** (the table under Phase 2): `metal/audit_batch_a_test.go`
  holds the five probes, gated by `GOINFER_METAL_AUDIT_A=1`. A smoke run at one repetition (exploratory, numbers not
  results) found two harness errors, both fixed before the pre-registration: T1.2 had read the CPU's scheduling window
  (`kernStart/kernEnd`, ~0.1 ms) as the token's GPU time, and T1.1 printed an empty GPU column (`PrefillLast` records no
  GPU timestamps; wall time only now).
- 2026-10-01: **the M26 job, owner-approved to run last.** T1.8 and T1.9 written and pre-registered (the table under
  Phase 2). `TestAuditM26_pagedProbe` was smoke-run on the tiny gemma4 MoE at 3 slots (paged build, the hold handshake,
  the profile split); no M26 by day. T1.12 dropped out: M26's prefill never reaches the expert-major path it measures.
- 2026-10-02: **Batch A and the M26 job graded** (results under Phase 2): T1.1 parked; T1.2 candidate floor 1024; T1.3
  C-B01 stands; T1.4 confirmed; T1.10 E-P01 proceeds for K ≤ 32; T1.8 parked; T1.9a M-11 reopened; T1.9b ungraded (the
  resident guard declined 64 slots) and re-queued alone, first in a night (owner).
- 2026-10-02: **C-P01 built** (`2838b7de`) on the owner's word, with its A/B pre-registered and queued after T1.9b.
  The A/B's run script was dry-run end to end on the tiny Gemma 4 MoE (3 slots, 100 tokens): every process completed
  and the summary graded, with tok/s and heap numbers that say nothing (the fixture loads from safetensors, so there is
  no mapping to save).
