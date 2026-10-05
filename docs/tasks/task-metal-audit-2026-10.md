# Metal audit program — 2026-10

**Status, 2026-10-04: phases 1 and 2 done; phase 3 built or closed through D-B02, E-P07 and the serve chain, whose served grade is queued (Phase 3, "The serve chain: served grade"). Open: E-P08; D-P01 and M-11 (M26, the owner's OK); T1.13 (nobara); C-N01 (aikit); the Track 4 decisions. The branch `metal-audit` merged to main on 2026-10-03, and the program continues on main (started 2026-10-01).** This is the execution plan for `docs/audit-metal-2026-09-30.md`: the
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
- **Memory.** 16 GB, with about 5 GB free by day. Since 2026-10-04 M26 may run by day on Metal, paged, guard on, with `-require-backend` (owner; `docs/tasks/task-m26-mac-2026-10.md`); before that it was off-limits on this Mac without the owner's say-so (a
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

### Batch B: pre-registration (written 2026-10-02, before any graded run; owner: run tonight, started early)

Two night jobs. Neither changes code, and each is graded or recorded as follows.

**T1.7 (B-P08), `metal-audit-t17`.** `TestAuditT17_attnStaircase` on the 1.5B `.gguf` and the 7B `.int4.metal.giw`:
- What it measures: in-sequence attention work at 1536 … 6144 keys in 128-key steps (37 depths). Production's decode
  token runs against the same token with every attention pipeline a no-op (R17's method). Each arm gets 8 tokens per
  depth, 5 reps, interleaved and alternated. Attention work is the median of the paired differences.
- **Reading 1, the audit's:** attention work ÷ keys, max over min across the depths. **Under 1.15: B-P08 killed.**
- **Reading 2, registered beside it because the two can disagree:** attention has a fixed per-token part, so its cost
  per key falls with depth even with no staircase. Reading 1 can pass on that alone. The staircase B-P08 claims is a
  jump across a trip boundary, so this reading is the 128-key step from 2048 and from 4096, each over the median of the
  other steps.
- **B-P08 stands on a model only if reading 1 is ≥ 1.15 and either boundary step is ≥ 3× the median step.** At 2–3×
  it is parked. Reading 1 ≥ 1.15 with both steps under 2× is killed (the spread is the fixed part), as is reading 1
  under 1.15.
- A by-day smoke over 3 depths (exploratory, not a result) checked that the probe runs: about 0.86–0.92 µs per key on
  the 1.5B near 1536 keys.
- Estimate about 15 minutes; queued at 25.

**T1.11's 0.5B cells and T1.14, `metal-audit-peer-b`.** One same-session `bench_peer.py` sweep on Metal: greedy,
3 runs per cell, depth 128 (phase A) and 2048 and 3900 (phase B).
- Engines: goinfer, mlx-lm 0.31.3 and Ollama 0.32.5, on the 1.5B and 7B. The 0.5B runs goinfer and Ollama only, since
  there is no mlx-community 0.5B here.
- **goinfer is main as shipped (`71812d57`), not this branch.** The ratio is about what a user runs, and the branch
  carries builds still waiting on tonight's grades.
- mlx-lm runs mlx-community's own 4-bit conversion, a different quant from q4_k_m: every mlx ratio carries that caveat.
- A by-day one-cell smoke (mlx-lm, 1.5B, depth 128, exploratory) checked that the mlx path runs under `bench_peer.py`.
- Estimate about 50–75 minutes; queued at 90.
- These are records, not gates:
  - **T1.11 (B-D01):** the 0.5B's goinfer ÷ Ollama at 2048 and 3900 (the row's pre-registered cells read 0.75× and
    0.58×). If both are ≥ 1.00, `docs/benchmarks.md`'s "AHEAD of Ollama in every cell" holds with the 0.5B added.
    Otherwise it is reworded to "every 1.5B and 7B cell" and the 0.5B cells are listed.
  - **T1.14:** goinfer ÷ mlx-lm and goinfer ÷ Ollama per cell (1.5B and 7B at 128, 2048 and 3900), the median of the
    3 runs with their spread. These replace the cross-session 0.82× and 0.75× (goinfer ÷ mlx at depth 128) as the
    ratios to quote. The box's recorded drift is about 3.5%, so a ratio within that of 1.00 is reported as level.

### C-P01: built and shipped on the branch (owner, 2026-10-02)

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

**Amendment, 2026-10-02, before any timed run (owner: "why not start with p-reading scales and weights together?").**
`2838b7de` copied the scales out of the mapping, which on a cold expert takes a page fault per 16 KB in series, and on
this path the bound is fault count (the WILLNEED record in `metal/gemma4_moe.go`); pread exists to avoid exactly that.
So the first build was the one most likely to fail this A/B. `494eb05a` preads every expert's scales with its nibbles
when they lie in the mapping (both pagers; an older file still copies from the WeightMat), tested as above with every
pread stage reading its scales too, and with an offset mutation in each pager failing its test. The A/B becomes three
arms, each its own binary:

- **old** `f56b40ec` (the cache), **copy** `2838b7de` (scales copied from the mapping), **pread** `494eb05a`.
- One discarded warm-up (old), then 8 rounds of the three in a rotating order (old/copy/pread, copy/pread/old,
  pread/old/copy, ...), 15 s between processes. Everything else as above.
- The rule above applies to **pread ÷ old**, the build that ships. **copy ÷ old** is reported, not graded: it says
  what the pread step is worth.
- Estimate: 25 processes at about 45 s, about 19 minutes; re-queued at 30. Dry-run end to end on the tiny Gemma 4 MoE
  (24 rounds' processes completed and the summary graded; its numbers say nothing, as before).

**Result, 2026-10-02 (run by day on the owner's word, 06:53–07:09 PDT): SHIP.** The queue ran the job at load 2.0–2.5
with the owner's apps open; the arms were interleaved, so all three saw the same conditions. No kill-watch fired;
swap stayed at 486 MB. Logs: `docs/measurements/metal-audit-2026-10/cp01-ab-2026-10-02/` (`summary.txt` is the graded
output).

| Arm | tok/s (8 rounds) | ÷ old, median (range) | rounds above old | Go heap at token 32 | per token: total / GPU-busy / stage (medians) |
|---|---|---|---|---|---|
| old `f56b40ec` | 9.92–10.38 | — | — | 3004.6 MB | 99.8 / 44.0 / 28.1 ms |
| copy `2838b7de` | 10.56–10.91 | 1.0674 (1.0366–1.0770) | 8 of 8 | 1553.0 MB | 93.8 / 42.4 / 24.4 ms |
| **pread `494eb05a`** | 10.49–11.02 | **1.0609** (1.0462–1.0894) | 8 of 8 | 1553.2 MB | 93.6 / 42.2 / 24.4 ms |

- **Verdict by the rule:** pread ÷ old 1.061 ≥ 0.97, and the heap fell 1451.4 MB ≥ 1.0 GB: **ship**. C-P01 is done on
  the branch, with pread as the staging path.
- **Not a slowdown but a speedup**, about 6 ms per token: staging fell 28.1 → 24.4 ms and GPU-busy 44.0 → 42.2 ms.
  The probe does not say why; less heap to keep resident on a 16 GB machine is the obvious candidate, untested.
- **Copy and pread were indistinguishable here** (1.067 and 1.061, overlapping ranges, identical staging time). The
  amendment expected copy to pay for page faults; with 73% of memory free, the scale pages were most likely already in
  the page cache. Pread is kept because it is the path built for memory pressure, which this run did not create; that
  case is unmeasured.
- The 1361 MB the old probe sized by formula against a measured 1451 MB drop: the rest is the heap's own slack around
  the freed slices, not measured separately.

**T1.9b, the re-run (06:43–06:48, same day, by day): ungraded again.** The resident guard declined the 64-slot build:
7.72 GB needed against a 4.69 GB budget, live-available memory with the owner's apps open. Swap did not move. It needs a
real night, first in the queue; re-queued. Log: `docs/measurements/metal-audit-2026-10/t19b-2026-10-02/`.

## Phase 3 — builds, in this order

### B-P03: PASSED its fidelity grade; the floor is 1024 (a same-day revert was a test artifact, undone 2026-10-03) (pre-registered 2026-10-02, before any graded run)

T1.2 put the candidate floor at 1024 keys (the block kernel 1.167× the legacy kernel on the 1.5B, 1.056× on the 7B;
768 did not qualify). Moving it puts the reassociating block kernel at 1024–1535 keys, where the exact legacy kernel runs
today, so it is graded under the owner's 2026-09-25 bar for reordering-only decode-attention kernels
(`docs/measurements/metal-decode-attn-fidelity-setb-PREREGISTERED.md`), at the new depths. The floor constant stays
1536 until this passes; the gate runs the candidate through `attnFAFloorOverride`.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-bp03-fidelity.sh` on the night queue, binary `metal-tagged-2c5cfe8f.test`, prompt set B. Deterministic: one run each. Estimate about 15 minutes; queued at 30. |
| P1 | `TestR17KernelAccuracy`, arms `bp03` (the exact kernel, production's block kernel), 10 prompts, depths 1024, 1280 and 1535, the 1.5B and the 7B from their `.int4.metal.giw` sidecars. Passes per model if the block kernel's pooled **median and p99** per-head relative L2 against float64 are each ≤ the exact kernel's, and capture sanity is 100% bit-identical. |
| P2 | `TestR17_decodeFidelityGate`, candidate `attention_fa` (production's block kernel), K = 1024 with the floor overridden to 1024, so all 64 teacher-forced positions per prompt (1024–1088 keys) run the block kernel in the candidate arm and the legacy kernel in the exact arm; the S-K1024 references, the 1.5B `.gguf`. Identity precondition: every prompt's seed-row KL ≤ 1.0, else VOID. **PASSES** on critA ∧ critB ∧ the 1.1× ceiling ∧ KL ratio ≤ 1.05; **PARKED** at 1.05 < ratio ≤ 1.10; otherwise **DOES NOT PASS**. |
| Controls | The `exact-null` candidate on the same cell (its verdict reported; a P2 fail of it is a false fail of the amended form, flagged). The 7B's P2 against its D7-K1024 references, reported, not graded: they are int8-weight CPU runs, so their level is not the S cells'. |
| Rule | **P1 passes on both models ∧ P2 PASSES (1.5B): the floor moves to 1024**, by day, with the snapshot golden that straddles it re-baked in the same commit (G-04) and the floor comments updated. **P2 PARKED**: to the owner. **Otherwise**: the floor stays at 1536, and B-P03 is recorded as failing fidelity at 1024. |

### A-P01: SHIPPED, graded 2026-10-03 (pre-registered 2026-10-02, before any graded run)

Built and bit-identical on the branch: `gemm_w4f16_tile<TM, TN>` (prefill.go), R16's GEMM with the tile as template
parameters, and `gemmTile`, which picks 32 tokens at ≤ 32 rows and 32 features at N ≤ 2048 and ≤ 64 rows, else
production's 64 × 64 `gemm_w4f16_store`. Kernel gate: 55,074,816 outputs across the 1.5B's and 7B's GEMM shapes, 8–72
rows, three epilogues and four tiles, equal to `gemm_w4f16_store` (the template at 64 × 64 included); reversing the
8-wide K chunks fails all 588 combinations. `TestMC5_prefillChunkInvariance` gained C = 16, 32, 48 after a 64-token
first chunk, so its chunks cross the selector: bit-identical on the fixture and the 1.5B (0 of 14,336,000 K/V, 0 of
151,936 logits), and red under a tile row-offset mutation. `TestPrefillRefIdentity` was not run: its prompts are 256
tokens and up, which the selector leaves on the 64 × 64 kernel. A one-rep smoke of the probes, run only to check them,
read 1.43× on the C = 32 pass and 1.69× on the gate/up GEMM at M = 32; one rep is not a result.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-ap01-grade.sh` on the night queue, from the untagged test binary `metal-e8c1e13a.test`: `TestGemmTile_bitIdentical`, then on the 1.5B and the 7B `TestAuditAP01_gemmSmallM` (each GEMM alone, shipped against the selector's tile, M = 16, 32, 48, 64, GPU time) and `TestAuditAP01_passCost` (`PrefillLast` at startPos 64 and 2048, C = 16, 32, 48, 64, under `gemmTilePolicy` shipped / bm32 / bn32 / both, wall time), 7 reps, interleaved. One KV slot, so no pass is taken by E-P01's step. Estimate about 5 minutes; queued at 15. |
| Precondition | `TestGemmTile_bitIdentical` passes. Any differing output kills. |
| Kill line for the 32-token rule | The audit's: gate/up at M = 32 under 1.15× shipped (`RESULT gate/up` on the 1.5B) drops the 32-token rule. |
| Graded | Pass wall, shipped ÷ policy, at startPos 64, C = 32 on the **1.5B**, paired per rep, median of 7 (`RESULT … both`). |
| Rule | **The default (both rules) ships if it reads ≥ 1.15× and the 32-token rule survives its kill line.** Otherwise the best single rule still standing ships if it reads ≥ 1.15× (the selector is narrowed to it). Otherwise **killed**: the selector returns the 64 × 64 kernel and the template goes. |
| Reported | The 7B, startPos 2048, C = 16, 48 and 64, and each GEMM's ratio. A shipping policy that reads below 0.98× at any reported cell goes to the owner before the merge (a threshold to move, not a rule to keep). |

### B-P04: KILLED and reverted, graded 2026-10-03 (pre-registered 2026-10-02, before any graded run)

Built and bit-identical on the branch (`22819cc8`): the SA rows kernels' lane-balanced twins (`sa_rows_acc_k512`),
taken where K % 512 == 0 (qkv, o and gate|up on the 1.5B and 7B; the 0.5B keeps the shipped kernels). Kernel gate 4608
of 4608 outputs equal; decode through the executor 0 of 40 positions differ on the 1.5B and the 7B. A one-rep smoke of
the grade harness, run only to check it, read **0.909×** (balanced slower; qkv 0.98 ms against 0.70). One rep is not a
result, but it says the audit's +3–7% projection may not hold.

| | |
|---|---|
| Instrument | `TestR18InSequence` (R18's grader) with its `pre-bp04` arm, by `docs/measurements/metal-audit-2026-10/run-bp04-grade.sh` on the night queue: binary `metal-tagged-22819cc8.test`, the 1.5B q4_k_m `.gguf` and the 7B `.int4.metal.giw`, depths 128 and 2048, 7 reps, 20 step pairs per category. Estimate about 12 minutes; queued at 25. |
| Precondition | The harness's identity lines, production and pre-bp04 each against the shipped kernels: 0 positions differ at both depths on both models. Any difference kills. |
| Graded | `B-P04 METRIC pre-bp04/production` (the previous kernels' in-sequence int4-GEMV work over the balanced kernels', paired per rep, median of 7) on the **1.5B at depth 128**, the model with the larger tail, as the audit specifies. |
| Rule | **≥ 1.02: ships** (stays the default on the branch). **Below 1.02: killed** (the audit's kill line, the owner's "park only a couple of percent" bar): the production selection and the twins are reverted, and the record keeps the numbers. |
| Reported | The 7B and depth 2048. If it ships on the 1.5B but the 7B reads below 1.00 at either depth, that goes to the owner (the selection could be made per model), before the merge. |

### C-B01: SHIPPED, graded 2026-10-03 (pre-registered 2026-10-02, before any graded run; amended the same day, before any graded run)

Built and bit-identical on the branch: the greedy chain (`metal/greedy_chain.go`, `61f21226`), widened to untied heads
in the commit after it. Each chained command buffer starts with `embed_gather_i8`, which writes the previous buffer's
argmax token's embedding into the input from the int8 embedding table on the device. It ends with the fused argmax
head `ForwardArgmax` uses (`gemv_w8a8_amax`, `argmax_finish`), which writes a 4-byte id. Token t+1's buffer is
committed before the host waits for token t, so T1.3's 0.58–0.66 ms gap has nothing to wait on. Two alternating uniform
sets carry the per-token position, key count and attention_fa split. Every 64 tokens the chain drains its autorelease
pool, one gap per 64. The decoder takes it for unbatched greedy (or `top_k=1`) decode with no processor, no adapter and
a plain table-row embedding. The resident refuses it for an embedding scale, learned positions, paged experts, an
adapter or recurrent state.

**The amendment, and why.** As first built, the chain gathered from the LM-head table and so served only tied heads.
A one-rep smoke on the real checkpoints then showed both bench Qwen GGUFs to be untied: each ships a separate
`output.weight` (Q6_K) beside `token_embd` (Q4_K), although Qwen2.5 ties the two in training. The 7B is untied too. So
the chain as first built served none of the bench set. The owner chose a device copy of the embedding table
(2026-10-02). `chainEmbedTable` makes it the first time a chain is asked for, never for a resident that only samples,
and aliases it from the `.giw` mapping where it can. It costs 136 MB on the 0.5B, 233 MB on the 1.5B and 545 MB on the
7B. It is priced first, on top of the build's own memory-guard price and against the memory available now. A table that
does not fit declines the chain, and decode keeps the full-logits path. A tied head gathers from the LM-head buffers
at no cost.

Identity, by day, tied and untied fixtures both:
- `TestGreedyChain_bitIdentical`: 80 tokens through `chainStart`/`chainNext`, with the attention_fa floor moved mid-run.
  0 tokens and 0 K/V elements differ, and the token after the chain matches.
- `TestGreedyChain_generateMatchesFullLogits`: 150 tokens through `Generate` over three generations, the third reusing
  the second's 54-token prefix. They equal the full-logits path's, and all 150 came from the chain. A third case, an
  untied head whose table has no room, declines the chain (0 served), and its tokens are unchanged.
- Mutations, each failing the tests: gathering from the chain's own output buffer, binding the wrong uniform set, not
  switching the split, opening the chain one position early, and an untied chain gathering from the LM-head table
  (the last fails only the untied cases).
- On the real 0.5B and 1.5B `.gguf`, one rep of 64 tokens: the chain's tokens equal the full-logits path's. Its timing
  was a smoke, exploratory, and is not a result.

| | |
|---|---|
| Instrument | `TestCB01ChainAB` by `docs/measurements/metal-audit-2026-10/run-cb01-grade.sh` on the night queue. One process per model: the 1.5B and 0.5B q4_k_m `.gguf` and the 7B `.int4.metal.giw`, at int4. A 64-token prompt, 256 greedy tokens per arm, 9 reps, the arms alternated (off-on, then on-off), and one warm-up of each discarded. Each arm's rate is its decode rate, from the first token's arrival to the last's. Estimate about 10 minutes; queued at 20. |
| Precondition | Every rep's two token streams are equal, the chain arm is served by the chain (at least 255 tokens), and the off arm not at all. The test fails on any miss, and a failure kills. |
| Graded | `C-B01 METRIC chain/off greedy` on the **1.5B**: the median over 9 reps of the per-rep ratio of decode rates, chain over off. (The instrument discards one cold generation before its warm-up, so every compared arm reuses the prompt the same way; see C-P02 below.) |
| Rule | **≥ 1.02: ships**, and the chain stays the default on the branch. **Below 1.02: killed**, by the owner's "park only a couple of percent" bar, since the chain adds a goroutine path to the decode loop and a resident table: the decoder's `useChain` and the resident's chain are reverted, and the record keeps the numbers. |
| Reported | The 0.5B and the 7B, each with its pairs-above-1 count. T1.3's gap predicts a larger share on the 0.5B (0.58 ms of a ~5.7 ms token, against 0.66 of ~13.4). If the 1.5B is killed but the 0.5B reads ≥ 1.02, that goes to the owner before the revert. A 7B below 1.00 also goes to the owner, since the selection could be made per model. |

### C-P02: SHIPPED, graded 2026-10-03 (pre-registered 2026-10-02, before any graded run)

Built and bit-identical on the branch: the sampled chain, C-B01's chain for temperature-only sampling. The decode
loop's device draw (`ForwardSample`) ran one synchronous command buffer per token, outside the encode-ahead executor
and with the host between tokens. The chain ends each buffer with `ForwardSample`'s own draw (the full LM-head row,
`gumbel_stage1`, `gumbel_stage2`) and commits the next before the host waits.

- **Draws.** `Sampler.NextDraw` keeps the seed fixed and counts the draw up, so the chain draws each forward with the
  start draw plus its offset. Every `SampleChainNext` passes the decoder's own draw for that forward, and the resident
  refuses one that is not the draw it used. The decoder takes its draws exactly as before.
- **Nothing comparable.** A row with no finite logit makes `gumbel_stage2` write −1. `ForwardSample` then takes the
  row's argmax, and so does the chain, from that forward's own logits row (one per uniform set). The buffers queued
  after it gathered past the vocabulary: `embed_gather_i8` now writes zeros there. They are waited out and re-committed
  from the argmax.
- **Where it runs.** Wherever the device draw serves the token, with no MC3, adapter or processor and a plain table-row
  embedding: today that is temperature-only sampling above 0.2 (at or below, optimistic forward runs).

Identity, by day:
- `TestSampleChain_bitIdentical` covers tied and untied fixtures at T=0.9: 80 tokens (80 distinct, so the draw is
  sampling), 0 tokens and 0 K/V elements differ against per-token `ForwardSample`. A wrong draw is refused.
- A third case gives one position's draw a row of −inf, so the device really writes −1. It matches `ForwardSample`'s
  fallback, with every later token and K/V element unchanged.
- `TestSampleChain_generateMatchesForwardSample` runs the same through `Generate` (150 tokens, a reused prefix
  included). The tokens are equal, every decode token is device-drawn on both arms, and all are served by the chain.
- Mutations: the draw written to the wrong uniform set fails every case. No wait-out and re-commit after a −1 fails the
  third case (62 of 80 tokens).
- On the real 0.5B and 1.5B `.gguf` at T=1.0, the chain's tokens equal `ForwardSample`'s in every rep of a smoke. Its
  timing was exploratory and is not a result.

**The instrument needed one fix first.** The smoke's first version compared a cold first generation with a warm second
one, and at T=1 they parted. A diagnostic showed two chain-off arms, cold then warm, also part at token 18, while chain
against off, both warm, never did. The warm arm reuses 63 of 64 prompt tokens and re-forwards the last on the decode
path, whose logits differ by ulps from the pass's: E-P01's effect (log, 2026-10-02), which a near-tie Gumbel draw
shows and an argmax does not. `TestCB01ChainAB` now discards one cold generation, so every compared arm is warm.

| | |
|---|---|
| Instrument | `TestCB01ChainAB` with `GOINFER_METAL_CB01_TEMP=1.0` (the serve default), seed 11, in the same night job as C-B01's grade (`run-cb01-grade.sh`): the same models, prompt, 256 tokens, 9 reps and alternation. |
| Precondition | As C-B01's: every rep's two token streams are equal, and the chain serves at least 255 tokens and the off arm none. In addition, every token after the first is device-drawn on both arms. |
| Graded | `C-B01 METRIC chain/off sampled T=1` on the **1.5B**. |
| Rule | **≥ 1.02: ships. Below 1.02: killed**, and the sampled chain alone is reverted (`ResidentSampleChain` and its branch). C-B01's grade decides the greedy chain separately. |
| Reported | The 0.5B and the 7B, as for C-B01. The audit's band for the missing encode-ahead alone was up to 3.6–4.2% on the 0.5B. The chain also removes the host round trip, which C-B01's T1.3 measured. |

### E-P06: KILLED and removed, graded 2026-10-03 (pre-registered 2026-10-02, before any graded run; the first night's run failed its idle gate, re-queued)

Built and lossless on the branch. `--spec ngram`'s depth controller prices a step-kernel verify by a cost curve, and
every Metal model reported one constant, `stepVerifyCost`: the 7B at depth 2048. Now a model loaded for speculation
(`decoder.Options.SpecNgram`, set by serve's and chat's `--spec ngram`) measures its own curve while the resident is
built (`calibrateVerifyCost`):
- What runs: production's token and the argmax-only verify at 2, 4 and 8 rows, on slot 0 at depth 2048 (the
  constant's own depth, so the curve stays as conservative in depth and differs only by model). The arms are
  interleaved over 7 rounds, the first 2 discarded, and each GPU-time median is taken.
- The curve keeps the constant's shape (3 and 5–7 interpolated, 9 an 8-row step plus one token).
- A failed or non-rising measurement keeps the constant, and so does a load without `SpecNgram`.
- Lossless: the curve sets only how many tokens a round drafts, never which ones come out.

Day checks:
- `TestVerifyCurveFrom` pins the arithmetic and the refusals.
- `TestVerifyCost_measuredAtLoadLossless`: a `SpecNgram` load measures the curve, and greedy after it equals greedy
  without it. Adaptive speculation with that curve equals plain greedy, through a drafter that proposes greedy's
  continuation with every 5th token wrong (88 drafted, 51 accepted). A load without `SpecNgram` reports the constant.
  `VerifyCost` ignoring the measured curve fails the test.
- Two first versions of the test checked nothing, and the second caught the first. An n-gram drafter on random weights
  proposes nothing (its match includes the token just generated, which a random model does not repeat). A first
  greedy comparison also paired a warm generation against a cold one: the same confound as C-P02's below.
- On the real models, a load-only look (exploratory, not a result): 2, 4 and 8 rows cost **1.20, 1.64 and 2.12**
  tokens on the 0.5B and **1.64, 1.93 and 2.44** on the 1.5B, against the constant's 1.79, 2.11 and 2.65. The load
  cost was within load-time noise.

So at depth 2048 the 1.5B's curve is 8–9% below the constant. The audit's 12–51% compared the 1.5B at depth 128 with
the 7B at 2048, so part of it was depth, which this does not take. The 0.5B's curve moves most.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-ep06-grade.sh` on the night queue, graded by `gates-ep06.py`. It is the spec record's served harness (`metal-spec-step-verify-2026-09-27.md`): copy (`bench_spec_copy.py`) and W7 chat (`bench_w7_plain.py`, fixed nonce), one client, greedy. Three arms rotated per round: plain (no spec, the do-nothing arm), spec-old (`--spec ngram`, the constant: serve at E-P06's parent) and spec-new (`--spec ngram`, the model's own curve). 5 rounds on the 1.5B, since the record's chat rounds spread 1.066–1.317×, and 3 on the 0.5B and 7B. Estimate about 50 minutes; queued at 60. |
| Precondition | Every spec reply equals plain's, on every model and workload. A difference kills. |
| Graded | `E-P06 METRIC`: on the **1.5B chat**, the median over 5 rounds of the per-round ratio spec-new ÷ spec-old. |
| Rule | **≥ 1.02 on chat, with copy on the 1.5B ≥ 0.98: ships.** Otherwise killed: `SpecNgram`'s calibration and the measured curve are reverted, and the record keeps the numbers. |
| Reported | The 0.5B (the largest curve change) and the 7B (whose curve should sit near the constant it was measured from: a check on the method), each against plain as well. |

### E-P03: KILLED and reverted, graded 2026-10-03 (pre-registered 2026-10-02, before any graded run)

Built and bit-identical on the branch. The MC3 batched step ran its four fragment projections (qkv, o, gate|up, down)
at FB = 2 on every model, where S0 recorded FB = 4 best for the 7B's gate|up and down.
- **The build:** `mc3_bt` and `mc3_btd` are now instantiated at FB = 4 beside 2, and `calibrateFB` picks per
  projection at build time. It times both on the first layers' real weights, the best of 10 interleaved reps after 3
  warm-ups, as `calibrateRows` does. It takes FB = 4 only where the projection's rows divide by 32 and FB = 4 is at
  least 3% faster. `calibrateRows` then runs on the chosen FB.
- **Why it cannot change a bit:** FB changes which rows a simdgroup owns, not how a row is summed.
- **`TestMC3Step_fb4BitIdentical`:** forces FB = 4 on every projection, with the fragment at every batch size. Over
  4 sequences × 12 steps, 0 logits differ from production's single-token forward. Halving FB = 4's grid fails every
  logit.
- **On the real 1.5B** (two loads, exploratory): calibration keeps FB = 2 everywhere. gate|up reads 0.250 against
  0.277 ms and down 0.136 against 0.164, as S0 recorded for that model.
- **A one-rep smoke on the 0.5B,** where calibration also keeps FB = 2 (so both arms are the same build), read
  0.97–1.10 per cell. That is the single-rep noise, which the rule below accounts for.

| | |
|---|---|
| Instrument | `TestEP03StepAB` by `docs/measurements/metal-audit-2026-10/run-ep03-grade.sh` on the night queue: the 7B `.int4.metal.giw` and the 1.5B `.gguf`, 8 KV slots at a 2560 context. B = 4 and 8 sequences at depths 128 and 2048. 9 reps, each running the calibrated FB and FB = 2 everywhere, alternated, 6 steps each at the same positions and inputs. An arm's time is the median step GPU time. Estimate about 25 minutes (filling 8 slots to 2048 is most of it); queued at 35. |
| Precondition | Every rep's first-step logits are equal across the arms. A difference kills. |
| Graded | `E-P03 METRIC fb2/calibrated` on the **7B at B = 4, depth 128** (the audit's B ≥ 3 band). |
| Rule | **≥ 1.02 with at least 7 of 9 reps above 1: ships.** Otherwise killed: `calibrateFB` and the FB = 4 instantiations are reverted, and the record keeps the numbers. If the 7B's calibration itself picks FB = 2 everywhere, the arms are the same build, and E-P03 is killed on the record of what it picked. |
| Reported | The 7B's other cells. The 1.5B as an A/A control when its calibration keeps FB = 2 (expected): its readings are the noise floor the 7B's are read against. |

Each item ships behind its own pre-registered band and kill line from §10. "Gated" means the pooled or set-B fidelity
gate runs at night before it ships.

1. **E-P01:** short prompts through the 8-row step. Bit-identical, so no fidelity gate. **Done 2026-10-02 below the
   floor; the above-floor half is held** (log, 2026-10-02).
2. **Decode attention, one campaign on one harness:** B-P03 (T1.2's candidate floor is 1024; re-bakes the snapshot golden that
   straddles the floor), B-P02, then B-P01. All gated.
3. **Small-M prefill:** A-P01 (bit-identical; **shipped** 2026-10-03, 1.804× on the 1.5B's C = 32 pass), then A-P02 (gated). Then Metal int8 slice 2 on the same tile selector.
4. **Decode GEMV residue:** B-P04 (**killed** 2026-10-03: 0.955×, the balanced kernels do more work than the idle tail costs), B-P06 (**killed** 2026-10-04: 256 threads beats 64, 128 and 512 on the 1.5B and 7B, every rep), and B-P05 only after O4, with the MC3 down kernel moved in
   the same change.
5. **C-B01:** the on-device token chain (bit-identical), with C-P02 as its sibling. **Both shipped** 2026-10-03 (1.5B 1.085× greedy, 1.066× sampled). T1.3: a 0.58–0.66 ms GPU-idle gap
   per token.
6. **MoE and hybrids:** D-G01's gate first. Then D-B01 on the Qwen3.5-9B hybrid, if the fit guard admits it resident
   at night; then D-B02 (T1.12 **probed** 2026-10-04: the routing sync is 1-2% of the pass, tile padding 1.47× at M = 512 is the lever; it also found the pass-scratch copy, **fixed**; the 16-row tile **shipped**: 1.09× on the slice's 512-token pass, 1.56× / 1.73× on the 1.5B's / 7B's 16-token pass), D-P04 (**shipped** 2026-10-04: bit-identical, 1.060× on the slice's token, 68 us per MoE layer), D-P03 (**parked** 2026-10-04: bit-identical, 1.019× on the slice's token, off by default) and D-B04. D-P01 needs M26 and so the owner's OK.
7. **The batched step:** E-P03 (**killed** 2026-10-03: no 7B projection is 3% faster at FB = 4), E-P02 (**killed** 2026-10-04: 0.777 ms against its 0.65 ms line, 1.11× on the 7B gate|up at B = 2), E-P05 (**parked** 2026-10-04: bit-identical, 1.016× at 2 rows to 1.06× at 8, off by default), E-P06, E-P08, E-P07. Metal int8 slice 3 joins here.
8. **Memory:** C-P01 (**done** 2026-10-02: −1451 MB of heap on M26, decode 1.061×), E-P09 (**done** 2026-10-02: 2 slots by default on Metal), F-D02.

Not planned until a probe says otherwise: the "not worth a probe" list at the end of §10, and B-P08 (T1.7 ran: it stands on the 7B, parked on the 1.5B; **the owner parked it on both on 2026-10-03**, low value).

### The night of 2026-10-02: results (graded 2026-10-03)

Raw results, provenance and the peer JSON: `docs/measurements/metal-audit-2026-10/night-2026-10-02/`. Every graded line
below is read against its pre-registration above, unchanged.

| Item | Graded reading | Verdict |
|---|---|---|
| **C-B01** greedy chain | 1.5B chain/off **1.085×** (1.032–1.151), 9 of 9 pairs above 1; identity held in every rep. Reported: 0.5B 1.063× (9/9), 7B 1.010× (9/9) | **ships** (≥ 1.02) |
| **C-P02** sampled chain, T = 1 | 1.5B **1.066×** (1.060–1.078), 9/9; every token device-drawn on both arms. Reported: 0.5B 1.160×, 7B 1.022× | **ships** |
| **A-P01** small-M prefill tiles | 1.5B pass wall shipped ÷ both rules at startPos 64, C = 32 **1.804×** (pairs 1.8 × 7); bm32 alone 1.487×, bn32 alone 1.152×; `TestGemmTile_bitIdentical` passed. Reported: 7B both 1.553× (bn32 1.000×) | **ships** (the default, both rules) |
| **B-P04** lane-balanced SA rows | 1.5B depth 128 pre-bp04 ÷ production **0.955×** (0.903–0.962): the balanced kernels do more in-sequence work. 7B 0.911–0.917× | **killed**, reverted |
| **E-P03** FB = 4 fragments | The 7B's calibration chose FB = 2 everywhere (gate\|up 1.158 against 1.165 ms, down 0.594 against 0.584: under the 3% bar), so the arms were one build; readings 0.999–1.001 on both models | **killed** on the record of what it picked, reverted |
| **T1.7** (B-P08) | Per-key max/min 1.707 (1.5B), 1.679 (7B). Boundary step ÷ median step: 1.5B 2.53 at 2048 and 2.50 at 4096; 7B 4.81 and 4.55 | 1.5B **parked** (2–3×); 7B: **B-P08 stands** — then **parked by the owner, 2026-10-03** (low value, not planned) |
| **C-B03** fence (aikit probe) | p50 saving 88.5 µs per boundary (190.3 against 101.8) | **REPORT**: goinfer decides whether paged MoE takes it |

**T1.11 and T1.14** (records, not gates). One same-session `bench_peer.py` sweep on Metal: main as shipped (`71812d57`),
Ollama 0.32.5, mlx-lm 0.31.3, greedy, 3 runs per cell, the instant idle gate, no thermal warning. Decode tok/s and
goinfer's ratios:

| Cell | goinfer | Ollama | mlx-lm | ÷ Ollama | ÷ mlx-lm |
|---|---:|---:|---:|---:|---:|
| 0.5B d128 | 166.6 | 145.8 | — | 1.143 | — |
| 0.5B d2048 | 110.0 | 136.8 | — | **0.804** | — |
| 0.5B d3900 | 72.6 | 127.9 | — | **0.568** | — |
| 1.5B d128 | 90.5 | 85.1 | 108.6 | 1.063 | **0.833** |
| 1.5B d2048 | 82.2 | 73.7 † | 97.3 | 1.115 † | 0.845 |
| 1.5B d3900 | 77.6 | 75.6 | 86.8 | 1.026 | 0.894 |
| 7B d128 | 30.3 | 25.0 | 36.6 | 1.212 | **0.828** |
| 7B d2048 | 26.4 | 24.1 | 32.0 | 1.095 | 0.825 |
| 7B d3900 | 25.0 | 23.3 | 33.9 | 1.073 | 0.737 |

- † Ollama's 1.5B at 2048 spread 15.3% across its 3 runs, so that cell's ratio is not quoted. Every other cell spread
  4.5% or less.
- **T1.11 (B-D01):** the 0.5B trails Ollama at depth (0.80× and 0.57×; it does not reach the block attention kernel,
  B-P01). The benchmarks row already scopes "AHEAD" to the six 1.5B and 7B cells (T0.5), and now carries these cells.
- **T1.14:** at depth 128, goinfer ÷ mlx-lm is **0.833× on the 1.5B and 0.828× on the 7B**, same session. These replace
  the cross-session 0.82× and 0.75×: the 1.5B's gap was right, and the 7B's was overstated by 8 points. mlx-lm runs its own
  4-bit conversion, not q4_k_m, and every mlx ratio carries that caveat.

**Not graded, re-queued:**
- **E-P06:** its served harness's own idle gate waits for load ≤ 1.0, which this Mac never reached with VS Code open
  (1.1–3.6 over 30 minutes). It is re-queued on `bench_peer`'s instant gate (busy ≤ 10%), the darwin default since TE1.
- **B-P03:** all five arms failed in seconds. The script runs the test binary from `~/goinfer-bench`, and the gate's
  prose seed is read from a path relative to `metal/` (`../testdata/...`). The script's `|| true` hid each arm's exit
  code, so the runner saw success. The script now runs from the worktree's `metal/` and exits non-zero on any arm that fails.
- **T1.9b:** the memory guard declined the 64-slot M26 build again (7.72 GB needed, 7.02 GB budget).
  **Graded 2026-10-03 on a forced run: R11(c) stands at 64 slots.** The owner chose a single run by day with the guard
  bypassed (`GOINFER_NO_RESIDENT_MEM_GUARD=1`), on the same binary (`f56b40ec`) and the same kill-watch (+1 GB, or two
  80 MB ticks). The build finished in 11.3 s. One second later swap jumped +1.60 GB in a single 1 s sample
  (780 → 2382 MB), RSS fell from 6.4 to 0.6 GB as it was paged out, and the kill-watch killed the process at t+12 s,
  before the first token. The pre-registered row: "the kill-watch fires: R11(c) stands". Deviations: by day, with
  VS Code open, and the guard bypassed. The guard's refusal was the right call. Logs:
  `docs/measurements/metal-audit-2026-10/t19b-forced-2026-10-03/`.

### Post-merge peer read: read 2026-10-03, consistency check FAILED and investigated (pre-registered 2026-10-03, before it ran)

TE5(a): the levers were graded new ÷ old with no peer arm, and the peer ratio is read once, after they ship, as its
own run. The merge (`c6f8100e`) changes what last night's T1.14 cells measured (C-B01 and C-P02 shipped), so
`docs/benchmarks.md`'s Metal decode row describes a build main no longer runs.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-peer-post-merge.sh` on the night queue. One same-session `bench_peer.py` sweep on Metal: goinfer at `c6f8100e`, goinfer_old at `71812d57` (last night's T1.14 build), mlx-lm 0.31.3 and Ollama 0.32.5. Greedy at depth 128, 2048 and 3900, and `temp1.0_notrunc` at depth 128; the 0.5B, 1.5B and 7B; 3 runs per cell, the instant idle gate. Estimate about 45 minutes; queued at 60. |
| Reading | A record, not a gate. Per cell: goinfer ÷ goinfer_old, goinfer ÷ Ollama and goinfer ÷ mlx-lm, with each run's spread. These replace T1.14's cells in the benchmarks row. |
| Consistency check | The new ÷ old cells must agree in direction with the graded in-process A/Bs. At depth 128 greedy that means ≥ 1.00 on all three models, and at T = 1 the same (C-P02: 1.5B 1.066×, 0.5B 1.160×, 7B 1.022×). A cell that reads below 1.00 beyond its own spread is investigated before any row is updated. The served number is end to end, so its size may differ from the in-process ratio, but its sign may not. |
| Not read here | A-P01 (prefill passes; `bench_peer` times decode only), C-P01 and E-P09 (memory). |

### B-P01: SHIPPED, graded 2026-10-03 (pre-registered 2026-10-03, before any graded run)

Built on main, **off by default** (`attnFABlk64On = false`): `attention_fa_blk64`, the block decode-attention kernel
at head dim 64 (Qwen2.5-0.5B, G = 7).
- **The kernel:** the hd = 128 block kernel's shape exactly, with 2 dims per lane in place of 4 (`half2` K/V loads,
  `float2` q and accumulators). It has the same grid, split rule (16) and partial layout, so `attention_fa_combine`
  merges it unchanged.
- **Reach:** `attnFAHeadDimOK` admits an hd = 64 layer only where the twin exists for its group size, so an hd = 64
  layer can never reach the hd = 128 `attention_fa`. The graded hd = 128 kernel's text is untouched.
- **Fidelity:** not bit-identical to the per-query-head kernel it replaces; the block softmax reassociates, as at
  hd = 128. So it is gated, on the amended decode-attention bar (2026-09-25).

By day:
- **Accuracy:** `TestR17KernelAccuracy` (arms `bp03`) on the 0.5B, 3 prompts at 1024, 2048 and 3900 keys, 3,024
  heads. Against float64 the twin's per-head relL2 is a median **1.66e-7** and p99 2.84e-6, the exact kernel's
  3.80e-7 and 5.59e-6; the twin is closer on 2,653 heads. The harness now admits an hd = 64 layer only under the
  hd-general arms (exact and production).
- **Speed, one rep, exploratory:** `TestBP01AttnAB` read legacy ÷ block attention 3.21× at 2048 keys and 3.84× at
  3900 (S = 16), and the token 1.59× and 1.96×.
- **The reference cell:** `TestPrefillGateReference` gains the 0.5B as cell **Q05** (f32, like S), built only when
  `GOINFER_CPU_REF_MODELS` names it. A K = 64 smoke built all 10 prompts in 17 s; those files were removed.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-bp01-grade.sh` on the night queue, from pinned tagged binaries, run from the checkout's package directories. (1) Build the Q05 references at K = 2048 and 3900, set B, on the CPU. (2) The fidelity gate (`TestR17_decodeFidelityGate`, `GOINFER_METAL_BLK64=1`) on the 0.5B against Q05 at K = 3900 and K = 2048, with candidate `attention_fa` (production: the twin) and, at 3900, the `exact-null` control. (3) P1: `TestR17KernelAccuracy`, arms `bp03`, 10 prompts at 2048 and 3900. (4) Speed: `TestBP01AttnAB`, 5 reps of 8 tokens per arm at 2048 and 3900, arms interleaved and rotated, the no-op arm subtracted per rep. Estimate about 45–60 minutes, most of it the references; queued at 75. |
| Precondition | Every step's test passes (the job exits non-zero on any failure). The reference files exist for all 10 prompts at both depths. |
| Graded | **P1:** the twin's median and p99 per-head relL2 vs float64 at most the exact kernel's. **P2 at K = 3900:** critA, critB and the 1.1× ceiling as printed, and the KL ratio candidate ÷ exact. **Speed:** `B-P01 METRIC legacy/blk S=16` at 2048 and 3900. |
| Rule | **Ships** (`attnFABlk64On` becomes true) if P1 holds, P2's critA, critB and ceiling hold, the KL ratio is ≤ 1.05, and the speed ratio is ≥ 1.5 at either depth. A KL ratio of 1.05–1.10 parks it. **Killed** (the twin and its switch removed) if P1 fails, P2 fails outright, or the speed ratio is below 1.5 at both depths (the audit's kill line). K = 2048 and the exact-null control are reported beside the decision. A null KL ratio outside 0.95–1.05 says the cell's own spread is wider than the bar, and that goes to the owner before shipping. |
| Reported | The S = 8, 24 and 32 arms, and the token ratio at both depths. |

### B-P02: SHIPPED, graded 2026-10-03 (pre-registered 2026-10-03, before any graded run)

Built on main, **off by default** (`attnFABlkAnyG = false`): the hd = 128 block decode-attention kernel instantiated at
G = 2, 3, 4, 5 and 8, beside the graded 6 and 7. A dense hd = 128 model of another group size ran the legacy
`attention_fa` at its core-count split. internlm2-1_8b and Qwen3-0.6B (16 query heads over 8 KV heads, G = 2) both
reach `attention_fa` today and take the legacy kernel.

**Fidelity is inherited, not re-gated.** `TestAttnFABlk_anyGMatchesG7` runs every new size against the graded g7 on
the same K/V and leading query heads (777 keys, 16 splits). Every per-head partial agrees bit for bit: 0 of 43,680
values differ. Comparing each head with g7's next head fails all of them, so the check can fail. The per-head
arithmetic is therefore the graded kernel's; only G, the loop bound, changes. This departs from the audit's "run
R17's gate once per new G". No CPU reference exists for a G ≠ 6/7 model here, and bit-exact per-head identity with
the gated kernel is stronger evidence than a KL ratio with its ±5% rounding-order spread. The owner can ask for the
per-G gate instead.

By day: a one-rep `TestBP02AttnAB` smoke (exploratory) on internlm2 read legacy ÷ block attention 2.37× at 2048 keys
and 2.30× at 3900, and the token 1.24× and 1.37×.

| | |
|---|---|
| Instrument | `TestBP02AttnAB` by `docs/measurements/metal-audit-2026-10/run-bp02-grade.sh` on the night queue: internlm2-1_8b, and Qwen3-0.6B reported, at int4, 2048 and 3900 keys, 5 reps of 8 tokens per arm (legacy `attention_fa`, the block kernel at S = 16, no-op), interleaved and rotated, the no-op arm subtracted per rep. Estimate about 10 minutes; queued at 20. |
| Precondition | Both models load with the block kernel selected at G = 2, and the test passes. |
| Graded | `B-P02 METRIC legacy/blk` on **internlm2** at 2048 and 3900. |
| Rule | **≥ 1.5 at either depth, with at least 4 of 5 reps above 1: ships** (`attnFABlkAnyG` becomes true). Below 1.5 at both depths: **killed**, the instantiations and the switch removed (R17's and B-P01's kill line). |
| Reported | Qwen3-0.6B, and the token ratio. |

### D-B04: SHIPPED, graded 2026-10-03 (pre-registered 2026-10-03, before any graded run)

Built on main, **off by default** (`gemvExtOn = false`), bit-identical. R18's rows-per-simdgroup form now reaches the
int4 GEMV sites it never did, through `gemvExt`:
- **DeltaNet's qkv and z projections** (24 of the Qwen3.5-9B's 32 layers).
- **A gated shared expert's gate|up and down.**
- **An ungated shared expert's down.**

**A correction to the audit:** its "bit-identical by R18's own argument" did not hold for the DeltaNet sites as
written. They run the coal kernel family, and R18's SA rows kernels sum in a different order. The fix is a new kernel,
`gemv_w4a8_coal_staged<R>`: R18's staged down kernel with the coal epilogue (no residual), so its sums are the coal
kernel's.

**Not built:** the routed-expert rows kernel (a new MoE kernel variant), and nothing on this Mac would time it
resident. **Built 2026-10-04 (owner), below: "D-B04's routed-expert half".**

By day:
- **`TestGemvExt_bitIdentical`** on the tiny Qwen3.5 hybrid (DeltaNet, MoE with a gated shared expert): 0 of 6,144
  logits differ over 24 teacher-forced positions, with 240 coal-rows and 96 SA-rows dispatches taken. Scaling the new
  kernel's output by 1 + 1e-7 fails 619.
- **No coverage for the ungated shared-expert site:** no tiny fixture builds such a resident. It dispatches R18's
  production down kernel.
- **On the real 9B:** a one-rep `TestDB04AB` smoke (exploratory) read the token 1.060× faster at depth 128 and 1.084×
  at 1024. The last token's logits matched across the arms.
- **A first smoke reported a logits difference. It was the test's design:** DeltaNet's recurrent state is not
  indexed by position, so the second arm ran on the first's state. The instrument now snapshots and restores it.

| | |
|---|---|
| Instrument | `TestDB04AB` by `docs/measurements/metal-audit-2026-10/run-db04-grade.sh` on the night queue: the Qwen3.5-9B `.int4.metal.giw`, depths 128 and 1024 (filled by decode tokens), 7 reps of 16 tokens per arm, alternated, the DeltaNet state restored before each arm. An arm's time is its token GPU-time median. Estimate about 8 minutes; queued at 15. |
| Precondition | The last token's logits are equal across the arms in every rep (the test fails otherwise), and rows-form dispatches are counted. |
| Graded | `D-B04 METRIC off/on` at **depth 128**. |
| Rule | **≥ 1.02 with at least 6 of 7 reps above 1: ships** (`gemvExtOn` becomes true; bit-identical, so the owner's permissive bar, not the audit's 1.05). **Below 1.02: killed**, `gemvExt` and the coal rows kernel removed. |
| Reported | Depth 1024. |

### The night of 2026-10-03: results (graded 2026-10-03)

Raw results and provenance: `docs/measurements/metal-audit-2026-10/night-2026-10-03/`. The Mac's night queue ran by day
at the owner's word, 11:28–13:38. Every graded line is read against its pre-registration above, unchanged.

| Item | Graded reading | Verdict |
|---|---|---|
| **B-P01** hd = 64 block twin, 0.5B | P1: the twin's median and p99 relative L2 against float64 are 1.66e-7 and 3.01e-6, against the exact kernel's 4.12e-7 and 5.55e-6; capture sanity 480/480. P2 at 3900 keys: critA, critB and the ceiling hold, KL ratio **1.0098**. Speed, legacy ÷ block attention: **3.17×** at 2048 keys, **3.63×** at 3900 (1.52× and 1.92× per token). Reported: the exact-null control PASSES at 3900; the K = 2048 cell holds critA, critB and the ceiling, with KL ratio 1.0113 | **ships** (P1, P2, KL ≤ 1.05, speed ≥ 1.5) |
| **B-P02** block kernel at G = 2, 3, 4, 5, 8 | Legacy ÷ block attention on the two G = 2 models: internlm2-1.8b **2.37×** and **2.32×** at 2048 and 3900 keys, qwen3-0.6b **2.47×** and **2.30×**; 5 of 5 reps above 1 everywhere (per token 1.24–1.62×) | **ships** (≥ 1.5, ≥ 4 of 5 reps) |
| **D-B04** staged rows GEMV | 9B off ÷ on: **1.064×** at depth 128 and **1.061×** at 1024, 7 of 7 reps above 1, the last token's logits equal in every rep; 5,376 and 10,752 rows-form dispatches | **ships** (≥ 1.02, ≥ 6 of 7) |
| **B-P03** floor at 1024 (reverted, then restored: see below) | P1 at 1024, 1280 and 1535 keys: the block kernel's median and p99 against float64 are below the exact kernel's on both models (1.5B 1.68e-7 / 1.83e-6 against 4.27e-7 / 3.68e-6; 7B 2.33e-7 / 1.90e-6 against 5.87e-7 / 5.31e-6), capture sanity 840/840 each. P2 on the 1.5B at K = 1024 **PASSES** (KL ratio 0.9987, identity 10/10). Controls: exact-null passes under the amended form (KL 1.0002); the 7B's P2, reported, holds the amended form (KL 0.9888) | **floor moves to 1024** |
| **E-P06** verify-cost curve at load | 1.5B chat spec-new ÷ spec-old **1.014×** (bar 1.02); copy 0.998×. Every spec reply equals plain's. Reported: the 0.5B 1.237× copy and 1.084× chat; the 7B 1.000× and 1.006× | **killed** (owner: "killed is good"), code removed |

**E-P06 did run the code it grades** (owner question, 2026-10-03). Speculation was live in both spec arms: 1.5B copy
decodes about 125 tok/s against plain's 55. Every spec-new server logged its measured curve at load, and the old binary
never did:

| Model | Measured verify cost, 2 / 4 / 8 rows | Against the constant 1.79 / 2.11 / 2.65 |
|---|---|---|
| 0.5B | 1.13 / 1.63 / 2.03 | much cheaper |
| 1.5B | 1.58–1.61 / 1.81–1.85 / 2.23–2.29 | somewhat cheaper |
| 7B | 1.76 / 1.99 / 2.35 | about the constant, which was the 7B's |

The gains track the gap: double digits on the 0.5B, under the bar on the 1.5B (whose copy workload drafts at full depth
either way), none on the 7B. The pre-registration grades the 1.5B, so the kill stands on its rule. The 0.5B's win is
recorded here, not acted on.

**The post-merge peer read** (records, not gates; `night-2026-10-03/peer-post-cells.txt` has every cell). goinfer
`c6f8100e` ÷ goinfer_old `71812d57`, depth 128:
- greedy: 0.5B 1.023×, 1.5B **0.991×**, 7B 1.004×;
- T = 1: 0.5B 0.999×, 1.5B 0.997×, 7B 0.997×.

The pre-registered consistency check asks for at least 1.00 on all six. C-B01's in-process 1.5B was 1.085×, and C-P02's
1.066× / 1.160× / 1.022×. **It fails, and was investigated, so no benchmarks row is updated.** The cause is a
composition gap, not noise:
- **Serve on Metal always batches.** With more than one KV slot `m.batcher` exists, and `Model.Generate` gives every
  generation an MC3 place (the `bt.claim` branch in `decoder/model.go`), a lone client included.
- **The chains require no batcher.** Both the greedy chain (`useChain`) and the sampled chain require `mc3 == nil`.
- **E-P09 made 2 slots the default** for Metal serve. So on main **neither chain reaches serve.** They run in the
  single-session CLI and in their own in-process A/Bs, which have no batcher. That is CLAUDE.md's rule about a unit test
  that supplies its own calling convention.

Goinfer against the peers, from the same sweep:
- goinfer ÷ Ollama at depth 128 is 1.19× (0.5B), 1.04× (1.5B) and 1.22× (7B).
- goinfer ÷ mlx-lm is 0.83× (1.5B) and 0.83× (7B). The 0.5B has no MLX checkpoint configured.
- At depth: the 0.5B falls behind Ollama, 0.80× at 2048 and 0.60× at 3900. The 1.5B and 7B stay level to ahead.

**What it leaves for the owner:** letting a lone MC3 generation (B = 1) take the chain, or the chain into the batched
step, is a new build with its own gate. Until then the C-B01 and C-P02 numbers describe the CLI and single-slot serve.

**B-P03 was reverted the same day (owner: "revert"), and restored once the cause was found.**

- **What was seen:** `TestSpecNgram_copyOnStepVerify` (real checkpoint, `GOINFER_METAL_MC3=1`, not in CI) diverged at
  token 105 of 192, about 1,077 keys, with the floor at 1024. It passed at 1100 and 1536.
- **The cause was the test, not spec and not the floor.**
  - The test ran plain decode cold, then spec. Spec's prompt reused 971 of the cache's positions and re-decoded the
    last one, position 971. The cold arm had computed that position in the f16 prefill, which is not bit-identical to
    decode.
  - So the two arms started from different K/V. That difference is at position 971, in every layer, at both floors.
  - The floor only decides whether the resulting rounding difference flips an argmax inside 192 tokens.
- **Measured with a scratch probe, 2026-10-03.** Three runs on one load: plain cold, plain warm (also reusing 971),
  then spec.

  | floor | warm plain ÷ cold plain | spec ÷ warm plain |
  |---|---|---|
  | 1024 | first different token 105; K/V first differs at position 971 | **0 different tokens, 0 different K/V elements** |
  | 1536 | tokens equal; K/V first differs at position 971 | **0 different tokens, 0 different K/V elements** |

  Plain decode against itself diverges exactly where spec seemed to.
- **Retracted:** the revert's diagnosis that "plain greedy decode's device-argmax path disagrees with the logits path"
  was wrong. Nothing on that path differs.
- **The fixed test:** it now runs a cold warm-up, then compares two arms that each reuse the same prefix. It checks
  that both reused the same count, and it asserts on the K/V spec leaves as well as the tokens.
- **The fixed test can go red:** a mutation raising the step's floor by 64 keys turned it red, on the K/V assertion
  alone (1,817,828 elements). The tokens still matched, so the old token-only check would have passed.
- **Floor 1024, with the fixed test:** green, along with `TestSpecNgram_multiTurnMatchesPlain`,
  `TestMC3Verify_sameSlotRowsBitIdentical` (straddling 1024), `TestMetalSnapshotGolden` and `TestAttnFABlkSelection`.
  The whole metal suite passes 229, with one failure: the pre-existing `TestPrefillParityMoEGatedShared`, below.
- **The snapshot golden** is back at the 1024 straddle (900 / 1022 / 1023 / 1100), as the grade baked it.
- **What it shows about cold against warm:** a reused prompt prefix on Metal re-decodes its last position, so a warm
  request can differ from a cold one with the same prompt. That is a property of prefix reuse, not of spec or the
  floor, and it was there before B-P03.

**The chains in serve (owner: "take on the serve-chain fix"), built 2026-10-03.**
- **What it does:** a generation decoding alone on the MC3 batcher runs the greedy or sampled chain, with the resident
  held across its tokens (`decoder/mc3_batch.go` `holdSolo`).
- **When it yields:** the first token anyone else wants the resident. That covers another holder's claim, an
  exclusive section (a newcomer's prefill), another decoder, or a consumer that does not take a token within 2 ms
  (`mc3HoldSendGrace`). The chain stops while the resident is still held, then the token takes the batcher as before.
- **Why yielding is exact:** the chain covers no recurrent model (`greedyChainWhyNot`), so recomputing the position the
  stopped chain had queued rewrites the same KV.
- **Gates (`metal/mc3_chain_test.go`, the committed tiny fixture, in every `go test`):**
  - Alone: 96 of 96 tokens ran held, greedy and at T = 0.8, identical to the unbatched decode.
  - A newcomer mid-generation: A chained, then batched 64 steps with B, both identical to alone.
  - A stalled consumer: the newcomer is still served, and the stalled A finishes identically.
  - Each gate fails under its own mutation: never hold, a hold never released, no send grace.
- **On the 1.5B:** the real-checkpoint MC3 identity gates (4 conversations × 3 turns, greedy and sampled) and the spec
  gates pass.
- **Not yet measured:** the served speed. Its night A/B is pre-registered below (Phase 3, "The serve chain: served grade").

**Found while running the metal suite (not tonight's change): `TestPrefillParityMoEGatedShared` fails on main.** Its
fixture `testdata/tiny-qwen2-moe` is a per-machine download, not committed. The local copy had no config until
2026-10-03, so the test skipped. Now it runs, and `BuildResident` declines with `W4A8 concat needs K%32==0 (group=32),
got K=44` (audit M-10). It fails the same on untouched HEAD, so it is pre-existing. The fixture's shared-expert width is
the cause; the test has never passed on this box.

### A-P02: pre-registration (written 2026-10-03, before any graded run; owner: "1 then 2", A-P02 then D-G01 → D-B01)

**What it decides.** `metalFastPrefillFloor` is 64 because R3 gated K = 64 and 128 and no cell below
(`docs/measurements/metal-prefill-floor-2026-09-20.md`). Below it, a resident with the batched step runs the prompt as
exact decode rows (E-P01), and one without it runs the sequential loop. A-P02 asks whether the pass can take prompts of
16 to 63 tokens. E-P01 already serves the default 2-slot serve on the Qwen2.5 dense models, so A-P02 reaches single-slot
loads (the CLI) and the families MC3 excludes (QK-norm, windows, sandwich norms, adapters, MoE). For residents with the
step it also decides where the pass takes over from the step, which T1.10 timed before A-P01 made the C ≤ 32 pass 1.8×
faster.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-ap02-grade.sh` on the night queue, from tagged test binaries pinned at the commit the script names. (1) The 1.5B's CPU f32 references, set A, at K = 16, 32, 48 and 64, regenerated together (`TestPrefillGateReference`, `GOINFER_CPU_REF_MODELS=S`); the R3 K = 64 files are set aside first. (2) The §3.2 pooled fidelity gate on the 1.5B, set A (`TestPrefillGateVsReference/S`, confirmation cells off), once per pool below and once per single cell. (3) `TestAuditAP02_shortPromptTiming` on the 1.5B and the 7B: a fresh K-token prompt three ways (sequential loop, the step in 8-row pieces, the pass with its floor off), K = 16, 32, 48, 64, 7 reps, arms rotated rep by rep, wall time. Estimate about 20 minutes; queued at 30. |
| Precondition | All 40 reference files present, and no cell VOID on the reference identity check. Otherwise nothing is graded. |
| Candidates | F ∈ {16, 32, 48}. |
| Fidelity | F is fidelity-eligible iff the pool P(F) = {K ∈ {16, 32, 48, 64} : K ≥ F} **SHIPS** (critA ∧ critB ∧ critC). The single cell K = F is reported, not deciding (R3's corroboration). |
| Speed, residents without the step | F is speed-eligible iff, on the **1.5B**, sequential ÷ pass at K = F has a median ≥ **1.5** over 7 paired reps, with ≥ 6 of 7 reps above 1.5 (the audit's kill line, read in-process). |
| Rule: the floor | **The floor moves to the smallest F that is both fidelity- and speed-eligible.** None: **killed**, the floor stays 64. **Parked** (to the owner): the smallest fidelity-eligible F reads a speed median in [1.3, 1.5). |
| Rule: the step bound | A resident with the step keeps the step for prompts below S\*, the smallest K in {16, 32, 48, 64} with K ≥ the new floor at which step ÷ pass on the 1.5B has a median > 1.0 with ≥ 6 of 7 reps above 1 (the pass faster). The step is exact and the pass is not, so the step keeps a prompt unless the pass is measured faster there. If the floor does not move, nothing changes. |
| Reported | The 7B's timing; every single-cell fidelity verdict; the per-cell KL table. Not measured: a served TTFT, which this grade does not claim. Unchanged: the decoder's 8-token threshold (A-P05). |
| What ships | By day, after the grade: the floor constant, the step bound in `promptStepOK`, the routing tests, `PrefillPath`'s banner text, and the docs that quote 64. |

### A-P02: SHIPPED in part, graded 2026-10-04 (the floor moves to 16; the step bound is held back)

Run: the Mac's night queue, 2026-10-03 22:45–22:57 PDT, `run-ap02-grade.sh` from the tagged binaries pinned at `f81f3a18`
(logs `~/goinfer-logs/metal-audit-2026-10/ap02/`). Preconditions held: 40 of 40 reference files, no cell VOID.

| | Reading | Verdict |
|---|---|---|
| Fidelity, pooled (1.5B, set A) | {16, 32, 48, 64}: critA 21 / 25 hard flips (exact / fast), critB 92.11% / 92.07%, critC mean KL 0.0433 / 0.0406, fast lower on 27 of 40 prompts. {32, 48, 64} and {48, 64} also SHIP | **F = 16 fidelity-eligible** |
| Fidelity, single cells | K = 16, 32 and 48 each SHIP (KL 0.0484 / 0.0444, 0.0356 / 0.0327, 0.0428 / 0.0397) | reported |
| Speed, sequential ÷ pass (1.5B) | K = 16 **3.480** (7 of 7 reps above 1.5); 32 6.83, 48 6.48, 64 8.56 | **F = 16 speed-eligible** |
| Step bound, step ÷ pass (1.5B) | K = 16 **0.755** (0 of 7 reps above 1: the step faster); K = 32 **1.485** (7 of 7) | **S\* = 32** by the rule |
| Reported, the 7B | sequential ÷ pass 2.52 / 4.98 / 4.77 / 6.31; step ÷ pass 0.627 / 1.243 / 1.191 / 1.577 at 16 / 32 / 48 / 64 | same crossover |

**Shipped:** `metalFastPrefillFloor` is **16**. A resident without the batched step (a single-slot load such as the CLI, a
family MC3 excludes) takes the pass from 16 tokens instead of 64; `--exact-prefill`'s help, `TestPrefillFloor` and
E-P01's routing test follow.

**Held back: the step bound stays at 64**, not the graded 32. Moving it surfaced an OPEN defect: on an MC3 resident, a
newcomer whose prompt takes the pass makes both generations diverge from their alone runs some tens of tokens later.
- Seen first as `TestMC3Chain_newcomerJoinsAndBothMatchAlone` and `_stalledConsumerDoesNotStarveANewcomer` going red when
  their 30–48-token prompts moved from the step to the pass.
- Reachable on `main` before this change for prompts of 64 tokens or more: the same tests with 100/80 and 90/70-token
  prompts are red at the old floor, at `0eb53e90` (before the serve chain), and with the chain off. A first differs at
  token 42 of 160 (B joined at 24), B at 47 of 64.
- Not the slot (the same prompt through the pass and 60 decode steps on slot 0 and slot 1 match bit for bit), not chunking
  (the fixture sets no chunk size), not the executor's slot switch (`useKVSlot` stops it).
- Kept as an opt-in repro, `metal/mc3_pass_identity_repro_test.go` (`GOINFER_MC3_PASS_REPRO=1`, red until fixed).
- A 32 bound would carry the defect to prompts of 32–63 tokens on serve's default 2-slot Metal resident; at 64 nothing
  changes for those residents. Lower `metalStepPrefillCeiling` to 32 once the defect is fixed.

**Root-caused and fixed the same morning (owner: "we need to root-cause and fix this"): not an MC3 defect, a warm
prompt taking a different route from the cold one.**
- **Below the decoder it never happened.** Driving the resident directly (A prefilled by the pass on slot 0 and decoded
  alone 24 tokens, B prefilled by the pass on slot 1, then batched steps), every step's logits equal both solo runs'.
- **The tests compared warm against cold.** Their alone runs ran first on the same model, so the joint run of A reused
  99 of its 100 prompt positions. On a fresh model (nothing to reuse) A and B both match their alone runs exactly.
- **The mechanism:** the decoder offered `PrefillLast` only suffixes of 8 tokens or more, so a repeated prompt (a 1-token
  suffix) ran its last position through decode, while the cold run had computed it inside the f16 pass. Below the
  floor both are decode bits, which is why short prompts never showed it. It was the B-P03 trap (cold against warm,
  `TestSpecNgram_copyOnStepVerify`) again, met from the batching side. The "held back" reasoning above rested on reading
  it as a batching defect, which it was not.
- **The fix** (`decoder.PrefillTailExact`): a backend whose short continuation reproduces its cold pass declares it, and
  `residentPrefillSeed` then offers a reused prompt's suffix of any length whenever the whole prompt reaches 8 tokens.
  The backend routes on the whole prompt's length, so warm and cold take the same route. Metal declares it for the
  dense and Gated-DeltaNet passes; MoE stays off (its expert grouping at one row is unproven); CUDA and WebGPU are
  unchanged. A 1-token suffix whose prompt ends in the step's range declines to the sequential loop, which computes
  the step's bits (the step needs two rows).
- **Gates:**
  - `TestPrefillLast_tailContinuationMatchesCold`: a 1, 2, 7 or 8-token pass continuing a cached pass prefix equals the
    cold pass in every logit; the decode continuation differs in all of them.
  - `TestGenerate_warmRepeatMatchesCold`: 1 and 2 slots, prompts of 12, 40 and 100 tokens, a warm repeat (n−1 positions
    reused, asserted) emits the cold run's 96 tokens. With the decoder change reverted it fails exactly the pass cases
    (1 slot at 40 and 100, 2 slots at 100; at token 35, 60 and 60).
  - `TestMC3Chain_newcomerJoinsLongPrompts` and `_stalledConsumerLongPrompts`: the former repro, now default-run and
    green.
- Forward goldens re-proven for the `decoder/model.go` change (41 passed, 23 skipped, 0 failed) and the parity hashes
  refreshed; metal suite 237 passed, 1 failed (the pre-existing `TestPrefillParityMoEGatedShared`).
- **The step bound, released to the graded 32 the same morning:** the reason it was held back is gone. A resident with
  the batched step takes the pass from 32 tokens and keeps the exact step below; E-P01's routing test follows, and
  `TestGenerate_warmRepeatMatchesCold`'s 2-slot 40-token case now covers the pass on a 2-slot resident.

`TestMC3Chain_aloneMatchesUnbatchedAndRunsTheChain` now uses a 12-token prompt: at 16 and over its unbatched (single-slot)
model takes the pass and its MC3 model the step, which differ by design. Metal suite: 232 passed, 1 failed (the pre-existing
`TestPrefillParityMoEGatedShared`).

### D-G01: pre-registration (written 2026-10-03, before any graded run; owner: "1 then 2")

**What it gates.** Expert-major MoE prefill (`metal/prefill.go`, the default whenever the batched lane runs) has only
a tiny-fixture cosine bar. It differs from the sequential path in three counted ways (audit D-G01): f16 scatter-adds,
experts applied in id order, and a router computed from f16 activations, which can flip a near-tied expert. This gate
compares it with the f16 lane's own MoE baseline on real weights, and it must catch three planted defects first.

**Fixture.** The first 4 of Qwen1.5-MoE-A2.7B's 24 layers, bytes unchanged (`scripts/slice_checkpoint_layers.py`,
`~/models/qwen15-moe-a27b-l4slice`, sha256 `7e48d607…`): its real router, 60 experts, top-4 routing and gated shared
expert. The whole model does not fit this Mac. A slice is a fidelity fixture: no output or speed of it is the model's.

| | |
|---|---|
| Instrument | `TestDG01_expertMajorMoEPrefill` (`metal/dg01_moe_gate_test.go`), one KV slot, int4, 10 set-A prompts at M = 64 and 512. Three arms per prompt: **seq** (the sequential decode loop, routing captured per token and layer), **row** (the batched pass with MoE row by row through decode's MoE kernels: the f16 lane's own baseline), **major** (the batched pass, expert-major). Per arm: each layer's routing, the K/V of every position at every layer, and the last position's logits. Night queue. |
| Precondition | Layer 0's K/V is equal between row and major (no MoE precedes it), and seq's routing capture holds k distinct valid experts per token. Otherwise the instrument is broken and nothing is graded. |
| Readings, per M, pooled over prompts | **Flips:** (token, layer) pairs whose selected expert set differs from seq's. **K/V:** per layer ≥ 1, relative L2 of the arm's K and V against seq's over all positions. **KL:** mean KL(seq ‖ arm) of the last position's distribution. |
| Gate | Major passes iff at both M: flips(major) ≤ flips(row) + 2√max(flips(row), 1); K/V(major) ≤ 1.25 × K/V(row) at every layer ≥ 1; KL(major) ≤ 1.25 × KL(row). |
| Mutations | Each must FAIL the gate, or the gate is not evidence: `topk-renorm` (scatter weights renormalised), `rank-swap` (each token's first two experts swapped with their weights left in place: route order), `shared-off` (shared expert skipped). Run first; a mutation that passes stops the grade. |
| Rule | **Passes:** expert-major stays the default, and this test is its gate of record. **Fails** (any criterion beyond 1.5× or the flips bound): **killed**, the default goes back to row by row (`GOINFER_MOE_EXPERT_MAJOR` default off) until the defect is found. A ratio in (1.25, 1.5] with the flips bound met: **parked**, to the owner. |
| Reported | Every per-layer ratio, per M; the flips as fractions; seq's own top-1 agreement with row and major on the last position. Not covered: sigmoid and group-limited routing (no such real MoE fits here), multi-model. |

### D-G01: PASSES, graded 2026-10-04 (expert-major MoE prefill stays the default; this is its gate of record)

Run: the Mac's night queue, 2026-10-03 23:01–23:05 PDT, `run-dg01-grade.sh` from the tagged binary pinned at `ec3cafd1`, on
the Qwen1.5-MoE 4-layer slice (`model.safetensors` sha256 `7e48d607…`), 10 set-A prompts at M = 64 and 512 (logs
`~/goinfer-logs/metal-audit-2026-10/dg01/`).

**The mutations, run first: all three FAIL**, so the gate is evidence. K/V ratio against the row baseline and KL ratio, at
M = 64 / 512:

| mutation | K/V, worst layer | KL | flips over the bound |
|---|---|---|---|
| `topk-renorm` (scatter weights) | 12.7x / 13.8x | 60.6x / 146.4x | yes / yes |
| `rank-swap` (route order) | 3.72x / 3.67x | 16.1x / 17.8x | yes / yes |
| `shared-off` | 16.7x / 18.0x | 90.6x / 149.0x | yes / yes |

**The gate: PASSES at both M.**

| | M = 64 | M = 512 |
|---|---|---|
| flips against seq (row / major) | 535 / **507** of 2,560 (20.9% / 19.8%); bound 581 | 4,629 / **4,476** of 20,480 (22.6% / 21.9%); bound 4,765 |
| K/V relative L2, major ÷ row, layers 1–3 | 1.011 / 1.003 / 0.985 | 1.017 / 1.001 / 0.984 |
| last-position KL, major ÷ row | 0.931 | **1.217** |
| top-1 against seq (row / major) | 4 / 5 of 10 | 9 / 10 of 10 |

- **Expert-major is as close to the sequential path as the f16 lane's own MoE baseline:** it flips fewer expert sets and
  its K/V is level with the row arm's.
- **The closest reading is M = 512's KL ratio, 1.217 against the 1.25 bar.** It passes, with a margin of under 3%, on
  ten prompts.
- **Both arms flip about a fifth of the (token, layer) expert sets against decode.** That is the f16 lane, not expert-major:
  top-4 of 60 is full of near-ties. It is what the "flips" bound is measured against, not a defect.
- **Not covered:** sigmoid and group-limited routing (no such MoE fits this Mac), and an accumulation-order change of a
  few ulps (the gate reads gross defects, as the mutations show).

So `GOINFER_MOE_EXPERT_MAJOR` stays default on, and `TestDG01_expertMajorMoEPrefill` is the gate the MoE prefill builds
(D-B02, D-P03, D-P01) grade against.

### D-B01: built, off by default; pre-registration (written 2026-10-03, before any graded run; owner: "go on D-B01")

**What is built.** A Gated-DeltaNet hybrid (Qwen3.5) can take the batched prefill pass behind `dnetPrefillOn`
(`metal/prefill_deltanet.go`), off until this grade.
- **DeltaNet layers:** the qkv, z and out projections run as prefill GEMMs over all M rows, and in_proj_b/a as a rows
  kernel, on f16 activations like every batched layer. The mixer runs as one dispatch per stage over the M rows, 13
  dispatches per layer for the whole prompt against 12 per token sequentially.
- **The mixer kernels are decode's, copied verbatim** with only row addressing added, and a token loop where the stage is
  a recurrence. `TestDeltaNetSeqKernels_matchDecodeBitwise` (default-run): every stage, the conv window and the state
  after the last row equal decode's kernels run M times, bit for bit, at M = 1, 7, 33 from a non-zero state. A one-op
  reorder in the recurrence turns it red (258 outputs, 16,721 state values).
- **Gated attention layers:** q_proj's [query ‖ gate] split, K‖V into qkv, ctx × sigmoid(gate) before o_proj; the pass
  reads its attention geometry from the first attention layer (`prefillGeom`), not layer 0. Head dim 256 runs the exact
  prefill attention kernel.
- **Chunked prefill:** `TestDB01_chunkedPrefillMatchesWhole`, 64 + 32 and 40 + 56 against one 96-token pass: 0 of
  248,320 logits and 0 window/state/K/V words differ. A window that does not carry turns it red.
- **Against the sequential path** (`TestDB01_prefillAgreesWithSequential`, the 0.8B, a sanity bar, not the grade): logits
  cosine 0.9957 / 0.9974 at M = 64 / 256, same top-1; per-layer K/V and state 5–10% relative L2 from the second layer
  on. Against a CPU f32 reference both sit equally far (KL 0.139 / 0.120 at M = 64, 0.122 / 0.122 at 256, pass /
  sequential): the gap is the activation lane, decode's per-row int8 against the pass's f16. Skipping the attention
  gate or misrouting the conv window reads cosine below 0.
- Exploratory, one rep, not a result: about 8× sequential ÷ pass at K = 128 and 512 on the 0.8B.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-db01-grade.sh` on the night queue, from tagged test binaries pinned at the commit it names. (1) The kernel and chunk preconditions. (2) CPU f32 references for Qwen3.5-0.8B, set A, K = 256, 512, 1024 (`TestPrefillGateReference`, cell Q35). (3) The §3.2 pooled fidelity gate on Q35 (`TestPrefillGateVsReference/Q35`, decision cells 256/512/1024). (4) `TestAuditDB01_prefillTiming` on the 0.8B at K = 128, 512, 2048, 7 reps; then the 9B at K = 128, 512, 5 reps, if the fit guard admits it resident. Estimate about 25 minutes; queued at 40. |
| Precondition | `TestDeltaNetSeqKernels_matchDecodeBitwise` and `TestDB01_chunkedPrefillMatchesWhole` pass at the pinned build, all 30 reference files present, no cell VOID. Otherwise nothing is graded. |
| Fidelity | The pooled gate **SHIPS** on Q35 (critA ∧ critB ∧ critC). |
| Graded speed | Sequential ÷ pass on the **0.8B at K = 512**, paired per rep, median of 7. |
| Rule | **Ships** (`dnetPrefillOn` on) if fidelity ships and speed reads ≥ 1.10 with ≥ 6 of 7 reps above 1.10 (the owner's ship bar, 2026-09-26). **Parked** to the owner at 1.02–1.10. **Killed** below 1.02 or if fidelity fails: the switch stays off. |
| Thesis (reported, not the ship rule) | The audit projected 3–8× and set a kill line of 2× at K = 512 on the smallest resident Qwen3.5. Reported against that as its own verdict. |
| Reported | K = 128 and 2048 on the 0.8B; the 9B (or why it did not load); per-cell fidelity. |
| What ships | `dnetPrefillOn = true`, the `PrefillPath` banner, the CHANGELOG, and the docs that say hybrids prefill sequentially on Metal. |

**Amendment (2026-10-04, before any graded run): the first run graded nothing, and two defects are fixed.**
- **The night of 2026-10-03 stopped at the reference step** (`run-db01-grade.sh`, binaries at `d07e54a8`): the
  reference key hashed the checkpoint as one file, and Qwen3.5-0.8B is a safetensors directory ("is a directory").
  Fixed: a directory hashes file by file (`checkpointDirSHA256`; goinfer's `.giw` sidecars and hidden files are left
  out), `TestCheckpointSHA256_directory`.
- **A production defect the gate's plumbing then found** (an exploratory smoke on a K = 8 cell, which is not a registered
  cell): `metalResident.PrefillLast` from position 0 did not reset the DeltaNet window and state, as `Forward(pos 0)`
  does. The pass continued the previous sequence's state, so the gate's pass arm, run after its sequential arm, read
  115 hard flips to 15. With `dnetPrefillOn` on, every fresh prompt in serve would have done the same. Fixed:
  `PrefillLast` resets at position 0, and `TestDB01_prefillFromZeroResetsState` checks it (0 of 248,320 logits differ;
  without the reset all of them do). My earlier tests had reset the state by hand, which hid it. The same K = 8 smoke then
  read 16 hard flips to 15 and agreement 87.8% to 87.7%: plumbing only, not a result.
- **Unchanged:** the instrument's cells, the rule and the bars. Added to the preconditions:
  `TestDB01_prefillFromZeroResetsState`. The job re-runs from binaries pinned at the commit that carries these fixes.

### E-P07: built, ON (bit-identical); its served confirmation pre-registered (written 2026-10-04, before any graded run; owner: "continue on audit-metal")

**What is built.** Qwen3-family dense models (per-head QK-norm) are in MC3: the batched step runs decode's `qk_norm`
over every row between the qkv projection and RoPE (`mc3_qk_norm_rows`, derived from `qk_norm`'s own source in
`batch_rows.go` like the other rows kernels, at decode's threadgroup width). Olmo's whole-vector QK-norm stays out.
So Qwen3 on a multi-slot Metal resident now gets MC3 batching, the step-kernel verify for `--spec ngram`, and E-P01's
short-prompt route.
- **Identity on the real Qwen3-0.6B** (`GOINFER_METAL_MC3=1`, the suite unchanged): `TestMC3Step_bitIdentical`,
  `_bitIdenticalDeep` (across the attention floor), `_rowsPathBitIdentical` (per-row and fragment forms, B = 2–4),
  `_drawsMatchForwardSample` and `TestMC3Verify_sameSlotRowsBitIdentical`: **0 differing values**;
  `TestMC5_prefillChunkInvariance`: 0 at every chunk size.
- **Default-run:** a generated qwen3 twin of the MC3 fixture backs `TestMC3Step_qwen3BitIdentical` (0 differing).
- **Mutation:** dropping the step's QK-norm dispatch fails every logit of every step on the real model and 983,040 on
  the fixture.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-ep07-w7.sh` on the night queue: MC3's W7 harness (`scripts/bench_w7_plain.py`, the 7B record's shape) on Qwen3-0.6B from `~/models`, serve defaults (`-max-concurrent` 4, `-kv-sessions` 4), 6 turns × 128 greedy tokens per client, `--fixed-nonce`, a fresh server per cell. **old** = serve-metal at `d50dbbca` (Qwen3 one generation at a time), **new** = at the E-P07 commit. 4, then 1, then 2 clients, old/new × 3 pairs in the order old new new old old new. Graded by MC3's own `gates.py`. Estimate about 15 minutes; queued at 30. |
| Precondition | Every new server logs the batched-concurrency line and every old one the one-at-a-time line; otherwise the arms are not what they claim. |
| Hard gates | (1) **identity**: every turn's `content_sha` equal across old and new, every cell. (2) reuse equal. (4) 4-client p99 turn new ÷ old ≤ 1.0. Any failure: **killed**, E-P07 reverted. |
| Graded | 4-client aggregate throughput new ÷ old, median of the 3 pairs. |
| Rule | **Stays on** at ≥ 1.10 with the hard gates held (your ship bar). **Parked** (to you) at 1.02–1.10. **Killed** (reverted) below 1.02. |
| Reported | 1- and 2-client cells (a lone generation should be unchanged: new ÷ old within 0.98–1.02 at 1 client is expected, not graded); MC3's own 4-client record on Qwen2.5 for context (1.785× on the 7B). |

**Amendment (2026-10-04, before any run):** "serve defaults" would give the new arm 2 KV slots (Metal's default since
E-P09), so 4 clients could not batch 4 wide. Both arms run with `-kv-sessions 4` passed explicitly, as the 7B record's
defaults then were. A smoke of both pinned binaries on Qwen3-0.6B: old logs "one generation at a time", new "… generations
at once … decode tokens batched". Nothing else changes.

### D-P04: SHIPPED (bit-identical; written 2026-10-04)

**The probe** (the audit's own: kill if the (1,1) route dispatch is under 5 us per layer). `BenchmarkMoERoute`
(`metal/moe_route_bench_test.go`) times 64 route dispatches in one serial encoder, net of 64 (1,1) `copy_u32` dispatches,
best of 200, on the M1 Pro by day (a kernel microbenchmark, so direction, not size):

| Shape | one thread, net per dispatch |
|---|---|
| Qwen1.5-MoE, 60 experts, top 4 | 67.8 us |
| OLMoE, 64, top 8 | 94.8 us |
| Qwen3-MoE / Gemma 4, 128, top 8, renormalised | 200.6 us |
| DeepSeek-V3, 256, top 8, sigmoid, bias, 8 groups keep 4 | 429.8 us |
| gpt-oss-20b (`route_gptoss`), 32, top 4 | 34.9 us |

7 to 86 times the kill line, so it went ahead. The audit's 4-9% projection came from "~10% of a ~5 ms token"; per
layer the measured cost is larger than that basis implied at 128 experts and up.

**What is built.** `moe_route_sg` and `route_gptoss_sg` (`metal/moe.go`), one simdgroup dispatched (32,32), replace the
one-thread kernels in resident MoE decode (`encodeMoERoute`, and Gemma 4's MoE router). Lane l owns experts l, l+32, ….
Elementwise work runs across lanes with the same expression per element. The softmax sum and the weight sum keep the
serial order, and lane 0 does the writes, the renormalisation and the scale. Top-k is a per-lane strict-`>` scan, then
`simd_max` of the value and `simd_min` of the index among the lanes holding it: the serial scan's lowest index on ties.
Group limiting scores whole groups per lane, because a group's top-2 sum does not depend on order. The one-thread kernels
stay in the library as the reference. Expert-major prefill's `moe_route_batch` keeps its one-thread body (it is once per
layer per pass, with the rows in parallel).

**Identity.**
- `TestMoERouteSG_bitIdentical` (default-run): 420 cases, 250 of them with tied scores. Indices and weight bits are equal to
  the one-thread kernels across softmax and sigmoid scoring, renormalisation, routed scale, bias, and group limiting. That
  includes a group size that does not divide nE, more picks than unmasked experts, logits on a 0.25 grid, all-equal
  logits, -inf entries, and gpt-oss.
- **Mutation:** a `simd_max` tie-break fails 188 cases; a reversed softmax sum fails 75.
- `TestDP04_routeSGDecode` (`GOINFER_DP04=1`, the Qwen1.5-MoE 4-layer slice): every logit of 48 decode tokens is
  bit-equal between the arms.

**Speed (in-process, whole token, by day, TE5(b)).** Same test: token GPU time, one-thread against one-simdgroup router,
arms alternated rep by rep from the same positions, 7 reps of 48 tokens.
- **serial/sg median 1.060** (per rep 1.070 1.062 1.066 1.060 1.058 1.060 1.054, all 7 above 1). Saved 68.1 us per MoE
  layer, which equals the microbenchmark's 67.8.
- The slice has 4 of the model's 24 layers and the full LM head, so its token ratio understates the full model's.
- The full model is projected at 24 × 68 us ≈ 1.6 ms per token. That is a projection, not a measurement: the full
  checkpoint is not resident on this Mac.
- The direction is resolved and the change is bit-identical, so it ships by day under TE5(b). No served number is
  claimed.

**Not measured here:** a full resident MoE token (none fits this Mac resident besides the slice), paged MoE (M26 and M35,
where the per-layer saving is the same but the token is 128-167 ms), and gpt-oss end to end (the kernel gate covers it).

### D-P03: PARKED, off by default (bit-identical; written 2026-10-04)

**What is built.** Resident MoE decode can run the k selected experts in four dispatches instead of 3k:
- k-slot gate|up;
- SwiGLU+quant over k rows (MC3's `mc3_swiglu_quant_rows`);
- k-slot down, storing each slot's unweighted sum;
- `moe_combine_k`, which adds `wgt[j]*acc*asc[0]` into the residual in slot order. That is `gemv_w4a8_moe_wacc`'s
  epilogue statement, verbatim.

The two GEMVs are derived from production source in `batch_rows.go` the way the MC3 rows kernels are, so their bodies
cannot drift. gpt-oss, the paged path and Gemma 4 keep the per-slot loop. `moeKSlotsOn` selects the path; it is
**off**.

**Identity.**
- `TestMoEKSlots_bitIdentical` (default-run): a tiny qwen2_moe with 8 distinct experts, top 3 and a live shared expert.
  16 positions with 15 distinct routed sets: logits bit-equal between the arms.
- **Mutation:** a combine in reverse slot order fails (1 ulp at position 5), and every slot reading slot 0 fails.
- `TestDP03_kSlotsDecode` (`GOINFER_DP04=1`, the Qwen1.5-MoE 4-layer slice, k = 4): 48 tokens' logits bit-equal.

**Speed (in-process, whole token, by day).** Same test and shape as D-P04's: **loop/k-slot median 1.019** (per rep
1.023 1.014 1.007 1.024 1.019 1.015 1.020, all 7 above 1), 21.5 us per MoE layer.
- That is 8 fewer dispatches per layer at about 2.7 us each, against the audit's projected 3.8 us launch floor.
- The audit's kill line was 2% of the layer. Taking the slice's LM head at about 1.3 ms (155 MB of int4 at 110-130 GB/s,
  an estimate) leaves about 0.8 ms per layer, so the saving is about 2.7% of a layer. That clears the kill line, but
  only just.
- The full 24-layer model projects to about 2.5% of a token.

**Verdict: parked.** The direction is resolved and the output is bit-identical, but low single digits is the owner's
park zone for speed ("park stuff if it's a couple percent above current code", 2026-09-26). The code and both tests
stay, off by default.
- **What could reopen it:** a k = 8 model (OLMoE, Qwen3-MoE, DeepSeek), where 20 dispatches go instead of 8. None fits
  resident on this Mac, so none was measured.

### D-B02: T1.12 probed (written 2026-10-04); and a pass-scratch fix it found, SHIPPED

**T1.12** (`TestDB02_expertMajorProbe`, `GOINFER_DP04=1`; exploratory sizing, not a graded result). The setup:
- The Qwen1.5-MoE 4-layer slice: 60 experts, top 4, a 5632-wide shared expert, fully resident.
- Prose prompts from the prefill gate set, 3 timed passes after a warm one.
- The `db02Trace` hook reports each command buffer's GPU time and each active expert's rows. The host intervals come
  from the hook's own clock.

| M | wall | GPU busy | routing syncs: grouping + unoverlapped encoding | host before the first buffer | expert GEMM rows ÷ routed rows (row tile) |
|---|---|---|---|---|---|
| 64 | 111.2 ms | 105.4 ms | 0.17 + 3.92 ms | 1.48 ms | 7.16× (57 active experts a layer, about 4 rows each) |
| 512 | 253.6 ms | 232.3 ms | 0.53 + 5.63 ms | 14.64 ms | 1.47× |
| 2048 | 731.7 ms | 657.0 ms | 0.73 + 7.49 ms | 77.72 ms | 1.23× |

Each MoE layer runs 286-300 expert dispatches (5 per active expert).

**What this says about D-B02.** The per-layer routing sync that device-side scheduling would remove is small:
- Host grouping, plus the encoding the GPU cannot overlap, is 2.4% of the 512-token pass and 1.1% of the 2048-token one.
- The 300 dispatches cost about 0.8 ms per layer of bubbles at the 2.7 us D-P03 measured.

The bigger D-B02 term is tile padding:
- At M = 512 the expert GEMMs run 1.47× the routed rows, and at these sizes they are compute-bound, not weight-bound.
  The routed FLOPs alone are about 142 GFLOP a layer, against 58 ms of GPU time a layer.
- MLX's bm = 16 tiling, or a device-side scheduler that packs several experts' rows into one tile, is the lever.
- `gemm_w4f16_tile`'s activation staging assumes TM ≥ 32 (`BPT = 4/(128/TM)`), so a 16-row tile is a kernel change
  and not just another instantiation.
- **Next:** the expert GEMMs' share of the busy time, and what a TM = 16 tile buys on the slice, before any
  device-side kernel.

**The host time before the first buffer was not D-B02 at all.** Every pass allocated its f16 scratch by building a
zeroed Go slice and copying it into a new Metal buffer. That was about 110 MB at M = 2048 on the slice (`guF` alone
46 MB). Metal fills a new buffer with zeros itself, so `prefillScratchU16` now allocates with `newBufferWithLength`
and skips the slice.

Checks:
- `TestPrefillScratch_zeroFilled`: a dirtied, released buffer of the same size never comes back non-zero.
- `TestPrefillScratch_identical` (default-run): the tiny dense fixture's logits are bit-equal at 20, 100 and 300
  tokens.
- `TestPrefillScratch_AB` (in-process, arms alternated, logits compared every rep, 7 reps; every rep bit-equal):

| Model | M = 512 | M = 2048 |
|---|---|---|
| 1.5B | **1.012** (6 of 7 reps above 1) | **1.008** (7 of 7 at or above 1.000) |
| Qwen1.5-MoE slice | 1.102 | 1.098 |

Copied ÷ zero-filled, the medians. The slice's 4 layers inflate the scratch's share; a full model sits near the
1.5B's 1%.

It ships: bit-identical, and it removes redundant host work without adding a mechanism. The Metal prefill, DeltaNet,
MoE and MC3 tests pass (76), apart from the long-standing `TestPrefillParityMoEGatedShared` (fails at HEAD too: a K = 44
fixture `int4Concat` refuses).

### D-B02: the 16-row tile, SHIPPED (bit-identical; written 2026-10-04)

**What is built.** `gemm_w4f16_tile` gains a 16-token tile (`gemm_w4f16_m16n32`, `_m16n64`). At TM = 16 the activation
staging gives each row 4 threads, so threads 64-127 stage nothing; the larger tiles' staging is unchanged, because the
guard folds at compile time.

**The selector's default rule changed.** A threadgroup costs the same whatever share of its rows is real, so `gemmTile`
takes a smaller token tile when it pads the pass to strictly fewer rows:
- 32 tokens, up to 128 rows;
- 16 tokens, up to 16 rows, or up to 48 rows when N ≤ 8192;
- a 16-token tile always takes 32 features;
- otherwise A-P01's feature rule.

A-P01's rule stays as the test arm `"a01"`.

**Identity.** `TestGemmTile_bitIdentical` covers both 16-token tiles against `gemm_w4f16_store`: 98,102,016 outputs, 7
shapes, 8-72 rows, 3 epilogues, all equal. `TestDB02_tileAB` compares the last row's logits in every rep of every cell
below, and they are bit-equal throughout.

**Kernel probe** (`TestGemmTile16_probe`, `GOINFER_GEMM16=1`; the raw output is
`docs/measurements/metal-audit-2026-10/db02-tile16-probe-2026-10-04.log`):
- 8 shapes (Qwen1.5-MoE experts, 1.5B, 7B) × 14 row counts from 8 to 256.
- At ≤ 16 rows the 16×32 tile is the fastest in every shape, 1.57-2.06× A-P01's choice.
- 16-token tiles lose from 96 rows, and on the widest gate|up GEMMs at 40-48 rows.

**Whole pass, in-process.** Fresh prompt, `PrefillLast` wall time, A-P01's rule ÷ the new rule, 7 reps alternated, the
batched-pass floor at 0. Cells marked "same tile" pick the same tile under both rules, so they are A/A controls.

| Model | Cell | a01 ÷ new | Reps above 1 |
|---|---|---|---|
| 1.5B | M = 16 | **1.555** (51.7 → 33.0 ms) | 7/7 |
| 1.5B | M = 40 | **1.081** | 7/7 |
| 1.5B | M = 48 | **1.082** | 7/7 |
| 1.5B | M = 72 | **1.108** | 7/7 |
| 1.5B | M = 88 | **1.112** | 7/7 |
| 1.5B | M = 24, 56, 100, 128, 160 (same tile) | 1.000-1.008 | — |
| 1.5B | M = 512 | 1.009 | 6/7 |
| 7B | M = 16 | **1.727** (214 → 124 ms) | 7/7 |
| 7B | M = 40 | **1.051** | 7/7 |
| 7B | M = 48 | **1.052** | 7/7 |
| 7B | M = 72 | **1.070** | 7/7 |
| 7B | M = 24, 100, 512 (same tile) | 0.998-1.003 | — |
| Qwen1.5-MoE slice | M = 64 | **1.532** | 7/7 |
| Qwen1.5-MoE slice | M = 512 | **1.090** | 7/7 |
| Qwen1.5-MoE slice | M = 2048 | **1.013** | 7/7 |

Two rules were tried before this one, recorded so the bounds are not re-derived:
- **Fewest padded rows, no bounds:** the 1.5B's 100-token pass ran **0.847×** (0 of 7) on all-16-row tiles, and M = 160
  ran 0.978×.
- **The 48-row bound without the N condition:** the 7B's 40-token pass ran **0.984×** (0 of 7). Its 37,888-wide
  gate|up already fills the cores at 64.

Each loss was read in a single run and the rule was then narrowed. The 1.5B and 7B rows in the table are from the final
rule's runs: M = 24-48 re-run after the last change, and the other cells unaffected by it.

**It ships** under TE5(b): every changed cell resolves in its own direction (7/7), the output is bit-identical, and
nothing regresses. It is the default in serve. Prompts of 16 tokens on a single-slot resident gain most (the A-P02
floor); MoE prefill gains through its per-expert GEMMs, D-B02's tile padding. Device-side expert scheduling is not
built; the remaining MoE padding at M = 2048 is 1.23× before this change.
- Tests: the Metal prefill, MoE, MC3, MC5, chunked, verify and GEMM tests pass (96), apart from the long-standing
  `TestPrefillParityMoEGatedShared`.

### B-P06: KILLED (written 2026-10-04)

**The probe.** `TestBP06_rowsTG`, at `c3e75327`, then removed with its knob:
- The dense decode sites' R18 rows kernels (qkv, o, gate|up, down) ran at threadgroups of 64, 128, 256 (shipped) and 512
  threads. The kernels already map rows from `threads_per_threadgroup`, so this changes only the dispatch.
- Whole decode tokens from the same positions after a 128-token prefill; 4 arms rotated rep by rep, 7 reps of 16
  tokens; each arm's time is its token GPU-time median.
- In-process, by day. The logits were bit-equal across every arm and token.

The ratio is 256's token time ÷ the arm's; every rep of every arm was below 1:

| Model | 64 threads | 128 threads | 512 threads |
|---|---|---|---|
| 1.5B (10.39 ms token) | 0.828 | 0.972 | 0.890 |
| 7B `.giw` (32.27 ms token) | 0.521 | 0.903 | 0.925 |

**Verdict: killed.** The kill line was below 1.03× on the 1.5B, and every other size is slower on both models. 256
stays.
- **Not built:** the audit's device-read (unstaged) variant at MLX's 64 threads. It is a kernel rewrite, and R18b's
  first cut lost in sequence for exactly an activation re-read.

### E-P02: KILLED by its kill line (written 2026-10-04)

**The probe.** `TestEP02_adjacentRows`, at `fdbf4740`, then removed:
- `mc3_gemv_w4a8_sa_rows_adj4` runs the batched step's per-row GEMV over B rows in one dispatch. Each weight tile's B
  threadgroups are adjacent in launch order (tile tg / B, row tg % B), and each runs `sa_rows_acc` as the per-row
  dispatch does.
- Against production below the fragment's calibrated size: B dispatches of `gemv_w4a8_sa_rows4`.
- 16 reps per command buffer, best of 15, arms alternated; GPU time. Every output float equal, in every cell.

| Shape | B = 2 (per-row → adjacent) | B = 3 | B = 4 |
|---|---|---|---|
| 7B gate\|up (N 37,888, K 3,584) | 0.863 → **0.777 ms**, 1.111× | 1.134× | 1.147× |
| 7B qkv width (N 4,608) | 0.111 → 0.100 ms, 1.110× | 1.153× | 1.178× |
| 1.5B gate\|up (N 17,920, K 1,536) | 1.009× | 1.019× | 1.035× |

**Verdict: killed.** The audit's line was "B = 2 above ~0.65 ms" on the 7B gate|up, and it reads 0.777. The second
threadgroup finds part of the tile in cache, not most of it; the DRAM floor is 0.435 ms.
- What it would buy at B = 2 on the 7B: about 0.086 ms × 28 layers = 2.4 ms of the 58.76 ms step, about 4%. That is in
  the park zone even without the kill line.

**Reopened and SHIPPED, 2026-10-04 (owner: "turn ... E-P02 on").** The kill was a thesis line (an absolute 0.65 ms on one
kernel), not a regression, and the owner's bar is 1.02x on the whole step. Rebuilt for production:
`mc3_gemv_w4a8_sa_rows_adj<R>` and a bias twin `mc3_gemv_w4a8_sa_bias_rows_adj<R>` (R = 2, 4), pipelines beside
`pSARows` / `pSABiasRows`, run from the batched step wherever calibrateRows gives qkv or gate|up to per-row GEMVs at
B >= 2 (`mc3AdjRowsOn`, on).
- **Identity:** `TestMC3Step_rowsPathBitIdentical`'s per-row case (now asserting the fixture's rows kernels exist, so it
  runs the adjacent form) at B = 2, 3, 4: 0 logits differ from single-token decode. Every row reading row 0's activations
  fails it (all 20,480 logits).
- **Whole step, in process, by day (`TestEP02_stepAB`, the 7B `.giw`, depth 128, 7 reps alternated, GPU time):**
  B = 2 per-row 58.046 ms against adjacent 53.927 ms, **1.076x**, 7 of 7 reps above 1 (about twice the 4% estimated
  above). B = 4 reads 0.997 (1 of 7): calibrateRows gives gate|up to the fragment above B = 2 there, so the adjacent form
  does not run, and the cell is a do-nothing control.
- **Left open:** calibrateRows still prices per-row as B separate GEMVs. With the cheaper adjacent form per-row may win
  at B = 3-4 too (the kernel read 1.13-1.15x there); re-pricing it is a separate change.
- **The follow-up, read and PARKED 2026-10-04 (owner: "E-P02 follow-up").** calibrateRows timed the adjacent form
  itself at each B against the fragment; on the 7B it then took qkv and gate|up per-row up to B = 3 (from 2). The whole
  step (`TestEP02_stepAB`, 7 reps alternated, GPU time): B = 2 1.073 (7 of 7, unchanged: adjacent on both pricings),
  **B = 3 1.000 (4 of 7)**, B = 4 0.999 (2 of 7, the fragment on both, a control). The kernel's 1.13x at B = 3 does not
  survive into the step. In the owner's park zone, and it would add calibration time to every load, so the code was not
  kept; the pricing stays B separate GEMVs.
- The 1.5B gains little because its weights are near the SLC's size, as the audit predicted.

### E-P05: PARKED, off by default (bit-identical; written 2026-10-04)

**The ceiling probe** (the audit's: a stub that skips rows 1-7's attention). The 1.5B, 8 rows each on its own slot at
depth 2048: **37.2% of the step** (29.82 → 18.72 ms, 7 reps 0.371-0.374), far over the 5% kill line. The 4096 cell
did not run: filling 8 slots through `ForwardBatch` took 220 s at 2048, and the test hit its timeout. The A/B below
fills by the prefill pass instead.

**What is built.** `batch_rows.go` derives rows forms of decode's flash-attention first pass (`attention_fa`,
`attention_fa_blk<G>` for G = 2-8, `attention_fa_blk64<7>`) and of `attention_fa_combine`. Row j serves batch row
`rowmap[j]`, with its own q, slot K/V, key count, split count and partial region.
- The grid is sized for the step's largest split count; a threadgroup past its own row's count returns. Each remaining
  threadgroup sees the index the row's own dispatch gave it, so each row computes exactly its own pair's bytes.
- `mc3FARowsOn` runs every deep row of a step through one dispatch pair per layer instead of one pair per row.

**Identity.**
- `TestMC3Step_faRowsBitIdenticalDeep` (default-run, the MC3 fixture, with the path on): two rows past the floor and two
  below, 12 steps, every logit equal to production's single-token decode.
- **Mutation:** every row merging row 0's partials fails `TestMC3Step_bitIdenticalDeep`, with every logit differing.
- `TestEP05_deepRowsAB` compares the arms' logits every step: bit-equal in every cell below.

**Speed (in-process, by day).** `TestEP05_deepRowsAB`: each slot filled to depth D by the batched prefill pass, steps
from position D, arms alternated, 7 reps of 12 steps, GPU time per step.

| Model | Rows | Depth | Per-row ÷ multi-row | Reps above 1 |
|---|---|---|---|---|
| 1.5B | 8 | 1100 | **1.057** (25.62 → 24.25 ms) | 7/7 |
| 1.5B | 8 | 2048 | **1.060** (27.40 → 25.86 ms) | 7/7 |
| 1.5B | 8 | 4096 | 1.015 | 7/7 |
| 1.5B | 4 | 2048 | 1.026 | 7/7 |
| 1.5B | 2 | 2048 | 1.016 | 7/7 |
| 7B `.giw` | 4 | 1100 | 1.029 | 7/7 |
| 7B `.giw` | 4 | 2048 | 1.029 | 7/7 |

**Why the gain is a sixth of the ceiling.** The stub removed 7 rows' K/V reads, not their dispatches. Each row reads
its own slot's K/V, and one dispatch cannot share that.
- What one dispatch does remove is the serial per-row launch and the low occupancy of a 32-threadgroup grid. That is
  worth 1.5-6%, and least at depth 4096, where the reads dominate.
- Sharing a K/V stripe is possible only between rows of the same slot (spec verify). That is the audit's second step and
  changes the reduction shape.

**Verdict: parked**, off by default. **Turned ON 2026-10-04 by the owner** ("turn E-P05 on"): bit-identical and faster in
every cell measured (1.016-1.06x), which clears the owner's bar of a clean non-regressing win; `mc3FARowsOn` is true.
- Bit-identical and resolved in every cell (7/7), but serve's default is 2 KV slots (E-P09), where it is 1.6%. Low single
  digits is the owner's park zone, as with D-P03.
- The case for turning it on is many slots (`-kv-sessions 8`) at depths of 1-2k keys, where it is 6%.
- The derived kernels and both tests stay.

### The night of 2026-10-04 (run by day, 09:48-10:47): D-B01 and E-P07 graded

Raw results and provenance: `docs/measurements/metal-audit-2026-10/night-2026-10-04/`. The queue ran in the daytime on
the owner's word, so the timed cells shared the machine with a desktop: the load at start was 0.94-1.73.

**D-B01 — SHIPS under its pre-registered rule.**
- **Preconditions pass:** chunk invariance, reset from zero, and the DeltaNet sequence kernels bitwise.
- **Fidelity:** the §3.2 pooled gate on Q35 (Qwen3.5-0.8B, set A, K = 256/512/1024, 1,920 positions) SHIPS.
  - critA: hard flips 65 fast against 65 exact.
  - critB: agreement 87.92% against 88.12%, d = 52.
  - critC: pooled KL 0.1147 against 0.1191, fast lower on 24 of 30 prompts, no cell over the ceiling.
- **Speed (graded cell, the 0.8B at K = 512):** sequential ÷ pass **8.058**, 7 of 7 reps above 1.10. Also 8.387 at
  K = 128 and 5.576 at K = 2048; the 9B 6.824 at K = 128 and 6.712 at K = 512.
- **Shipped: `dnetPrefillOn` is on** (2026-10-04). The default and tagged Metal suites pass with it on, apart from
  the long-standing `TestPrefillParityMoEGatedShared`; the D-B01 tests now restore the switch instead of forcing it
  off. CHANGELOG entry under Unreleased.

**E-P07 — speed passes, identity (hard gate 1) FAILS; graded as written, decision to the owner.**
- **Speed** (W7 on Qwen3-0.6B, 3 pairs): 4-client aggregate new ÷ old 2.026 / 2.069 / 2.041×, median **2.041×** against
  the 1.10 ship bar. p99 under load 0.484× (bar ≤ 1.0). The 1-client solo guard p50 0.983×, p99 1.018× (bar ≤ 1.05).
  2 clients, reported: 1.28×.
- **Identity fails in 205 turns, for two reasons. Neither is a fault in E-P07's step, which is bit-identical to decode
  (its kernel tests).**
  1. **Old against new differs at turn 1, with one client.** The turn-1 prompt is 29 tokens. On a multi-slot server
     A-P02 sends 16-31-token prompts to the exact batched step when the model can batch, and to the f16 pass when it
     cannot. Before E-P07 Qwen3 could not batch, so old took the f16 pass and new the exact step. Different lanes give
     different bits, so old = new was never a property this change could keep. Every later turn's history follows
     from turn 1.
  2. **New against new differs under concurrency** (new2_2 12 turns, new4_2 8, new4_3 8; none at 1 client). The
     failing turns reuse different amounts of their prompt (prefilled 133 against 111 tokens), because slot assignment
     depends on timing. A different reused prefix leaves a different suffix, and the suffix's length picks the lane
     (exact step below 32, f16 pass from 32). So served output under concurrency is not run-to-run identical whenever
     prompts straddle the lanes.
     - This follows from A-P02's two lanes plus timing-dependent slot reuse. It is not Qwen3-specific: it applies to
       any MC3 family since A-P02's 32-token step ceiling, which postdates the 7B W7 identity record.
- **Rule as written:** any hard-gate failure kills E-P07 and reverts it. Not reverted yet: the failure's mechanism is the
  gate's premise and A-P02's lane split, not the change. The owner decides between:
  - (a) revert as written;
  - (b) re-run the identity gates with both arms on the exact lane (`--exact-prefill`), where old = new and run-to-run
    identity are properties the system should have, keeping the speed gates as graded.
- The run-to-run point (2) stands either way, as a finding about served determinism under concurrency.
- **Owner decision (2026-10-04): (b), re-run.** The amendment is written below before that run.

**E-P07 amendment 2 (written 2026-10-04, before the re-run).** The identity gates re-run with both arms on the exact
lane.
- **Instrument:** `docs/measurements/metal-audit-2026-10/run-ep07-w7-exact.sh`, the same pinned binaries, model, cells
  and order as the first run, with `-exact-prefill` added to both arms' serve arguments. No prompt takes the f16 pass:
  old prefills sequentially, and new prefills sequentially or through the batched step, which is bit-identical to
  decode.
- **Gates:**
  - (1) identity: every turn's `content_sha` equal across every old and new cell at each client count.
  - (2) reuse equal.
  - The precondition is unchanged: 9 one-at-a-time servers and 9 batched.
- **Rule:** both hold → E-P07 stays on, its speed graded from the first run (2.041×, p99 0.484×, solo guard passed).
  Either fails → killed and reverted.
- **Reported:** the exact lane's own speed figures from gates.py, which grade nothing.


### E-P07 amendment 2 graded (2026-10-04, 13:44-14:29; raw: `docs/measurements/metal-audit-2026-10/night-2026-10-04b/ep07-exact/`)

- **Identity (gate 1): PASSES.** On the exact lane every turn's content is equal across all old and new cells at 1, 2
  and 4 clients: 0 content mismatches, against 205 failing turns on the mixed lanes. E-P07 does not change what is
  served.
- **Reuse (gate 2): fails in 27 turns.** The batched server reused more of a turn's prompt than the one-at-a-time
  server (prefilled 133 against 111 at turn 3), and at 4 clients new against new differs too. How much a turn reuses
  depends on which slot it lands on, which depends on timing. It is a cache statistic, not output.
- **Speed on the exact lane, reported:** 4-client aggregate 1.591×, p99 0.761×, solo guard p50 0.987× / p99 1.014×.
- **Owner decision (2026-10-04): E-P07 stays on** ("option 1, keep it on"). The output identity the gate exists for
  holds; the reuse-equality gate assumed a slot-independent reuse the system does not have.

### The serve chain: served grade — pre-registration (written 2026-10-04, before any graded run; owner: "this first")

**What it decides.** The serve chain (`0277f4ed`, "The chains in serve" above) lets a lone generation on the MC3
batcher run the greedy or sampled chain with the resident held. Its identity gates pass; its served speed was never
measured. Before it, the post-merge peer read found the chains bought nothing in serve (1.5B 0.991×), because serve on
Metal always batches. This grade says whether the chain now pays where serve runs it.

**The precondition, checked by day (2026-10-04, untimed).** Real serve on the 1.5B at its defaults (2 KV slots), two
64-token requests per arm, counts from the new shutdown line (`3b9ef839`, `internal/serveapp/main.go`):
- new arm: 128 of 128 decode tokens held, in 2 holds, greedy and at T = 1.0;
- old arm: 0 held, 128 solo batcher runs.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-servechain-grade.sh` on the night queue. Two arms, no peer (TE5(a)), both pinned binaries built at `3b9ef839`: goinfer as committed, and goinfer_old with the hold never taken (`servechain-old-arm.patch` beside the binaries: `if false && …` on the one branch that calls `holdSolo`, the gate's "never hold" mutation). `bench_peer.py` on Metal at serve's defaults, depth 128, `greedy` and `temp1.0_notrunc`, the 0.5B, 1.5B and 7B at int4, 3 runs per cell. Four passes, the engine order alternating (new first, old first, new first, old first). `servechain_cells.py` tabulates the passes and checks the precondition. Estimate about 33 minutes (about 8 per pass, 5 of them the 7B's four cells); queued at 40. |
| Precondition | Every cell of every pass: the new arm ran at least 95% of its decode tokens held, the old arm none, and both report `metal-resident (int4)` (from each cell's serve log). A failure voids the grade, and the cause is found before any re-run. |
| Graded | The **1.5B greedy** cell. G = the median over the four passes of the pass's new ÷ old (each arm's bench_peer mean over its runs). |
| Rule | **Stays on** if G ≥ 1.02 and at least 3 of 4 passes are above 1.00. **Killed** if G < 1.00: the hold is reverted (`holdSolo`, `releaseHold` and the branch in `generateInto`), and the record keeps the numbers. **Otherwise parked to the owner** (1.00 ≤ G < 1.02, or G ≥ 1.02 with fewer than 3 passes above 1.00): the chain's in-process grade (1.085×) and the served number then disagree in size, and the owner decides. |
| Reported | The 1.5B at T = 1.0, and the 0.5B and 7B in both configs, by the same statistic. A 1.5B sampled G below 1.00 goes to the owner, since the sampled chain could be dropped from the hold alone. The in-process grades were 1.085× greedy and 1.066× sampled on the 1.5B; the served number includes HTTP streaming per token, so it can be smaller. |
| Not read here | The peer ratios. If the chain stays on, the benchmarks Metal decode row is refreshed by its own same-session peer read (TE5(a)), queued separately. |

### E-P08: SHIPPED (bit-identical; built and read 2026-10-04, owner: "do E-P08 now")

**What it does.** A greedy generation's batched token came back as its whole logits row: the step copied 151,936 floats
(about 600 KB) a row to the host, applied any softcap or logit scale there, and the decoder took the argmax. The audit's
fix, built: the decoder now sends such a row as a **Greedy draw** (`decoder.ResidentBatchDraw.Greedy`, only to a resident
implementing `decoder.ResidentBatchGreedy`; Metal does), and the step returns its id. Metal (no ForwardArgmax in
production, so this was every greedy batched row) takes it on the GPU when no transform applies, and on the host from
the transformed row when a softcap or logit scale does, which is the id the decoder would have picked either way. The
step's argmax-only rows (the prompt step, spec verify), scanned on the host after the command buffer, take the same GPU
argmax. `argmax_rows_part` (64 partials a row, NaN skipped) then the existing `argmax_finish` take it on the GPU, and the host
reads 4 bytes a row (`mc3DeviceArgmaxOn`, on). The id is `argmaxF32`'s: the first maximum, and 0 for a row that starts with
NaN, which the host checks from element 0.
- **Kernel gate** (`TestArgmaxRowsPart_matchesHost`, default-run): 42 rows at V = 151,936, 152,064, 32,000, 1,000, 257 and
  1: random, ties at the maximum, maximum last, all -inf, all equal, NaN first, NaN scattered; the device equals
  `argmaxF32` on every one. Letting a later equal value win fails it.
- **Step gate** (`TestMC3Step_deviceArgmaxMatchesHost`, the MC3 fixture): 54 rows at B = 2, 3, 4 over 6 steps, argmax-only
  ids and temperature-0 draw rows equal to `argmaxF32` of the same step's full logits rows.
- **Through the decoder:** `TestMC3Chain_newcomerJoinsAndBothMatchAlone` (two greedy generations batched through
  `Generate`, each identical to its alone run) now asserts that the batched steps' greedy rows came back as device ids.
  The decoder suite passes; the parity hashes were refreshed for the `decoder/model.go` edit (forward goldens 41 passed,
  0 failed).
- **Whole step, in process, by day** (`TestEP08_stepWallAB`, the 1.5B, depth 128, 7 reps alternated, wall time, ids equal
  every rep):
  - serve's route (`GOINFER_EP08_ROUTE=serve`: Greedy draws against full rows argmaxed on the host): **1.034 at B = 2,
    1.048 at B = 4, 1.102 at B = 8**, 7 of 7 reps above 1 in each;
  - the argmax-only form: 1.022, 1.036, 1.081.
  - The audit's ceiling was 1.4% at B = 4 and 2.9% at B = 8; the copy and host scan cost more than its S4 slope priced.

### D-B04's routed-expert half: SHIPPED (bit-identical; built and read 2026-10-04, owner: the leftovers)

**What it does.** `gemv_w4a8_moe_rows<R>` and `gemv_w4a8_moe_wacc_rows<R>` (R = 2, 4): the routed-expert GEMVs with
R18's R rows per simdgroup, `sa_rows_acc<R>` over the routed expert's own weight block. `moeExpertRows` picks them at
build for the generic MoE and Gemma 4's MoE, resident and paged (`moeExpertRowsOn`, on), wherever the row counts admit
full threadgroups; gpt-oss and the int8 expert path keep their one-row kernels.
- **The trap, found by the gate:** the accumulate epilogue, written `out += wgt*acc*asc`, differed from the one-row
  kernel's in the last bit. The sums are equal; under fast math the compiler lowered the two differently (an epilogue
  probe on 3 shapes x 3 seeds: the one-row kernel computes `fma(wgt*acc, asc, out)` under fast and under precise math,
  and five other forms differ in 157-822 outputs). The rows kernel now writes that `fma` out.
- **Gate** (`TestMoEExpertRows_bitIdentical`, default-run): the tiny Gemma 4 MoE and the tiny Qwen3.5 MoE, all experts
  resident and paged at 2 slots, every logit of 8 positions equal with and without the rows form, the rows form engaged
  (R > 0). The unpinned epilogue failed it.
- **Speed, in process, by day** (`TestDB04R_expertRowsAB`, the arms' logits bit-identical first, 7 reps alternated):
  - the Qwen1.5-MoE 4-layer slice, resident, GPU time a token: 4.607 -> 4.402 ms, **1.046x**, 7 of 7;
  - M26, paged at 24 slots, under the kill-watch (swap flat): GPU-busy a token 32.9 -> 31.2 ms, **1.052x**, 7 of 7. Its
    wall time a token is dominated by expert staging from disk (86-105 ms a rep) and read 0.970 with 3 of 7 above 1,
    unresolved, which is why the paged arm is graded on GPU-busy time.

### PrefillTailExact for MoE: ON (bit-identical; 2026-10-04, owner: the leftovers)

The warm-against-cold fix (`decoder.PrefillTailExact`, A-P02's record) was declared for the dense and Gated-DeltaNet
passes only: "MoE's expert grouping at one row is unproven". Now proven and declared for every resident
(`PrefillTailExact` returns true): the shapes that could differ decline the pass, so their suffix runs the sequential
loop as before (paged generic MoE, resident Gemma 4 MoE, gpt-oss, int8 KV), and a paged Gemma 4 MoE's layer-major path
is decode's kernels.
- `TestPrefillLast_tailContinuationMatchesCold_MoE`: Mixtral, two Qwen3-MoE shapes and the Qwen3.5 MoE hybrid, tails of
  1, 2, 7, 8, 16 and 33 rows continuing a cached prefix equal the cold pass in every logit; decode of the last position
  differs (the control).
- `TestGenerate_warmRepeatMatchesCold_MoE`: through `Generate`, a 20-token prompt repeated (19 positions reused) emits
  the cold run's 40 greedy tokens on Mixtral and Qwen3-MoE. With MoE excluded again, Mixtral's repeat diverges at token 5;
  Qwen3-MoE does not diverge either way, so Mixtral is the discriminating case. (A 40-token run with 20 tokens did not
  discriminate on either; the test was lengthened until it did.)

### Peer refresh: pre-registration (written 2026-10-04, before it runs; owner: "we should set this up to run tonight, right?")

`docs/benchmarks.md`'s Metal decode row describes a build main no longer runs: the post-merge read's consistency check
failed (the chains never reached serve), the 0.5B's depth cells predate B-P01, and since then the serve chain, B-P02,
D-B04, E-P05/E-P02/E-P08 and native int8 shipped.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-peer-refresh.sh` on the night queue, priority 80 (after tonight's serve-chain grade and M26 prefill grade). One same-session `bench_peer.py` sweep on Metal: goinfer at the pinned rev, mlx-lm 0.31.3, Ollama 0.32.5; greedy at depth 128, 2048 and 3900, and `temp1.0_notrunc` at depth 128; the 0.5B, 1.5B and 7B (no MLX 0.5B checkpoint); 3 runs per cell; the instant idle gate; serve at its defaults (2 KV slots). No old arm (TE5(a): the levers were graded new / old already). Estimate about 45 minutes (the post-merge read's 4 engines took about an hour); queued at 50. |
| Reading | A record, not a gate. Per cell: goinfer ÷ mlx-lm and goinfer ÷ Ollama, each with its runs' spread. They replace the Metal decode row's cells in `docs/benchmarks.md`, with this provenance. |
| Void if | Tonight's serve-chain grade turns the serve chain off (the sweep's goinfer ran with it on), or any cell's goinfer DecodePath is not `metal-resident (int4)`: then the row is not updated from it, and the sweep re-runs on the build that ships. |
| Reported | The 0.5B at 2048 and 3900 against the post-merge read's 0.80x and 0.60x of Ollama (B-P01's block kernel postdates it). |

### Layer-major prefill for the generic paged MoE: SHIPPED (bit-identical; built and read 2026-10-04, owner)

The exact layer-major prefill M26 got (`prefillG4Paged`, `docs/tasks/task-m26-mac-2026-10.md` 4b) for the generic
pager: `prefillMoEPaged` (`metal/prefill_g4paged.go`), for Mixtral, Qwen MoE and the Qwen3.5/3.6 DeltaNet hybrids (M35's
shape), routed from `PrefillLast` (`moeLayerMajorOn`, on; gpt-oss excluded). Per layer, one command buffer of every row's
mixer (attention, or the DeltaNet mixer) and router through decode's own kernels, the routes read, then phase 2 (routed
experts from the pool and the shared expert) in runs that fit the pool, the grouping now shared with the Gemma 4 path
(`layerMajorExpertGroups`). A DeltaNet layer's recurrent state is per layer, and its rows run in order, as the
sequential loop runs them.
- **Gate** (`TestMoELayerMajor_matchesSequential`): Mixtral, Qwen3-MoE (k = 2 and 3) and the Qwen3.5 MoE hybrid, paged
  at 2 and 3 slots, prompts of 1, 5, 9 rows and 6 rows on a 3-row prefix: last logits, every attention layer's K/V and 3
  decode steps (which read the DeltaNet state) bit-identical to the sequential loop. Phase 2 taking the next row's state
  fails it. The Gemma 4 gates pass on the shared grouping.
- **Through the decoder** (`TestMoELayerMajor_generateThroughDecoder`): `Generate` on paged Mixtral and the paged Qwen3.5
  hybrid, 16 greedy tokens equal with and without, the route taken.
- **Speed, in process, by day** (`TestG4LayerMajor_M26AB`, now driving either pager): the Qwen1.5-MoE 4-layer slice
  paged at 8 slots (M35 is not on this Mac), 5 reps alternated, logits and K/V equal every rep: sequential ÷ layer-major
  **1.261 at 128 tokens, 1.256 at 512**, every rep above 1.22. Staging is unchanged (1,633 against 1,642 experts at
  128): the gain is the round trips, as on M26.
- `TestPrefillLast_declinesPagedGenericMoE` still holds as written: it pins `prefillOK` (the f16 pass's gate), which a
  paged generic MoE still fails; the layer-major branch runs ahead of that pass, on decode's kernels.

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
- 2026-10-03: **F-D02's fixture assertion added** (written on nobara) **and RUN on the Mac: PASS in 0.18 s**, at `0dcc0d1d`. The test file is tagged `darwin && goinfer_testhooks`, so the
  command is `go test -tags goinfer_testhooks -count=1 -run TestWeightAlias_olderBundleTakesCopyPath -v ./metal/`; without the tag it prints `ok ... [no tests to run]`. (The first
  version of this note gave the command without the tag and said the test was compiled for darwin on the strength of `GOOS=darwin go vet ./metal/`, which also leaves the file out;
  `go vet -tags goinfer_testhooks` on that target is the check that compiles it, and it is clean.) The Mac's log: `non-metal bundle (format v15): llama-tiny.int4.giw` takes the copy path
  with the "scales are not in the layout Metal binds ... converted at load" note, and `metal-target bundle (format v15): llama-tiny.int4.metal.giw` aliases, so the writer's format has not
  moved and the test needed no relabelling. The existing older-bundle test already covered the v15 non-metal case without saying so: it builds
  its "old" bundle with today's `prequant` and `GIWTargetNone`, and the writer emits weights format v15 for every target (checked on `testdata/llama-tiny`: both the non-metal and the
  metal-target bundle read v15), so what the test called "a v12 file" is a v15 non-metal bundle. It now reads each bundle's weights-format version from its header (the `GINFW` blob at
  byte 64 behind the `GINFB` v3 bundle header) and asserts both are at least v15, its labels say "non-metal bundle", and its comment says why. The other F-D02 option, binding v15 scales
  directly instead of converting them, is not built: the non-metal case is rare (below).
  - **The other alias tests, same run:**
    - `TestWeightAlias_fixturesByteIdentical`, `_closeCycles` and `_onByDefault` pass.
    - The fixture sweep skips 18 fixtures for structural reasons: 14 families with no Metal resident, two vision-only towers the decoder does
      not load as a model, and `tiny-qwen2-moe`, which has no `config.json`.
    - `TestWeightAlias_logitsByteIdentical` skips without `GOINFER_HEAVY_TESTS` and a real `.giw`. Run with both on the 0.5B's
      `.int4.metal.giw`, it passes: 48 tensors, 170.6 MB of nibbles not copied.
  - **The audit's open question (its "what I could not read" item 8) is answered: the non-metal v15 case is rare.**
    - **A Mac user's normal path writes the metal layout.** `serve` and `chat` turn a `.gguf` into a sidecar through
      `prequant.EnsureCachedGIW`. Its target is `decoder.GIWTargetForBackend(opts.Backend)`, and `-backend auto` is resolved to `metal`
      (`loadflags.resolveAuto`, inside `Options()`) before the sidecar is chosen. So the sidecar is `<base>.<quant>.metal.giw`, kind 7 with
      fused groups. The Mac's own `~/models` sidecars are named that way.
    - **The non-metal layout reaches Metal only off that path:**
      - a bare `prequant` with no `-target`, which resolves to `cpu-arm64` on a Mac;
      - an explicit non-metal `-target`;
      - a `.giw` built on another box and passed to `--model` directly.
      A cpu-target sidecar is never reused by a Metal load, because the cache key carries the target.
    - So binding v15 scales directly is a rare-path nicety and was not built (F-D02 step 3 stops here, by its own condition). F-D02 is closed
      for the common path.
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
- 2026-10-02: **C-P01 built** (`2838b7de`, then the scale pread `494eb05a`) on the owner's word, with its A/B
  pre-registered, amended to three arms before any timed run, and queued after T1.9b.
  The A/B's run script was dry-run end to end on the tiny Gemma 4 MoE (3 slots, 100 tokens): every process completed
  and the summary graded, with tok/s and heap numbers that say nothing (the fixture loads from safetensors, so there is
  no mapping to save).
- 2026-10-02: **C-P01's A/B: SHIP** (pread ÷ old 1.061, 8 of 8 rounds; heap 3004.6 → 1553.2 MB), run by day on the
  owner's word. Copy and pread were indistinguishable at 73% free memory. **T1.9b declined again** by day (budget 4.69
  GB) and is re-queued for a real night.
- 2026-10-02: **E-P01, Phase 3 item 1: done for prompts that end below the fast-prefill floor.** `metalResident.PrefillLast`
  runs such a prompt as decode rows on the batched step kernels (`prefillByStep`, gated by `promptStepOK`: the
  step-kernel verify's own eligibility, `VerifyCost`) instead of declining to the decoder's sequential loop. The rows
  are that loop's bits, so no output changes; T1.10 timed the step at 0.22× sequential on the 1.5B and 0.25× on the 7B
  at every K. Pieces before the last read only their argmax; a one-row last piece runs `Forward`; the route checks
  cancellation between pieces, which the one-pass batched prefill cannot.
  - `TestPrefillLast_stepRouteBitIdentical` (default-run, through `PrefillLast`): fresh prompts of 8, 9, 20, 33 and 63
    tokens and a 23-token suffix at 40, every K/V element and every last-row logit equal to the sequential loop, on the
    fixture and on the 1.5B (0 of 151,936 logits). Mutations: the first row's logits for the last's, positions shifted
    by one, and the batched pass forced on prompts ending at 72, 81 and 132 keys each fail it.
  - **Held: the above-floor half** (suffixes of up to 32 tokens on a prompt that reaches the floor, which T1.10 measured
    faster than the pass). Built first, it broke chunked prefill: such a suffix is also a chunked prefill's tail, and
    on the step it ran decode's numerics where the whole pass ran the pass's. `TestMC5_prefillChunkInvariance` did not
    see it, because N = 1000 leaves every tail it tries above 32 tokens; with C = 81 (a 28-token tail at 972) all
    20,480 last logits differed. C = 81 is now in the gate. Opening the half needs the decoder's chunk cutter to keep
    a tail above 32 tokens on a resident that takes the step route, which is a decoder change, not done here.
  - Not measured: the served TTFT. The speed claim is T1.10's in-process timing of the same step pieces; a served
    check of a short prompt's TTFT, step against sequential, belongs to a night with Batch B.
- 2026-10-02: **E-P01's above-floor half, measured before turning it on: it costs cache-independence, so it stays off
  pending the owner.** Built behind `promptStepAboveFloor` (test-only, off): suffixes of up to 32 tokens on a prompt
  past the floor run on the step. `TestPromptStepAboveFloor_reuseVsCold` (1.5B, set B, 10 prompts, a 512-token prefix,
  then the suffix; a reuse turn against a cold prefill of the same whole prompt, then 32 greedy tokens from each):

  | Suffix | seed argmax agrees | seed KL(cold ‖ reuse), mean / max | 32 greedy tokens diverge |
  |---|---|---|---|
  | 8 | 8 of 10 | 0.0060 / 0.0151 | 8 of 10 prompts |
  | 16 | 9 of 10 | 0.0077 / 0.0352 | 6 of 10 |
  | 32 | 7 of 10 | 0.0073 / 0.0200 | 7 of 10 |

  The control (the half off: the suffix on the batched pass) equalled cold bit for bit in all 30 cases. Why it moves:
  the pass computes the suffix's K/V from f16 activations, the step from decode's int8 ones. So with the half on, a
  conversation's later turns would depend on whether the prefix cache hit; today that is so only for suffixes under 8
  tokens, which run sequentially. Its gain, from T1.10: the step at 0.40× the pass at 16 tokens (about −55 ms on the
  1.5B) and 0.80× at 32 (about −19 ms). It would also need the decoder's chunk cutter to keep tails above 32 tokens,
  not built. A-P01, graded tonight, speeds the same short passes while keeping the pass's numerics.
- 2026-10-02: **E-C01 confirmed and fixed** (latent before, live with a slow `LogitProcessor`): under MC3 a solo token's
  logits and the prompt's seed were the resident's shared host buffer, read by the generation after leaving the
  resident. `TestMC3_soloLogitsSurviveAnotherGeneration` changed A's tokens on 3 of 3 runs before; both seams now copy
  while the resident is held, and each copy's removal fails the test. E-P09: the owner asked whether 2 KV slots would be a
  good compromise; recommended as a Metal-only default (keeps the batched step, saves about 224 MB on the 1.5B and 470
  MB on the 7B from the first token, costs the 3rd and 4th concurrent clients their batching). Not built; awaiting the go.
- 2026-10-02: **C-B01 built, bit-identical, queued for its grade** (owner: "start it now"; widened to untied heads on
  the owner's choice of a lazy device embedding table, after the smoke found the bench set untied). The greedy chain
  (`metal/greedy_chain.go`), its decoder wiring, and two identity tests. Each identity test's mutations fail it. The
  grade's pre-registration is above ("C-B01: built, pending its grade"); the grade itself is queued for tonight.
  The decoder half of the identity test first paired a `Session` turn, which ran nothing on Metal: a plain session decodes on the
  CPU path. It now pairs two `Generate` calls whose second prompt extends the first, which is how a resident reuses a
  prefix.
- 2026-10-02: **C-P02 built, bit-identical, queued for its grade** (owner: "yes" to the sampled chain). It shares C-B01's
  chain and night job (the job re-queued with both modes). On the way it measured a pre-existing effect, not the
  chain's: at T=1, a warm generation (prefix reused, last prompt token re-forwarded on the decode path) and a cold one
  part at token 18 on the 0.5B, where greedy does not. That is E-P01's numerics difference showing through a near-tie
  draw.
- 2026-10-02: **E-P06 built, lossless, queued for its grade** (owner: "this while we wait?"). The verify cost curve is
  measured at load for a model loaded for `--spec ngram`. The 1.5B's real curve at depth 2048 is 8–9% below the shipped
  constant, smaller than the audit's 12–51%, which mixed depth with model.
- 2026-10-02: **Batch B prepared and queued for tonight** (owner: "do this for tonight, ill start it early"). T1.7 gains a
  second, pre-registered reading (the boundary step), because the audit's per-key max/min can pass on attention's fixed
  per-token part alone. The peer run's goinfer is main as shipped, not the branch.
- 2026-10-02: **E-P09 done** (owner: "go"). Metal keeps 2 resident KV slots when `-kv-sessions` is not given; a given
  count is kept. Serve passes the distinction as `decoder.Options.ResidentKVSlotsDefault`. The docs (`server.md`,
  `flags.md`) and the CHANGELOG say so, including that batching 4 clients on Metal now takes `-kv-sessions 4`.
- 2026-10-02: **E-P03 built, bit-identical, queued for its grade** (owner: "do e-p03 now"). FB = 4 is instantiated beside
  2, and `calibrateFB` picks per projection at build. On the 1.5B it keeps FB = 2, as S0 found. The 7B decides tonight.
- 2026-10-03: **the night of 2026-10-02 graded** (results above). C-B01, C-P02 and A-P01 ship. B-P04 and E-P03 are killed
  and their code reverted; the records keep the numbers. T1.7 stands on the 7B and is parked on the 1.5B. T1.11 and T1.14
  are recorded. E-P06 and B-P03 are fixed and re-queued, and T1.9b waits for the owner.
- 2026-10-03: **T1.9b graded on an owner-forced run** (guard bypassed, by day, under the kill-watch): swap +1.60 GB
  within a second of the build, killed before the first token. R11(c)'s "lower default" stands at 64 slots.
- 2026-10-03: **the branch goes to main** (owner: "put what we can on main"). Main gets everything graded or test-only.
  The one ungraded build, E-P06, ships switched off (`verifyCostCalibrate = false`, pinned by
  `TestVerifyCost_offUntilGraded`), so `--spec ngram` keeps pricing the verify by `stepVerifyCost` until tonight's
  grade, which runs pinned binaries and is unaffected. If it passes, the switch goes; if it is killed, the code goes.
- 2026-10-03: **the night of 2026-10-03 graded** (results above; the queue ran by day at the owner's word):
  - **Ship:** B-P01, B-P02 and D-B04. `attnFABlk64On`, `attnFABlkAnyG` and `gemvExtOn` are now true.
  - **B-P03 passes:** `attnFADepthFloor` is 1024. The snapshot golden's straddling checkpoints were re-baked: the
    mixtral-tiny and gemma4 entries did not move, and llama-attnfa-tiny's moved to 900 / 1022 / 1023 / 1100.
  - **E-P06 is killed, and its code removed.** The owner asked whether it ran the speculative code; it did, and
    calibrated.
  - **The post-merge peer read's consistency check failed.** Investigated: serve's MC3 claim keeps both chains off
    serve's path. No benchmarks row was updated.
  The audit continues on main from here.
- 2026-10-03: **B-P03 reverted, then restored** (owner: "revert", then "can we fix so the floor works at where it
  was?"). The `--spec ngram` divergence at 1024 was the test comparing a cold arm with a reused-prefix one. Spec equals a
  plain arm on the same reuse, tokens and K/V, at 1024. `attnFADepthFloor` is 1024 again, and the test compares like
  with like (B-P03's section).
- 2026-10-03: **B-P01 built, off by default, queued for its grade** (owner: "B-P01 first"). The hd = 64 block attention
  kernel for the 0.5B. It is more accurate than the kernel it replaces, and a one-rep smoke read it 3.2–3.8× faster
  at attention. The 0.5B's CPU reference cell Q05 is built tonight as the grade's first step.
- 2026-10-03: **B-P02 built, off by default, queued for its grade** (owner: "then B-P02"). The block kernel at
  G = 2–8 agrees bit for bit with the graded g7 head for head, so its fidelity is inherited. Its speed is graded
  tonight on internlm2 (G = 2): a one-rep smoke read 2.3–2.4× at attention.
- 2026-10-03: **D-B04 built, bit-identical, off by default, queued for its grade** (owner: "then D-B04"). The audit's
  DeltaNet sites needed a coal-family rows kernel to stay bit-identical; it is new. A one-rep smoke on the 9B read
  1.06–1.08× on the token. The routed-expert part is not built.
- 2026-10-04: **the serve chain's served grade pre-registered and queued** (owner: "this first"). The serve chain was
  built on 2026-10-03 (`0277f4ed`) with its identity gates; only its served speed was owed. Serve's shutdown line now
  reports held tokens (`3b9ef839`), and real serve on the 1.5B ran 128 of 128 decode tokens held with the hold on and
  none with it off. The night job is two arms at the same rev, the old one never holding.
- 2026-10-04: **E-P05 on, E-P02 reopened and shipped, E-P08 shipped** (owner: "turn E-P05 on. and E-P02. do E-P08 now").
  All bit-identical. E-P02 1.076x on the 7B's B = 2 step; E-P08 1.022 / 1.036 / 1.081x at B = 2 / 4 / 8 on the 1.5B.
- 2026-10-04: **D-B04's routed-expert half shipped and PrefillTailExact on for MoE** (owner: the two leftovers). Both
  bit-identical; the rows form 1.046x on the Qwen1.5-MoE slice's token and 1.052x on M26's GPU-busy time.
- 2026-10-04: **the peer refresh pre-registered and queued** (owner: "we should set this up to run tonight, right?"),
  after the serve-chain grade. **C-N01:** aikit v1.56.0 has the fix in its tree (`a143d8d`), but the `gpu` module is
  tagged separately and its latest tag, `gpu/v0.33.3`, predates it, so goinfer's metal module cannot take it yet; it waits
  for a `gpu/` tag.
- 2026-10-04: **M-11's cheap half shipped: async phase 2** (1.132x a token on M26); C-B03's fence built, gated bit-exact
  and never stale, and off (0.878x / 0.848x). Lever 3's prefetch the same (0.894x / 0.947x). Record:
  `docs/tasks/task-m26-mac-2026-10.md`, "Decode levers".
- 2026-10-04: **E-P02's calibration follow-up parked**: adjacent per-row up to B = 3 on the 7B, whole step 1.000 at B = 3.
  Not kept.
- 2026-10-04: **layer-major prefill for the generic paged MoE shipped** (owner): bit-identical, 1.26x on the paged
  Qwen1.5-MoE slice's prompt at 128 and 512 tokens.
