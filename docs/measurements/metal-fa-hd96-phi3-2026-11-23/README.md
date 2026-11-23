# Metal's attention_fa at head dim 96 (Phi-3 mini): registration (2026-11-23)

**Status: SHIP, 2026-11-23: gates 1, 2 and 3 pass (gate 3 at 1.292× at depth 3,900); on by default at a context of at most 4,096 (Francis, 2026-11-23, § "Default"). Registered
2026-11-23, before any gate reading.** Owner's go: Francis, 2026-11-23 ~08:45 PST ("let's finish the
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

## Addendum 2026-11-23 08:40 PST, before any gate reading: gate 3's arm check and scripts

- Metal has no per-op cost report, so CUDA's launch-count probe cannot run here. [`probe_lane_metal.py`](probe_lane_metal.py) takes
  its place: each arm serves the depth-2,048 prompt with `GOINFER_ATTNFA_DEBUG=1`, and the lane arm must log `attention_fa` dispatches
  at hd 96 while the exact arm logs none. The cell does not run otherwise.
- [`run_gate3_served.sh`](run_gate3_served.sh) is CUDA's served script on Metal (macOS settle under load 2.0, the Mac's idle gate);
  it grades with CUDA's [`grade_cell.py`](../fa-hd96-phi3-2026-11-22/grade_cell.py) unchanged.
- Unchanged: the gates, cells and bars above.

## Gate 1, 2026-11-23 09:05–09:14 PST (MacBook Pro M1 Pro, macOS 26.7.3): PASS

- **Run:** [`gate1/tier0-mac.log`](gate1/tier0-mac.log), the kernels at `44c4045aa`, base `26a73a8b0` (origin/main the branch started from),
  under the Mac's shared lock.
- **`TestAttnFABlk96MatchesFloat64`:** all 32 cases inside the bar; worst cosine 1.0000000, worst max abs 1.87e-7.
- **The plant** (`TestAttnFABlk96_plantGoesRed`, one of each lane's three P·V dims dropped): 576 heads outside the bar, worst cosine 0.66.
- **Selection and the rows form:** `TestAttnFAHeadDimOK_hd96`, `TestFlashDecodeHD96_metalSelection` (off and on) and
  `TestAttnDigest_mc3RowsOtherGeometries/hd96-G1` (the batched step bit-identical to the single-row path) pass, with the existing
  block-kernel pins (`TestAttnFABlkIsTheGradedKernel`, `TestAttnFABlkMatchesFloat64`, `TestAttnFABlkSelection`).
- **Existing kernels unchanged:** all 1,097 base digest lines are present with the same hash on the branch
  ([`gate1/digests-base.tsv`](gate1/digests-base.tsv), [`gate1/digests-hd96.tsv`](gate1/digests-hd96.tsv)); the 100 new lines are
  the hd-96 kernel, its rows form and the model-decode lines of the new fixture case.
- **Not this change's:** `TestAttnDigest_coversEveryKernel` fails on base and branch alike, on the same seven prefill kernels
  (`attention_prefill_steel_hd64` / `hd80` / `hd96` / `hd256` / `hd512`, `_split`, `attention_prefill_tiled`), which no test in
  `attnDigestRun` records. Both new entry points have digests.
- **Gate 2's inputs:** the `P3M` reference files copied from nobara byte for byte ([`p3m-refs.sha256`](p3m-refs.sha256)), and the
  Mac's `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf` is nobara's file (sha256 `8a83c7fb…` on both).

## Gate 2, 2026-11-23 09:15–09:24 PST (MacBook Pro M1 Pro, on AC): **PASS** by the registered rule

- **Run.** `TestR2_decodeFidelityGate` at `31ab6adb6` with the registration's environment (`P3M`, `q4k`, `GOINFER_METAL_GATE_HD96=1`,
  set B, K = 3,900), under the Mac's exclusive lock, started at load 1.27. Log: [`gate2/gate2.log`](gate2/gate2.log). Graded by
  [`grade_gate2.py`](grade_gate2.py).
- **Preconditions held:** the reference-identity check on all 10 prompts (seed-row KL 0.0001–0.0004 against the CPU f32 rows); every
  layer took the lane in the candidate arm (the harness fails otherwise); the lane's outputs are not the exact arm's (the pooled KL
  sums differ, ratio 1.0002).

| | exact (`attention`) | lane (`attention_fa_blk96`) |
|---|---|---|
| mean teacher-forced agreement | 99.84% | 99.84% |
| hard flips | 0 / 640 | 0 / 640 |
| mean KL(reference ‖ arm) | 0.0001 | 0.0001 |

- **(a) hard flips: PASS**, 0 against 0. **(b) agreement: PASS**, +0.00 pt, the lane at or above exact on 10 of 10 prompts.
  **(c) KL: PASS**, ratio 1.0002 (the harness's unrounded figure; the grader's 1.0000 is from the 4-decimal printed means).
- Beside it, not deciding: the harness's own amended verdict PASSES; its strict pooled critC reads false because the lane's KL is lower
  on only 2 of 10 prompts, a criterion CUDA's rule does not have.
- Both arms sit about 30× closer to the reference than CUDA's gate 2 did (mean KL 0.0001 against 0.0031 there): a different backend's
  exact path, not a property of the lane.
- **Verdict: PASSES.** Gate 3 runs next.

## Gate 3, served, 2026-11-23 09:25–09:34 PST (MacBook Pro M1 Pro, on AC): **SHIP** by the registered rule

- **Run.** [`run_gate3_served.sh`](run_gate3_served.sh) under the Mac's exclusive lock, one serve binary from `0d6493608` (sha256
  `ef7999e4…`) for both arms. Records: [`gate3/`](gate3/) (`provenance.txt`, `probe.log`, `decode-metal-all.json`, `decode-metal.log`,
  `verdict.txt`, `serve-logs/`).
- **The arm probe passed:** the lane arm logged 544 `attention_fa` dispatches at hd 96 (32 layers × 17 decode steps at depth 2,048);
  the exact arm logged none.
- **All 8 cells valid:** each decoded `metal-resident (q4k)`, the token gate ok, swap grew 0.0 MB, start loads 0.75–1.77, on AC
  throughout (`pmset`: AC since 09:05:14).

| depth | lane ÷ exact, per block | median | lane, tok/s | exact, tok/s |
|--:|---|--:|--:|--:|
| 128 | 0.9990, 1.0049 | **1.0020** | 41.35 | 41.27 |
| 3,900 | 1.2912, 1.2928 | **1.2920** | 28.49 | 22.05 |

- **Verdict: SHIP.** The median is at least 1.02 at 3,900 (1.292) and at least 0.98 at 128 (1.002). The lane lands behind
  `--flash-decode-hd96`, which now applies on Metal as on CUDA; whether it becomes Phi-3's default on either backend is Francis's call.
- At 3,900 the token goes from 45.4 ms to 35.1 ms. The review's estimate was 20–25% of the token at depth; the reading is 23%.
- **Bytes.** With the option off, none. With it on, the resident's partial buffer (32 heads × 98 floats × 32 splits × 4 B = 401 KB) and
  the batched step's per-row copy (up to 8 rows, 3.2 MB), which the step's scratch reserve prices into the fit check. The longest
  context that fits does not move: Phi-3 mini's window is 4,096 positions, and both arms plan it.

## After the merge of origin/main, 2026-11-23 09:45 PST: the Metal lane stays opt-in under CUDA's new default

#1241 (Francis, 2026-11-23) made CUDA's hd-96 lane the default at a context of at most 4,096: serve and goinfer-chat now set
`FlashDecodeHD96` with `FlashDecodeHD96Default` when the flag is not given. Read alone, that would have turned this lane on by default
on Metal, a default no gate here decided. Metal now takes it only when the flag is given (`hd96Chosen`). Checked on the Mac at
`764bc044c` ([`post-merge/checks.log`](post-merge/checks.log)): `TestFlashDecodeHD96_metalSelection` passes its not-given, chosen and
caller's-default cases, and a served Phi-3 request at depth 2,048 logs no hd-96 `attention_fa` dispatch without the flag and 160 with
`-flash-decode-hd96`. The gate readings above are unchanged: both gate-3 arms passed the flag explicitly.

## Follow-ups from review, 2026-11-23 (after the merge of #1245)

**1. Gate 1 never exercised the block table or the window** (review finding 1). `TestAttnFABlk96MatchesFloat64` and
`TestAttnFABlk96Digest` ran on the identity table at window 0, where a run's second part (`base1`) and `winStart` reduce to `b0 + j`
and a kernel that ignored either would pass. Both now also run permuted 64-key blocks behind a non-identity table (one block more
than the keys fill, so physical block 0 and the order both differ) and windows of 1,000 and 2,500 keys; at 16 splits the 3,900- and
4,100-key chunks start off a block edge, so runs cross it. On the Mac at `cc83c8c3d` ([`followups/tests-mac.log`](followups/tests-mac.log)):
- **The kernel:** all 60 cases inside the bar (the 32 above and 28 new), worst cosine 1.0000000, worst max abs 1.87e-7. The
  gate 1 verdict stands.
- **`TestAttnFABlk96_tableAndWindowPlantsGoRed`:** a run that crosses a block edge without the table (`base1 = base0`) puts 360
  heads outside the bar on the paged cases and **0 on the cases gate 1 had**, which is the gap closed; a window ignored
  (`winStart = 0`) puts 288 outside on the windowed cases and 0 on the rest; a `base1` off by one key goes red on both (360 and
  576: an index shift shows on the identity table too).

**2. Gate 2's grader read the KL ratio from the rounded means** (review finding 2). `grade_gate2.py` divided the two printed means,
each to 4 decimals (0.0001 / 0.0001). It now reads the harness's `KLratio`, computed from the unrounded means: **1.0002**, as gate 2's
section above quotes. Re-graded from the same log: (a) PASS, (b) PASS, (c) PASS, outside the 1.05–1.10 band. **The verdict does not
change: PASS.**

## Default, 2026-11-23: on at a context of at most 4,096 (Francis)

- **The decision.** Francis, 2026-11-23, about 11:15 PST, relayed by the coordinator: option (a), CUDA's rule (#1241) on Metal. The lane
  is on by default for a head-dim-96 load whose resident context cap is at most 4,096, what the gates read; `--flash-decode-hd96` reaches
  a longer context, and `=false` turns it off. Conditional on the review follow-ups above (the table and window cases, and gate 2 re-graded) coming back clean, which they did.
- **Its shape.** With the flag not given, serve and goinfer-chat set `FlashDecodeHD96` with `FlashDecodeHD96Default`. The resident sizes
  the lane whenever `FlashDecodeHD96` is set, and `faHD96On` lets an hd-96 layer take it when chosen, or under the default while
  `ctxCap` (after `kvRaise`, the depth a conversation can reach) is at most `metalHD96DefaultMaxCtx`. `TestFlashDecodeHD96_metalSelection`
  holds the rule (red before the change: the two default rows): on at 4,096 and off at 4,097 under the default, on at 8,192 when chosen,
  off at `=false`, and a head-dim-64 layer on with the option on and off.
- **What it rests on:** gates 1–3 above (gate 3 SHIP, 1.292 at 3,900 and 1.002 at 128) and the bytes in gate 3's section.
