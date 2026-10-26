# SP0 — the prize per family: acceptance by teacher forcing, verify-row cost, a projected speedup (registered 2026-10-24)

SP0 of track SP (`docs/tasks/task-forward-architecture-2026-10.md` §6a). No product code: the instrument is a test file
(`cuda/sp0_prize_test.go`) and an offline projection (`project.py`). This page is the registration, committed before any
reading. Readings land below it, on this branch, for the owner's review before anything merges.

## The rule (the owner's, from §6a; registered before any data)

- **Per family, on its box:** the projected decode speedup at that family's best verify width is the decision.
  - **≥ 1.30×:** the family goes on to SP1 (windowed) or SP2 (recurrent).
  - **1.20–1.30×:** parked. Recorded, not advanced; the owner decides whether a better drafter or a tree (SP3) is worth a
    second reading.
  - **< 1.20×:** not advanced.
- **What it is:** a projection to decide whether to start a track. It is not a ship bar; a track that starts still ships on
  its own gates (identity, and the owner's default speed bar against plain decode on the served path).
- **The projection, per width w:** speedup(w) = tokens per cycle(w) × t_step / (t_verify(w) + t_draft(w)).
  - tokens per cycle comes from teacher forcing (below);
  - t_step is one plain decode step;
  - t_verify(w) is one verify pass of w rows;
  - t_draft(w) is the drafter's cost per cycle: the DFlash trunk plus its head, and 0 for n-gram.
  - For n-gram, a cycle with no draft is a plain step. So the projection sums each cycle's own cost,
    t_verify(1 + drafted) or t_step, over the cycles the simulation produced.
- **The best width** is the w with the highest projection, from a ladder fixed here (no stopping early on it).

## Targets, drafters and boxes

| family | target (nobara `~/models`) | drafter | kind | box |
|---|---|---|---|---|
| Qwen3.5 (recurrent: Gated DeltaNet) | `Qwen3.5-9B-Q4_K_M.gguf` (its `.int4.e4h.cuda.v17.giw` sidecar) | `z-lab/Qwen3.5-9B-DFlash` @ `5fc3b3d474760f18c516db87d84c37edbfd3ede6` (apache-2.0, 2.58 GB) | DFlash | nobara (CUDA) |
| gpt-oss (windowed: sliding window 128 on half the layers; MoE) | `gpt-oss-20b-MXFP4.gguf` (its `.int4.e4h.cuda.v17.giw`), experts streamed | n-gram (`decoder.NgramDrafter`, defaults: MinMatch 2, MaxMatch 16) | n-gram | nobara (CUDA) |
| Qwen3.6-35B-A3B (recurrent; MoE) | `qwen3.6-35b-a3b-q4_k_m.int4.e4h.cuda.v17.giw` loaded directly (mmap), experts streamed | `z-lab/Qwen3.6-35B-A3B-DFlash` (on nobara as `qwen36-35b-dflash`) | DFlash | **nobara (CUDA), by the owner's decision** (below) |
| harness validation (dense) | `qwen3-4b` (HF directory) | `qwen3-4b-dflash-f32` (on nobara) | DFlash | nobara (CUDA) |

- **Not targets here:**
  - **Mellum** (sliding window 1024): §6a's windowed slot allows gpt-oss or Mellum, and gpt-oss's 128-token window is the one
    a verify pass wraps soonest.
  - **Bolt E4B** with its DFlash drafter (the candidate in `docs/measurements/bolt-e4b-g0-2026-10.md` § "SP0 candidate"). Its
    drafter loads only as a `dflash`-architecture GGUF or NVFP4 safetensors, and neither loads today
    (`decoder/dflash.go:LoadDFlashDrafter` reads `config.json` + `model.safetensors`). It stays the windowed DFlash candidate
    for when one of those loads.
  - **Flint**, until its F3 lands (§6a).
- **The drafter download** is the first step of the nobara job (`hf download`, pinned revision, into
  `~/models/qwen35-9b-dflash`). The job checks the downloaded `model.safetensors` against the Hugging Face LFS sha256.
  Pushing it to the archive (`models-push`) is a later step.

- **The 35B on nobara, not the Mac (Francis, 2026-10-24 3:33 pm, relayed by the coordinator: "measure it on nobara as one more SP0
  cell").** §6a put it on the Mac. Two things block that: the checkpoint (19 GB as a `.giw`, 21 GB as a `.gguf`) does not fit
  the Mac's free disk above the 5 GB floor, and Metal hosts no block drafter (`metal/` implements no `AttachBlockDrafter`).
  - **What the reading means:** the 35B's prize on nobara's expert-streaming path.
  - **What it does not mean:** the Mac's prize. That reading stays open until Metal can host the drafter and the Mac has the
    disk.
  - **The cost arms:** 10 repetitions after the 3 warm-ups, not 20, because each call streams experts.
- **Resident context 4608 positions, not 8192 (fixed before any data).** The fit check refused the 9B at 8192: 0.54 GB of KV
  plus the drafter's 1.70 GB reserve, against 2.56 GB free beside the weights. The cost arms need at most 4096 + 16 positions.
  The fit-check record is `~/goinfer-logs/sp0-2026-10/fitcheck-2026-10-24-1521/` on nobara.
- **The 35B is loaded from its `.giw`, not its `.gguf` (fixed before any data).** The second fit check
  (`fitcheck-2026-10-24-1540/`) passed the other three loads. It refused the 35B's `.gguf` on the host fit guard: a test binary
  loads a `.gguf` directly into memory, and the guard found that would page to swap. The `.giw` is mapped instead, as the B13
  expert-paging gate loads it on this box.

## Acceptance by teacher forcing (needs no rollback in the target)

- **The teacher:** the target's own greedy continuation, `maxNew` = 256 tokens, from plain resident decode (`Model.Generate`,
  temperature 0). Every prompt's continuation is generated before the resident is touched directly.
- **The DFlash simulation** replays `BlockSpec.generate`'s round logic against the known continuation:
  1. One batched prefill of prompt + continuation with the drafter's taps armed (`SetBatchedCapture`, `PrefillSeedArgmax`)
     gives every position's tap rows.
  2. Each round drafts a block `[anchor, mask × (w − 1)]` (`DraftBlock`, `DraftTokens`).
  3. The round accepts the longest prefix that matches the continuation.
  4. It extends the drafter's context with the rows of the anchor and the accepted drafts, exactly as `generate` folds a
     verify pass's capture, and moves the anchor to the continuation's next token.

  The rounds are what a loop with exact rollback would run, because a lossless verify emits exactly the greedy continuation.
- **The n-gram simulation** calls `NgramDrafter.Draft(prompt + accepted so far, K)` each round and accepts the matching
  prefix. A round with no draft is a plain step.
- **The harness's own gate (validation, run first; registered):**
  - **What:** on the dense pairing, where the real loop runs (`BlockSpec.Generate` with `OnRound`), the simulation's
    committed count per round is compared with the real loop's, at width 8, on the same 12 prompts.
  - **Bar:** at least 95% of rounds identical, and total rounds within 2%.
  - **Below the bar** the harness is wrong and every DFlash reading is void. The batched capture of a full prefill and of a
    verify pass are different passes, so a few differences are possible.
- **Widths:**
  - DFlash: w ∈ {2, 3, 4, 6, 8, 12, 16}, clamped to the drafter's block size. One simulation per width, because the drafter
    drafts a block of width w, not a prefix of a wider one.
  - n-gram: K ∈ {1, 2, 3, 4, 6, 8} drafted tokens (w = 1 + K).

## Prompts (fixed here)

- **code**, 6 prompts, and **math**, 6 prompts: `cuda/default_width_sweep_test.go:widthSweepSuites`, the set the
  verify-width sweep used.
- **edit**, 6 prompts, n-gram only: a code file in the prompt and a small change asked for, with the whole file returned. This
  is the copy-heavy traffic n-gram exists for (`cuda/sp0_prize_test.go:sp0EditSuite`).
- **Thinking:**
  - The template's default for every suite: off for Qwen3.5 (the closed `<think>\n\n</think>\n\n`), and the harmony default
    for gpt-oss.
  - Plus, for the Qwen3.5 target, the math suite with thinking on (`enable_thinking`), because §6a's agent traffic thinks.
- Each class is reported separately, then pooled per family. The decision reads the pooled projection. Each class's projection
  is reported beside it, with no per-class veto.

## Verify-row cost

- **Where:** in the same process, after the acceptance phases, at two context lengths, L ∈ {512, 4096}. The KV is filled
  with the concatenated real continuations, so MoE routing sees real text.
- **The arms, each at start position L and overwriting the same rows:**
  - `step`: `ForwardArgmax`, one row: the plain decode step.
  - `verifyN(w)`: `PrefillLastNArgmax`, w rows. This is the call `BlockSpec`'s verify makes.
  - `forwardN(w)`: `ForwardN`, w rows. This is the call n-gram's verify makes. On this backend it is the sequential loop for
    a Gated-DeltaNet family (`cudaResident.ForwardN`), so for Qwen3.5 it is recorded but `verifyN` is the cost.
  - `draft(w)`: `DraftBlock` + `DraftTokens` with the drafter's context filled to L (DFlash only).
- **Order:** per repetition, the arms run in a rotated order, reversed on alternate repetitions (A B B A over the
  repetitions). 3 warm-up repetitions are discarded, then 20 repetitions; the median and the interquartile range are reported.
- **Which arm each projection uses:**
  - Qwen3.5 uses `verifyN`. It is the batched pass with the row-batched DeltaNet recurrence, so it is the cost a chain verify
    in SP2 would pay, less the fold (SP2's gate prices the fold against R-01 phase 1's 11.0%).
  - gpt-oss uses the cheaper of `forwardN` and `verifyN`, named in the reading.
- **The MoE caveat, stated before the data:** on CUDA an MoE target's batched pass runs its FFN row by row
  (`cudaResident.prefillStaticDecline`'s MoE note), so gpt-oss's verify cost grows with w. That is this box's cost today, and
  the reading says so. It is not Kapitsa's 1.56× for 8 rows, which batches the experts.
- **Projection length:** L = 512 for the decision, near the teacher continuations' own context. L = 4096 is reported beside
  it, to show whether the ranking moves with context.

## Pre-registration (`docs/measurements/noise-registry.md` §8)

- **Tier:** day. **Testing tier:** 1.
  - Acceptance is numerics on one small real model per family.
  - Cost is a per-call timing in one process, interleaved.
  - No served gate: SP0 projects, and the tracks it starts measure served speed on their own gates.
- **Interleaving:** the cost arms rotate per repetition (above). Acceptance is deterministic (greedy, fixed prompts), so it
  needs no interleaving.
- **Cell validity:** the job records the load average and `MemAvailable` at the start of each target's run. A run started
  above 8.0 load, or under the fit plan's need plus 2 GB, is void and re-run.
- **Reused readings:** none. No earlier reading covers these pairings' acceptance or this verify-row cost.
- **Fit check:** the job's own binary and loads, as a queued job of its own (`run-sp0.sh`, `MODE=fitcheck`, about 10 min with
  the drafter download):
  - every load the measurement makes, load only, drafter attached: Qwen3-4B with its drafter, Qwen3.5-9B with the drafter
    reserve, and gpt-oss-20b with experts streamed (`MoECacheExperts`);
  - Qwen3.6-35B-A3B with its drafter, experts streamed;
  - all at `ResidentContext` 4608 and f32 KV (the CUDA drafter refuses an f16 cache, `cudaResident.AttachDrafter`);
  - a refusal exits 75;
  - the measurement job is queued only after every load fits.
  - Desktop compositor state is recorded.
- **Never-swap:** the test binary runs directly (`go test -c`). Its `VmHWM` and peak `VmSwap` are sampled from
  `/proc/<pid>/status` until it exits.
- **Instrument:** the cheapest that answers. Teacher forcing needs no rollback, so the tracks themselves do not have to exist
  to price them. Per-call timing in one process is the whole-token in-process instrument TE5(b) starts a speed question
  with.
- **Noise:** per-call timing in one process. The interquartile range of the 20 repetitions is reported for each arm, and a
  projection whose band crosses 1.30 when t_verify takes its quartiles is reported as such.
- **Stopping rule:** fixed. All prompts, widths and arms, no early stop.
- **Ambiguous band:** 1.20–1.30×, parked, the owner decides (above). A family whose projection at the cost quartiles spans
  1.30 is also parked.
- **Cost (estimate):** the job runs under nobara's timing lock (night.py serve queue, by day, behind any queued job).
  - Drafter download: 2–4 min.
  - Build: 2 min.
  - Validation pairing (Qwen3-4B): about 4 min (load 30 s, 12 × 256 tokens through `BlockSpec` and plain greedy, the simulation).
  - Qwen3.5-9B: about 10 min (load and drafter attach 1 min; 18 continuations × 256 tokens at about 50 tok/s, 1.5 min; seven
    simulation widths, 3 min; cost, 2 × 23 repetitions × 5 widths, 2 min).
  - gpt-oss-20b: about 15 min. Expert streaming on 8 GB is the unknown: 18 × 256 tokens at an assumed 15–25 tok/s is 3–5 min.
    The cost arms are slower per call.
  - Qwen3.6-35B-A3B: about 25 min. 12 continuations at an assumed 5 tok/s with experts streamed is 10 min; the cost arms at 10
    repetitions are about 12 min.
  - **Total about 60 min.** The start and finish times are stated when the job is queued.

## Readings (2026-10-24, nobara, 16:24:47–16:40:51 PDT; for the owner's review, not merged)

**Verdicts against the registered rule** (projected ≥ 1.30× at L = 512, its box; 1.20–1.30× parked):

| family | at L = 512 (the decision) | at L = 4096 (beside it) | verdict |
|---|---|---|---|
| Qwen3.5 (recurrent): 9B + DFlash | 1.55× (w = 4) | 1.29× | **on to SP2** at 512; at 4096 it sits just under the bar |
| Qwen3.6-35B-A3B (recurrent, MoE): + DFlash, experts streamed, on nobara | 1.22× (w = 3) | 1.05× | **parked** (the owner decides) |
| gpt-oss (windowed, MoE): 20b + n-gram | 1.07× (K = 1) | 1.28×, band 0.74–2.30 | **not advanced** (SP1 not started on its account) |

`sp0-measure` on nobara's `night.py serve`, under the timing lock, at `b03412f3` (harness and job), driver 595.91.07, kwin_wayland
running. Every cell started below 3.1 load with at least 59.6 GB available and peaked at 0 MiB swap (`raw/cells.txt`). Raw records:
`raw/*.jsonl`. Projection: `python3 project.py dflash|ngram|validate raw/<file>`.

### The harness's own gate: FAIL as registered, with the reason

| bar (registered) | reading |
|---|---|
| ≥ 95% of rounds identical | 151 of 1,219 (0.124) |
| total rounds within 2% | the loop 151, the simulation 1,219 |

**Why it failed:**
- **The two agree wherever both ran.** On every one of the 151 rounds the real loop ran, the simulation committed the same count: 151 of
  151, every prompt, in order (`python3 project.py validate raw/validate-qwen3-4b.jsonl` prints each).
- **The loop stopped early.** It stops speculating after 6–30 rounds, because `BlockSpec`'s acceptance guard
  (`breakEvenTokensPerRound` 2.5 over `guardWindow` 6 rounds) switches it to plain decode, whose steps are not rounds.
- **The bar was the mistake.** It assumed the loop speculates to the end.
- **The guard likely tripped because of thinking mode.** The Qwen3-4B prompts were rendered with the template's default, which is
  thinking on. The earlier pairing measurements (`docs/spec/08-dspark-dflash.md`) rendered thinking off, the drafter's training
  distribution.

**The re-run, by the owner's decision (Francis, 2026-10-24 10:18 pm, relayed by the coordinator: "option A"), registered before
it runs:**
- **What runs:** the same 12 prompts, rendered thinking off (`MODE=validate-off`, `SP0_THINK=off`), so the loop speculates to the
  end.
- **A prompt on which the guard still stops the loop** (its committed tokens fall more than one block short of the continuation)
  is graded on the rounds the loop ran, and named in the record.
- **Every other prompt** is graded on all its rounds, and its round total counts toward the 2% bar.
- **The bars are as first registered:** at least 95% of compared rounds identical, and total rounds within 2% over the prompts
  that ran to the end. With no such prompt the 2% bar has nothing to test, and the harness stays VOID.
- **`project.py validate` implements this.** On the first run's data it grades all 12 prompts on the rounds they ran (151 of 151
  identical), finds no untripped prompt, and prints VOID.

**The re-run's result: HARNESS VALID** (2026-10-24, 22:21:29–22:22:28 PDT, at `88f7cfd0`, load 0.20 at its start, 0 MiB swap;
`raw/validate-qwen3-4b-off.jsonl`, `raw/cells-validate-off.txt`).
- **Did the loop run to the end?** Yes. The guard stopped no prompt, and the real loop speculated to the end of all 12.
- **Rounds identical:** 477 of 487 (0.979; bar 0.95).
- **Total rounds:** the loop 489, the simulation 486, 0.61% apart (bar 2%).
- **The rounds that differ** are mostly a prompt's last round. Near the end of the continuation, the real loop narrows its width to
  the tokens left (`blockSpecRoundWidth`), where the simulation keeps w = 8 and is cut at the continuation's end.

**So the DFlash readings below are graded, no longer provisional.** The first validation's FAIL is kept above as recorded.

### The projections at L = 512 (the registered decision length)

| family | best width | tokens per cycle | t_step | t_verify | t_draft | projected | band (cost quartiles) | registered reading |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B + DFlash | w = 4 | 3.30 | 16.63 ms | 24.02 ms | 11.28 ms | **1.55×** | 1.55–1.56 | ≥ 1.30: on to SP2 |
| Qwen3.6-35B-A3B + DFlash, experts streamed | w = 3 | 2.76 | 17.57 ms | 36.37 ms | 3.35 ms | **1.22×** | 1.10–1.23 | 1.20–1.30: parked |
| gpt-oss-20b + n-gram | K = 1 | 1.19 | 13.76 ms | 18.8 ms (w 2) | 0 | **1.07×** | 0.83–1.29 | < 1.20: not advanced |

- **Qwen3.5-9B, per class:** code 1.49×, math 1.59×, math with thinking on 1.59×, all at w = 4. Its projection by width rises to
  w = 4 and falls past it: 1.12, 1.41, 1.55, 1.41, 1.17, 0.89, 0.73.
  - The drafter's block is a large part of the cycle: 11.3 ms at w = 4. It is 6 layers at hidden 4096, and the target's 248k-row
    head runs over the drafted rows.
  - `verifyN` is the batched pass with the row-batched DeltaNet recurrence. `forwardN`, the sequential loop, costs 67.1 ms at
    w = 4.
- **Qwen3.6-35B-A3B:** a verify pass of 8 rows costs 5.0× a step (88.4 ms against 17.6), where Kapitsa reports 1.56× on its
  backend. That is the CUDA MoE path running its FFN row by row, as registered, with experts streamed on an 8 GB card.
  - Acceptance is good: 5.54 tokens per cycle at w = 8. The cost is what parks it.
- **gpt-oss-20b, n-gram:** the drafter finds a match on 22–42% of rounds (edit 0.42 at K = 1, code 0.22). Past K = 1 its projection
  falls on every class.
  - At L = 4096 its best is 1.28× at K = 1, but the band is 0.74–2.30. The step's own spread at 4096 is wide, so that reading
    resolves nothing.

### At L = 4096 (reported beside the decision, not part of it)

- **The first run stopped short of these arms.** Both DFlash cells stopped filling the context to 4096. The capture armed for the
  drafter keeps a batched prefill from chunking, and one 4096-row pass ran out of device memory (`CUDA_ERROR_OUT_OF_MEMORY`).
- **The fix:** the fill is prefilled, and the drafter's context fused, in 512-row pieces.
- **The re-run:** `MODE=cost4096`, 2026-10-24, 17:06:37–17:12:08 PDT, at `48e1b806`, under the timing lock, at load 1.94 or below,
  with 0 MiB swap (`raw/cells-cost4096.txt`, `raw/cost4096-*.jsonl`). It read only those arms, against the acceptance records
  above.

| family | best width | t_step | t_verify | t_draft | projected | band | at L = 512 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B + DFlash | w = 4 | 18.05 ms | 30.45 ms | 15.80 ms | **1.29×** | 1.28–1.29 | 1.55× |
| Qwen3.6-35B-A3B + DFlash | w = 4 | 18.80 ms | 55.21 ms | 8.14 ms | **1.05×** | 1.02–1.09 | 1.22× |
| gpt-oss-20b + n-gram | K = 1 | 31.50 ms | 24.5 ms (w 2) | 0 | 1.28× | 0.74–2.30 | 1.07× |

- **What drives the drop:** context costs the drafted cycle more than the plain step.
  - For the 9B at w = 4, the step rises 9% (16.6 to 18.1 ms), the verify pass 27% (24.0 to 30.5), and the drafter's block 40%
    (11.3 to 15.8). The drafter attends over its whole context.
  - The 35B's verify pass rises 12% at w = 4 (49.2 to 55.2 ms) and its drafter's block 117% (3.75 to 8.14 ms).
- **How to read it with the decision:** the 9B's 1.30× clears at 512 and falls just inside the parked band at 4096. Its prize is
  largest at short context, and an agent session's context is long.

### What these numbers are not

- **Not served speed.** They are per-call timings in one process, and acceptance is greedy and teacher-forced on 6 + 6 prompts.
- **Not SP2's whole cost.** The projection leaves out SP2's fold (registered for SP2's own gate, against R-01 phase 1's 11.0%).
- **Not the Mac's prize.** The 35B's reading is nobara's (the owner's decision); the Mac reading stays open.
