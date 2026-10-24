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
| Qwen3.6-35B-A3B (recurrent; MoE) | `qwen3.6-35b-a3b-q4_k_m.gguf` (its `.int4.e4h.cuda.v17.giw`), experts streamed | `z-lab/Qwen3.6-35B-A3B-DFlash` (on nobara as `qwen36-35b-dflash`) | DFlash | **nobara (CUDA), by the owner's decision** (below) |
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
