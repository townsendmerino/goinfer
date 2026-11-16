# KV8's owed reads on CUDA: Mamba-2 and a real model's cut (registered 2026-11-16 07:40 PST, before any run; nobara: 16-core amd64, 62 GB RAM, one RTX 2070 SUPER 8 GB)

Plan of record: `docs/tasks/task-kv-architecture-2026-10.md`, the KV8 row ("Owed: ... Mamba-2, and a real model's logits for the cut's bit-identity") and
`docs/measurements/kv8-msgboundary-2026-11/README.md` (the second slice's "The cut rests on the split being bit-identical ... not on the logits of a real model").
Both gates are correctness gates, not timed ones: they hold the timing lock shared. Every model is read from `~/models` on NVMe.

## What the tree says before any run (read, not measured)

- **State slots and the per-slot Keep checkpoint cover Mamba-2** (`backend/cuda/recurrent_checkpoint.go`, `backend/cuda/recurrent_slots.go`; `nemotron-tiny` is in S-1, S-2 and C-1 and in KV8a's gates).
- **The host snapshots do not.** `backend/cuda/kvsnap.go:cudaResident.snapshotDecline` refuses a Mamba-2 model and `KVSnapshotRecurrent` is false for it, so serve computes no
  `MsgBoundary` or `Boundary` for it, and `TestKV8Msg_cudaMamba2KeepsNoBoundarySnapshot` pins the refusal. A Mamba-2 prompt also has no batched kernel
  (`backend/cuda/prefill.go`: "a prompt runs one forward per token"), so there is no pass for a mark to fall between. The message-boundary and preamble snapshots therefore do not exist for a Mamba-2
  hybrid, and a restore "at each snapshot point" has nothing to restore from.

## G-M1: what the Mamba-2 hybrids on CUDA hold today (tier 0, tiny fixture, no new code)

| | |
|---|---|
| **Question** | At main, on `nemotron-tiny`, do the existing Mamba-2 gates still pass (state slots, the per-slot Keep checkpoint's bit-exact restore, the pinned refusal of a boundary snapshot)? |
| **Metric** | `go test -tags 'cuda goinfer_testhooks' -run` over the nemotron cells of S-1, S-2, C-1 and KV8a's gates, and `TestKV8Msg_cudaMamba2KeepsNoBoundarySnapshot`, read with `-v` for `--- PASS` (a SKIP is not a pass) |
| **PASS** | every named test runs and passes, 0 skips, and the refusal test shows 0 snapshots taken |
| **FAIL** | any fails: a regression in what KV8 shipped |
| **Ambiguous** | a test that skips on this box: reported NOT RUN, not counted |
| **Stopping rule** | one run, no repeat |
| **Means** | PASS: the Mamba-2 pieces that exist hold; the message-boundary and preamble snapshots are a build (the section above), not a read, and wait for the owner's word on whether to build them. |

## G-R1: a real checkpoint's logits at the cut (tier 1, `qwen3.5-0.8b`, resident CUDA int4)

| | |
|---|---|
| **Question** | On a real checkpoint, is a prefill cut at a snapshot point (the second slice's split), and a restore from a snapshot taken at that point, bit-identical to the uninterrupted prefill? |
| **Instrument** | `backend/cuda/kv8_real_cut_test.go:TestKV8Real_cutEqualsUninterrupted`. `~/models/qwen3.5-0.8b` loaded as the other real CUDA tests load it (`Quant: int4`, 4,096 context). A prompt of 3,000 real ids (the repository's `CLAUDE.md` text through the model's tokenizer). Default 2,048-row passes. |
| **Arms** | **U**, one `PrefillLast` over [0, 3000), twice (U1 and U2). **Cut**, `PrefillLast` over [0, q) then [q, 3000), for q in {64, 960, 1152, 2048, 2368} and the two-cut sets {960, 2368} and {64, 1152, 2368} (the preamble, message and Keep points of the real flow). **Restore**, a snapshot of slot 0 at q taken after [0, q), the slot then run on a different prompt, `RestoreKV` to q, then `PrefillLast` over [q, 3000), for the same q. |
| **Metric** | per arm against U1: the number of float32 words whose bits differ in (a) the last row's logits, (b) the logits of 8 greedy decode steps after the prompt, (c) every DeltaNet layer's conv ring and recurrent state, (d) every f32 attention layer's K and V rows [0, 3000). Plus the arg-max of every logit row compared. |
| **Gated arms** | every arm whose passes are all at least 16 rows (a shorter pass runs decode's GEMV, `TestB35SplitPrefill`); a split with a shorter pass is reported, not gated, as production lowers such a point by a block. |
| **PASS** | U1 equals U2, both plants differ, and every gated arm has **0** differing words in (a) to (d) |
| **AMBIGUOUS, parked for Francis** | some words differ but every arg-max matches and the largest difference in any logit is at most 1e-3 absolute: the cut is behaviour-preserving here but not bit-identical, and the claim "bit-identical" is struck rather than the bar moved |
| **FAIL** | any gated arm with a differing arg-max or a largest logit difference over 1e-3, or any restore arm that differs where its cut arm does not |
| **VOID** | U1 differs from U2 (the instrument is not deterministic on this card, so no equality claim can be graded), or a plant is identical to U (the comparison cannot see a difference). A void run is fixed and re-run, never graded. |
| **Plants (must be red)** | the recurrent state zeroed between the two calls of a cut; the restore arm skipping the state upload by restoring from a snapshot of a different prompt's state |
| **Stopping rule** | all arms, one run; no early stop. A larger prompt or the 9B only for a stated reason why the mechanism changes with scale (not read here) |
| **Cost** | 3,000 ids on a 0.8B is seconds a prefill; about 20 prefills and 10 snapshot cycles, **about 3 minutes** of card time. State the start and finish from `date` when it runs; shared timing lock; the box's load average and free memory recorded at the start |
| **Means** | PASS: the cut's bit-identity rests on a real checkpoint's logits, state and K/V, not only tiny fixtures and greedy replies. AMBIGUOUS: Francis decides what the cut may claim. FAIL: the cut changes a real model's numbers; the second slice's identity claim is withdrawn and the cut stays opt-in until a mechanism is found. |

**Void if:** a run outside the shared timing lock, a model read from `/srv/models`, a skip counted as a pass, a bar moved after a reading.

## What the fit check found (07:26 PST, `fit-*/fit.log`, written before G-M2)

- **`~/models/granite-4.2-3b` is not a Mamba-2 hybrid.** Its config is `GraniteForCausalLM`, 40 dense layers; it loads resident and keeps no recurrent checkpoint. The Mamba-2 hybrids on this box are
  `nemotron-h-4b-instruct-128k` (`NemotronHForCausalLM`, 52 layers, `M-M-M-M*-...`; the CUDA bundle loads resident: 4,322 MiB on the card, a 98.5 MiB recurrent checkpoint kept, 12,288-token context)
  and `granite-hf` / `granite-4.0-h-tiny` (a 7B mixture-of-experts hybrid, whose resident Mamba-2 path is recorded as quality-degraded and is not read here). **G-M2 runs on nemotron-h-4b.**

## G-M2: what a served Mamba-2 hybrid gets today, in T2's shape (tier 1, `nemotron-h-4b-instruct-128k`, one cell)

| | |
|---|---|
| **Question** | In `served_msgboundary.py`'s conversation (about 7,400 ids; r1 cold, r2 the next turn, r3 the last message rewritten from its first words, r4 the turn after r3), what does serve reuse and what does the first token cost on a Mamba-2 hybrid, given that its host snapshots are refused? |
| **Instrument** | `served_msgboundary.py:cell`, one cell of the build at `8a2412048`'s tree (`CGO_ENABLED=0 go build -tags cuda ./backend/cuda/cmd/serve`), `--ctx 12288`, greedy, thinking off, 24 new tokens, `~/models/nemotron-h-4b-instruct-128k` (local NVMe, its CUDA bundle), the exclusive timing lock, the 1-minute load and available memory recorded at the start, a cell started above 8.0 load void |
| **Reported, not graded** | each request's prompt ids, rows reused, first token; the server's snapshot lines; peak server `VmSwap` |
| **Expected, from the code** | no snapshot taken (0 rows); r2 reuses r1's Keep checkpoint rows (Keep is the device checkpoint, which Mamba-2 has); r3 reuses 0 rows and pays a full per-token prefill; r4 reuses r3's Keep rows |
| **PASS** (the status quo is as the code says) | the snapshot count is 0 and r3 reuses 0 rows and r2 reuses at least one row and `VmSwap` at most 512 MB |
| **FAIL** | any snapshot taken on a Mamba-2 model, or r2 reusing 0 rows (the Keep checkpoint not serving), or `VmSwap` over 512 MB |
| **Ambiguous** | none: counts, not timings, decide. A timing is not graded |
| **Stopping rule** | one cell, no repeat; a void cell is re-run once |
| **Cost** | load about 9 s, four requests: r1 and r3 are per-token prefills of about 7,400 ids, each about 1 to 3 minutes by the sequential path (measured only by this run); **expected about 6 to 10 minutes**, stated before the run; over 20 minutes stops it and says so |
| **Means** | PASS: the cost of lacking the message-boundary and preamble snapshots on a Mamba-2 hybrid is r3's first token, on record, and whether to build them is the owner's decision with that figure in hand. FAIL: the Keep path or the refusal is not what the code says. |
