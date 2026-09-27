# MC3 — batched decode on Metal SHIPS: 4 clients at 1.59× the serialized aggregate, p99 turn 0.66×, a lone request unchanged, every reply identical (2026-09-26)

MC3 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md): several conversations on one Metal resident
model decode at once. Each runs on its own MC1 resident KV slot, and their decode tokens join shared steps. The
kernels come from S0 ([`concurrency-mc3-s0-2026-09-26.md`](concurrency-mc3-s0-2026-09-26.md)) and the in-sequence step
from S1 ([`concurrency-mc3-s1-2026-09-26.md`](concurrency-mc3-s1-2026-09-26.md)). This record is the build, and its
grading on the W7 workload against gates pre-registered before any W7 timing of it (`505cc6fc`).

**Result: all five gates pass.** At 4 clients, `serve`'s default configuration:
- reads **121.5 tok/s against the serialized build's 76.2** (1.593×; the three pairs read 1.583–1.599×);
- cuts p99 per-turn latency from **7.09 s to 4.67 s** (0.659×);
- leaves a lone request where it was (p50 1.002×, p99 1.004×);
- gives every turn of every conversation the same reply, and the same reuse, as the serialized build.

By the pre-registered decision rule, MC3 ships enabled under `serve`'s existing default (`-max-concurrent` 4).

## What was built

- **The batched step** (`metal/batch.go`, `metalResident.StepBatch`):
  - per sequence, production's own kernels: norm + quantisation; RoPE; the KV store into the sequence's slot;
    attention over it (attention_fa at or above `attnFADepthFloor`, planned for that sequence's own depth); ctx
    quantisation; SwiGLU; the final norm;
  - once for all sequences, S0's matrix-unit matmuls with production's epilogues.
  - Built only for an eligible resident: plain dense W4A8, at least two KV slots, and a GPU whose `simd_sum` is the
    xor tree the kernels reproduce (checked at build; otherwise batching stays off). It is compiled as its own small
    library with the resident's fast-math setting.
- **The coordinator** (`decoder/mc3_batch.go`, `Model.EnableResidentConcurrency`):
  - Each generation takes a holder place, and a KV slot no other holder uses.
  - Prefill and slot bookkeeping run in an exclusive section between steps.
  - Each decode token is submitted, and a run serves it: in one `StepBatch` when at least two batch-eligible tokens
    are waiting, else through production's own per-sequence call. So a lone generation takes exactly today's path.
  - Tokens join at token boundaries, with a 4 ms straggler window opened when the resident comes free.
  - Speculation, vision and adapters take the resident exclusively, and that claim fails while any holder has a
    slot.
  - The optimistic forward and device top-K are off under MC3: both touch the resident outside the section.
- **serve:** `-max-concurrent` admits a GPU-resident model that can batch, capped by its slots. A model serving
  speculation, a drafter or an adapter keeps one generation at a time. The concurrency line prints once it is
  decided (below).

## Found on the way (each fixed, each pinned by a test shown to go red)

| defect | how it showed | fix | pinned by |
|---|---|---|---|
| MC1's slot clamp priced after the build had allocated its weights: the live-memory budget no longer included them while its base still did | S1's first load was granted 2 of 8 slots (28 MB each) with 5.5 GB available | `6807ab95` | `TestMetalKVSlots_pricedBeforeTheBuild` (2 of 4 before, 4 of 4 after) |
| `Buffer.At` sets the offset from the allocation start; it does not add | S1: sequences 1–3 read sequence 0's K/V (every logit off by 12–41) while sequence 0 matched | in S1's code | `TestMC3Step_bitIdentical` |
| serve printed the concurrency line inside the load banner, before `setConcurrency` decided it | every model's banner said "one generation at a time", so MC3's first smoke run read as not engaging | `2b1cc280` | `TestSetConcurrency_reportsTheDecidedValue` |
| the straggler window was timed from each token's submission | a token submitted during a run had "waited" the whole run, so 4 generations phase-locked into 3-wide + solo on every token: 255 of 256 runs straggler-started, 4.83 s against 3.91 s once fixed | `d2225ec4` | `TestMC3_concurrentGenerationsFillSteps` (old window: 12 of 82 steps 4-wide, 53 of 95 runs straggler-started) |

## Identity and correctness gates run on the build

- `TestMC3Step_bitIdentical` and `_Deep` run production `forwardMulti` against `ForwardEmb` on twin slots, 4 sequences
  × 12 steps: **0 logits differ**. This holds at depths 5 / 23 / 40 / 300, and at 1936 / 1530 / 100 / 1600, where one
  sequence crosses `attnFADepthFloor` mid-run.
- `TestMC3_concurrentMatchesAloneOnMetal` runs 4 conversations × 3 turns × 32 greedy tokens through `Model.Generate`
  on the 1.5B. Ids and reuse are identical to the same conversations run alone. It served 277 tokens through 96
  batched steps; the serve-path greedy tokens a step serves take the host-argmax path.
- The decoder's `TestMC3_*` tests use a fake resident that panics on any concurrent entry and hashes each slot's
  history into its logits. Under `-race` they pass 10–30 repeats, and a wrong-slot mutant turns the identity test
  red.
- The tagged Metal suite ran 182 pass, 0 fail; the decoder suite 648 pass, bar the parity manifest, refreshed
  goldens-proven (38 passed, 0 failed); serveapp ok.

Logs: [`mc3-metal-identity.log`](concurrency-mc3-2026-09-26/mc3-metal-identity.log),
[`metal-suite-tagged.log`](concurrency-mc3-2026-09-26/metal-suite-tagged.log),
[`decoder-suite.log`](concurrency-mc3-2026-09-26/decoder-suite.log).

## The W7 grading

As registered (`505cc6fc`):
- M1 Pro 16 GB, macOS 26.6.2.
- *old* = `serve-metal` at `9efc3185`: MC1 plus R18b plus the slot fix, so a Metal resident serialises.
- *new* = `serve-metal` at `d2225ec4`.
- Both ran at serve's defaults; the server logs show the old build's 7 cells serialising and the new build's
  7 cells printing *"4 generations at once, each on its own resident KV slot, decode tokens batched"*.
- `scripts/bench_w7_plain.py` on qwen2.5-coder-1.5b-instruct q4_k_m from `~/models`: 6 turns × 128 greedy tokens per
  client, `--fixed-nonce`, a fresh server per cell.
- Cells interleaved old new new old old new at 4 clients, then at 1 client, then one pair at 2 clients.
- Idle-gated per cell (load1 ≤ 2.0); 2026-09-26 22:34–22:39 PDT.

The first launch (22:32) was stopped before any cell ran: the bench's free-page wait (1500 MB) cannot be met when model
mmaps leave ~150 MB of pages free with 72% reclaimable. It was relaunched with a 100 MB floor
([`w7-run-attempt1-memwait.log`](concurrency-mc3-2026-09-26/w7-run-attempt1-memwait.log)).

Raw: [`w7.json`](concurrency-mc3-2026-09-26/w7.json), [`w7-run.log`](concurrency-mc3-2026-09-26/w7-run.log),
[`w7-servers.log`](concurrency-mc3-2026-09-26/w7-servers.log), [`run-w7.sh`](concurrency-mc3-2026-09-26/run-w7.sh).
Gates computed by [`gates.py`](concurrency-mc3-2026-09-26/gates.py) → [`gates-output.txt`](concurrency-mc3-2026-09-26/gates-output.txt).

| gate (pre-registered) | result |
|---|---|
| 1. identity: every turn's reply equal, new vs old, same client count and index, every cell | **pass**: 14 cells, every turn |
| 2. reuse: every turn's prompt − reused equal | **pass**: 0, 157, 311, 466, 618, 766 reused on turns 1–6, every client, every cell |
| 3. 4-client aggregate, new ÷ old, median of 3 pairs ≥ 1.2× | **pass: 1.593×** (1.583, 1.593, 1.599) |
| 4. 4-client p99 turn, new ÷ old, median ≤ 1.0 | **pass: 0.659×** (0.649, 0.659, 0.659) |
| 5. lone request, p50 and p99 turn, new ÷ old, median ≤ 1.05× each | **pass: p50 1.002×, p99 1.004×** |

| cell | old | new |
|---|---:|---:|
| 4 clients, aggregate | 76.2 / 76.2 / 76.7 tok/s | **121.5 / 121.9 / 121.5 tok/s** |
| 4 clients, p99 turn | 7.09 / 7.17 / 7.08 s | **4.67 / 4.65 / 4.66 s** |
| 4 clients, p50 turn | 6.56 / 6.54 / 6.51 s | 4.18 / 4.15 / 4.18 s |
| 1 client, p50 turn | 1.658 / 1.657 / 1.657 s | 1.659 / 1.663 / 1.660 s |
| 1 client, p99 turn | 1.788 / 1.795 / 1.890 s | 1.795 / 1.907 / 1.890 s |
| 2 clients (reported), aggregate / p99 | 76.3 tok/s / 3.58 s | 81.7 tok/s (1.071×) / 3.37 s |

Reading:
- **4 clients: 1.59×, against S1's in-sequence 1.84× at B = 4.** The gap is what W7 adds and S1 does not:
  - the conversations grow to ~900 tokens, and per-sequence attention does not amortise (S1 at depth 512: 1.73×);
  - a newcomer's prefill of each next turn runs whole between steps, pausing the others (chunked prefill is MC5).
- **2 clients: 1.07×.** S1 measured B = 2 at 1.09–1.10× in sequence, so this is the fragment's fixed cost, as
  predicted.
- **p99 under load falls by a third.** A turn at 4 clients now takes 2.47× a lone request's, where the serialized
  build's took ~4× (the queue). That ratio is reported, not gated, under the owner's reading.
- **The lone request is untouched.** Its tokens run production's own call. The p99 pair at 1.063× (pair 2) is one
  slow turn among six, and the median is 1.004×.

## Scope, and what is next

- **Shipped:** Metal, plain dense W4A8 families (the qwen2 shape graded here), up to 8 sequences per step, capped
  by `-kv-sessions`' slots (4 by default). Anything else keeps one generation at a time, and the banner says why:
  MoE, recurrent, hybrid, vision, speculative, adapter models, and a GPU whose `simd_sum` is not the tree.
- **Not measured here:**
  - the 7B end to end (S1 put its in-sequence B = 4 at 1.76–1.84×);
  - sampled requests under concurrency: a token the device sampler draws runs as its own per-sequence call, and a
    sampler that needs full logits joins steps;
  - more than 4 clients.
- **Candidates next, not registered:** a hybrid B = 2 (per-row GEMVs for the int4 projections, the batched LM head); an
  encode-ahead executor for steps (host time was 2.7 ms of a 25 ms B = 4 step); CUDA, only on its own measurement.
