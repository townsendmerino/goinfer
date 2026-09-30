# `--spec ngram` on CUDA: chat clears the candidate's own rule (S 1.25×, L 0.79×) — copy blows past it (S 2.10×, L 1.54×, spec beats batching even at 4 clients, a separate owner question) (graded 2026-09-29, run 2026-09-29 01:xx PDT)

MC4's CUDA premise measurement, [`docs/prompts/nobara-mc4-spec-alone-cuda-2026-09.md`](../prompts/nobara-mc4-spec-alone-cuda-2026-09.md)
(pushed `b6f732ca`), a copy of the 2026-09-27 Metal design
([`spec-vs-batching-metal-2026-09-27.md`](spec-vs-batching-metal-2026-09-27.md)), under the same
pre-registered rule, unchanged. The run itself completed overnight 2026-09-28→29 (queued, not run by
day); this is the morning grading the run's own note asked for.

**Result, by the pre-registered rule (the 1.5B, the graded cells): a split verdict.**
- **chat clears the candidate's own condition:** S = 1.253× (a lone request), L = 0.790× (under 4-client
  load spec falls behind batching) — `S > 1.05` with `L < 1.0`. **"Speculate when alone, batch under
  load" applies on CUDA for chat traffic**, and is graded separately once built.
- **copy overshoots it entirely:** S = 2.104×, and **L = 1.540× — spec beats MC3's batching even at 4
  clients.** The rule names this case explicitly: `L ≥ 1.0` is not this candidate's question, it is
  MC4's own "spec inside a batch" trigger, and it goes to the owner.
- **Identity: perfect.** Every reply's hash, spec vs batch, every turn, every workload, every client
  count: 0 differences (0/24 per chat pair, 0/8 and 0/2 per copy pair, 1.5B and 7B). Item 30's
  trailing-token fix holds under load on CUDA — unlike Metal's 2026-09-27 record, where later turns
  diverged.
- **4-client p99 (chat):** spec's tail is 1.39–1.41× batch's (2204–2219 ms vs 1570–1581 ms) — the cost of
  running every client through one-at-a-time speculation instead of MC3's batching, when nobody switches.

## 1. Setup

As registered: nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07.
- One binary, `serve-cuda` at `2d052fc4` (item 30's two-turn fix + the MC3-CUDA batched-head lever both
  already in), two arms, a fresh server per cell:
  - *batch*: serve's defaults, MC3 on;
  - *spec*: the same plus `-spec ngram` ("one generation at a time").
- Two workloads, qwen2.5-coder-1.5b-instruct Q4_K_M int4 from `~/models`, greedy:
  - *copy*, [`scripts/bench_spec_copy.py`](../../scripts/bench_spec_copy.py) `--backend cuda`: a ~1000-token
    section of `decoder/model.go` at `cc5f8c2c`, 256 tokens, 2 requests per client;
  - *chat*: W7 as graded for MC3, [`scripts/bench_w7_plain.py`](../../scripts/bench_w7_plain.py)
    `--backend cuda --engines goinfer`, 6 turns × 128 tokens, `--fixed-nonce`.
- 4, then 1 clients per workload, batch/spec × 3 pairs in the order batch spec spec batch batch spec,
  idle-gated per cell (CUDA's own gate: load1 ≤ 2.0 and GPU memory/compute-process count back to the
  run's own baseline).
- Reported only: qwen2.5-7b-instruct Q4_K_M, copy workload, 1 and 4 clients, 3 pairs, complete (no 413s
  — CUDA's prefill-memory guard did not repeat Metal's 2026-09-27 finding here).
- Every server's log line reads `decode path: cuda-resident (int4)` — no cell is void.

Raw: [`copy-15b.json`](spec-vs-batching-cuda-2026-09-28/copy-15b.json),
[`chat-15b.json`](spec-vs-batching-cuda-2026-09-28/chat-15b.json),
[`copy-7b.json`](spec-vs-batching-cuda-2026-09-28/copy-7b.json),
[`servers.log`](spec-vs-batching-cuda-2026-09-28/servers.log),
[`run.sh`](spec-vs-batching-cuda-2026-09-28/run.sh). (Files live under the 2026-09-28 run directory —
the run started 2026-09-28 evening and finished after midnight PDT; this record is dated by the grading.)

## 2. Results (1.5B, the graded cells)

| workload | clients | batch tok/s | spec tok/s | spec ÷ batch, per pair | median | p99 request, spec ÷ batch |
|---|---:|---:|---:|---|---:|---:|
| copy | 1 | 185.2–185.5 | 390.0–390.8 | 2.111 · 2.103 · 2.104 | **S = 2.104×** | — |
| copy | 4 | 262.9–265.1 | 407.0–409.9 | 1.559 · 1.540 · 1.537 | **L = 1.540×** | — |
| chat | 1 | 227.5–228.0 | 285.1–287.2 | 1.251 · 1.261 · 1.253 | **S = 1.253×** | — |
| chat | 4 | 357.2–357.9 | 282.0–282.8 | 0.790 · 0.792 · 0.789 | **L = 0.790×** | 1.39–1.41× |

- copy's aggregate is dominated by n-gram's near-perfect accept rate on a verbatim-copy prompt (the
  traffic `-spec ngram` exists for): even at 4 clients competing for one exclusive decode slot, spec's
  raw throughput (407–410 tok/s) still beats MC3 batching four independent int4 decodes (263–265 tok/s).
- chat is the workload the candidate was designed around: a lone request gains a real 25% from
  speculating, and under load speculating alone falls to 79% of what batching delivers to four clients
  at once — exactly the shape "speculate when alone, batch under load" is meant to fix.
- Aggregates include each request's prefill; copy's prompts are larger (~1000 tokens) than chat's later
  turns are new (chat reuses most of its prefix via `PrefillReused`), so the two workloads are not
  directly comparable to each other, only spec-to-batch within themselves.

## 3. The decision

By the registered rule, applied per workload (the 1.5B, the only graded model):

- **chat: `S > 1.05` (1.253×) with `L < 1.0` (0.790×) → the candidate applies on CUDA.** "Speculate when
  alone, batch under load" is worth building for chat-shaped traffic here, the same conclusion Metal
  reached for the opposite reason (Metal had no S to keep at all). Its build gets its own grading on
  nobara, re-running the Metal-registered gates (identity every turn; alone copy ≥ 1.25× and chat ≥
  0.97×; 4 clients ≥ 0.97× of batch; staggered aggregate ≥ 1.00× with p99 turn ≤ 1.10×) against
  whichever CUDA baseline that build is measured from.
- **copy: `L ≥ 1.0` (1.540×) → not this candidate's question.** The rule names this exact case: spec
  beats batching even at 4 clients, which is MC4's own "spec inside a batch" trigger, a different and
  larger feature (running speculation *inside* MC3's batched decode, not switching between the two).
  That goes to the owner rather than into "speculate when alone, batch under load."
- **Why CUDA and Metal disagree on copy's magnitude:** CUDA's verify is comparatively cheap (θ 0.155–0.251,
  a 2.14× single-stream win on a verbatim-copy prompt,
  [`spec-decode-lane-2026-09-21.md`](spec-decode-lane-2026-09-21.md)) and MC3's batched decode on CUDA,
  while real, splits four int4 GEMVs' worth of memory bandwidth four ways — batching's per-client share
  at 4 clients (263–265 ÷ 4 ≈ 66 tok/s/client) is well under spec's single-stream rate on this content
  (~130–137 tok/s/client-equivalent, 390–410 ÷ 3 requests in flight). Metal's batched decode did not
  give up nearly as much per client, so its L never crossed 1.0. This is a reading, not a re-measurement
  of Metal.
- **Identity holds under load, unlike Metal's 2026-09-27 record**, where every later chat turn diverged
  between spec and batch. Item 30's fix (both turns' trailing token forwarded through the target) is
  the difference: that fix did not exist when Metal's record was taken. This CUDA run is direct evidence
  the fix generalizes under concurrent load, not just the two-turn unit tests it was built against.

## 4. The 7B, reported only

| clients | batch tok/s | spec tok/s | spec ÷ batch, per pair | median |
|---:|---:|---:|---|---:|
| 1 | 58.5 | 121.4–121.9 | 2.075 · 2.077 · 2.083 | 2.077× |
| 4 | 93.0–93.8 | 118.4–119.7 | 1.274 · 1.276 · 1.284 | 1.276× |

Same shape as the 1.5B copy result, smaller magnitude: spec still beats 4-client batching (1.276×), and
identity is perfect here too (0/8 and 0/2 mismatches). Not graded — reported per the prompt's own scope.

## 5. Next

- **Update `task-concurrency-2026-09.md`'s MC4 candidate CUDA bullet** with this result and a link,
  replacing "it gets its own grading on nobara."
- **File copy's `L ≥ 1.0` finding as its own item** ("spec inside a batch" on CUDA) rather than folding
  it into this candidate — it is a different mechanism (speculating concurrently with batched decode,
  not switching between the two modes) and needs its own design before any build.
- **Build "speculate when alone, batch under load" for chat**, gated on nobara per §3's re-run list,
  before it ships.
