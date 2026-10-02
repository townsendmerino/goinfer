# MC4 candidate "speculate when alone, batch under load" (`-spec-adaptive`) on CUDA — graded run, PRE-REGISTERED 2026-10-01 (21:xx PDT), before any graded timing

Status: **registered, not yet run.** Queued on nobara's night queue (`mc4-candidate-cuda`). The result goes in a
section below this one, dated by the grading.

This is the build-then-grade step that
[`spec-vs-batching-cuda-2026-09-29.md`](spec-vs-batching-cuda-2026-09-29.md) §3 asked for ("its build gets its own
grading on nobara, re-running the Metal-registered gates"). The Metal-registered gates are in
[`../tasks/parked/task-concurrency-2026-09.md`](../tasks/parked/task-concurrency-2026-09.md), MC4, "MC4 candidate:
speculate when alone, batch under load". The build is `6e9fcb99` (opt-in `-spec-adaptive`).

## Setup

- nobara-pc, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07 (anchor), compute mode Default.
- **One binary**, `serve-cuda` at `efaae8a6` (built once with `CGO_ENABLED=0 go -C cuda build -tags cuda`, named by its
  hash, kept at `~/goinfer-bench/mc4-candidate-cuda-2026-10-01/`). It contains `6e9fcb99`, item 30's trailing-token fix
  and the MC3-CUDA batched head.
- **Three arms**, a fresh server per cell (`run.sh`, this directory):
  - **b — batch:** serve's defaults, MC3 on. The do-nothing arm; every ratio is against it.
  - **s — spec-exclusive:** `-spec ngram` (today's: one generation at a time).
  - **c — candidate:** `-spec ngram -spec-adaptive`.
- **Models:** qwen2.5-coder-1.5b-instruct Q4_K_M int4 (graded) and qwen2.5-7b-instruct Q4_K_M (reported), both from
  `~/models` (NVMe), never `/srv/models`. Greedy.
- **Workloads:**
  - *copy*: `scripts/bench_spec_copy.py --backend cuda`, as in the premise run (a ~1000-token section of `decoder/model.go`
    at `cc5f8c2c`, 256 tokens, 2 requests per client), at **4 and 1 clients**.
  - *chat*: W7 as graded for MC3 (`scripts/bench_w7_plain.py --engines goinfer --backend cuda --fixed-nonce`, 6 turns ×
    128 tokens), at **4 and 1 clients**.
  - *staggered* (copy requests, 1.5B only): `bench_spec_copy.py --rounds-list 12,9,6,3 --stagger-s 2`. Four clients join at
    0 / 2 / 4 / 6 s and send 12 / 9 / 6 / 3 requests, so they leave at different times and the cell has stretches with one
    generation decoding and stretches with several, the case that exercises the switch.
    - **This is CUDA-scaled, not the Metal text's 0 / 5 / 10 / 15 s.** In an exploratory smoke on this box a CUDA copy request
      takes ~0.6 s under spec, so a 5 s stagger leaves almost no overlap (client 0 finished by 6 s). The 2 s stagger overlaps
      all four clients between 6 s and ~17 s, with ~2 s alone at the start and ~4 s at the tail.
    - `--rounds-list` and `--stagger-s` are new options on `bench_spec_copy.py`, default off, so the old shape is unchanged.
  - The 7B is reported on copy at 4 and 1 clients only.
- **Rounds:** 3, interleaved, the arm order rotated each round (b s c / s c b / c b s), so no arm is always first or last. A
  "pair" is the three arms of one round. Every cell is idle-gated (`/proc/loadavg` load1 ≤ 2.0, GPU compute-process count
  and memory at the run's own baseline). 72 cells.
- **Metrics**, paired per round, median of 3: aggregate tok/s (completion tokens ÷ wall), and p99 request latency (copy and
  staggered) or turn latency (chat), nearest rank over the cell's own requests. `gates.py` computes them; its arithmetic
  was checked against the 2026-09-28 raw files (it reproduces S 2.104× / 1.253×, L 1.540× / 0.790×, 7B 2.077× / 1.276×).

## Gates (the Metal-registered ones, bars unchanged)

All compare **candidate ÷ batch** on the 1.5B:

1. **Identity — hard, stops the item.** Every reply's hash equals batch's, every turn, every workload, every client count.
   An HTTP error in a graded cell is a miss too.
2. **Alone (1 client):** copy ≥ 1.25×; chat ≥ 0.97× (hard for chat).
3. **At 4 clients: ≥ 0.97× on both copy and chat (hard).** It must not give up batching.
4. **Staggered:** aggregate ≥ 1.00× **and** p99 request latency ≤ 1.10× batch's.

## Decision rule

- **Identity fails:** a bug, not a result. Stop, find it, fix it; nothing else is read.
- **Chat gates 2 and 3 and the staggered gate all pass, and copy gate 3 passes:** `-spec ngram -spec-adaptive` is safe
  under load on CUDA, and `docs/server.md` says so. Whether it becomes a default is a separate owner decision.
- **A chat hard gate (2 or 3) fails:** parked with the numbers.
- **Copy-only miss (copy gate 3 and/or the staggered gate fails while every chat gate and identity pass): AMBIGUOUS, goes
  to the owner. It is not auto-parked and not shipped as a default.** This is the one place this registration departs from
  the Metal text ("any hard gate fails: parked"), disclosed before the run, because the premise run already split
  the verdict by workload: copy was `L ≥ 1.0` there, a different question from this candidate's, and a per-workload
  result is what is useful to decide on. The options it would give the owner are the flag as chat-shaped-traffic-only with
  the copy caveat in `server.md`, or leaving it parked.
- **Copy alone between 1.03× and 1.25×:** the owner's call.

## Projections, written before the run (so the miss zone is named)

- **Copy at 4 clients is expected to MISS gate 3.** The 2026-09-29 exploratory smoke (single run per arm, not paired,
  [`spec-vs-batching-cuda-2026-09-29-inside-batch-smoke/smoke.log`](spec-vs-batching-cuda-2026-09-29-inside-batch-smoke/smoke.log))
  read the candidate at 206.0 tok/s on 4-client copy against batch's 263–265 from the premise run, roughly 0.78×. The
  mechanism the investigation gave is concurrent admission itself, which no per-round policy removes.
- **Chat is the workload the candidate was built for.** The premise run's chat: spec-exclusive 0.790× under load, alone 1.253×.
  A candidate that batches under load and speculates alone is expected to sit near 1.0× at 4 clients and near 1.25× alone. A
  miss on chat is the surprise that would matter.
- **Staggered: no projection.** It is the first measurement of that shape.
- So the likeliest outcome by this rule is the copy-only miss: **ambiguous → owner**, with chat clearing.

## Not covered, disclosed

- **Step 0 of the Metal registration was not run on CUDA.** It asked for a by-day in-process check on a real resident that a
  generation forced to switch spec → batch → spec emits plain decode's ids and a bit-identical KV, and that a lone
  speculating generation keeps ≥ 0.97× of today's exclusive spec rate. What exists is the fake-resident stress test
  (`decoder/spec_adaptive_switch_test.go`, 200/200 under `-race`). This run's identity gate covers the property end to end
  on a real resident and across turns, since a cache one token short changes every later turn, but it does not compare KV
  bytes, and it does not measure the alone cost except through gate 2's copy and chat ratios (which include it).
- The staggered gate's p99 is over 30 requests per cell, so it is the slowest request or close to it.
- The bench scripts are read from the working tree when the job runs; the run header records the tree's hash.

## Cost

72 cells. The premise run (36 cells) took 9 min 33 s from start to DONE, so this is ~20–25 min of cell time; the staggered
and 4-client chat cells are the longer ones. Estimate **45 min** for the queue (generous). Raw output lands in `mc4-candidate-cuda-2026-10-01/raw/`,
the job log in `~/goinfer-logs/night/runs/<date>/mc4-candidate-cuda.log`. Grade with
`python3 docs/measurements/mc4-candidate-cuda-2026-10-01/gates.py`.

The Metal run of the same gates (`serve-metal`, the Mac's idle gate, ~60 min on the Mac's night queue) is separate and not
queued from here.
