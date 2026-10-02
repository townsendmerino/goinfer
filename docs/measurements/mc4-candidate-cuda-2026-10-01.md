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

---

# RESULT (graded 2026-10-02 morning; run 2026-10-01 night, nobara-pc, job `mc4-candidate-cuda`, 21 min)

**IDENTITY FAILED, so by the registered rule this is a bug, not a result: gates 2–4 are recorded below and NOT read.** The rule
in this record's "Decision rule": *"Identity fails: a bug, not a result. Stop, find it, fix it; nothing else is read."*

Provenance: nobara-pc, RTX 2070 SUPER, driver 595.91.07, `serve-cuda` at `efaae8a6` (pinned binary), qwen2.5-coder-1.5b-instruct
Q4_K_M int4 (7B reported), from `~/models`; 3 rotated rounds × 3 arms, 72 cells, all present (`raw/`); every cell idle-gated
(load1 ≤ 2.0; the log shows several waits between 4-client cells and none that timed out); greedy. `gates.py` unchanged since the
registration. Raw: `mc4-candidate-cuda-2026-10-01/raw/`, log `~/goinfer-logs/night/runs/2026-10-01/mc4-candidate-cuda.log`.

## Gate 1: identity (hard)

Replies compared by hash with the first batch cell of the same client count. Counts are differing replies out of the arm's replies.

| workload, clients | candidate | spec-exclusive | batch vs batch |
|---|---|---|---|
| copy 1.5B, 1 | 0 / 6 | 0 / 6 | 0 / 6 |
| copy 1.5B, 4 | **21 / 24** | 1 / 24 | 1 / 24 |
| chat 1.5B, 1 | 0 / 18 | 0 / 18 | 0 / 18 |
| chat 1.5B, 4 | **72 / 72** | 0 / 72 | 0 / 72 |
| staggered copy 1.5B | **69 / 90** | 0 / 90 | 0 / 90 |
| copy 7B, 1 (reported) | 0 / 6 | 0 / 6 | 0 / 6 |
| copy 7B, 4 (reported) | **22 / 24** | 0 / 24 | 0 / 24 |

- **Alone, the candidate is identical** to batch (every 1-client cell, both models). **Under concurrent load it differs from batch on
  nearly every reply** (all 72 chat replies at 4 clients), while the spec-exclusive arm is identical to batch at the same load.
  So this is not general numeric noise: it is specific to `-spec-adaptive` once a generation switches.
- Baseline noise exists but is tiny: one reply in the copy-4 cells differs in a batch round (and one in a spec-exclusive round) out of 72.
- **What it does and does not show.** It shows the candidate's output is not plain decode's under load, which is the lossless claim
  the candidate rests on (`6e9fcb99`: "a switch at a round boundary changes only who runs the next committed token"). It does **not**
  say where: only content hashes were recorded, not text, so the first divergent token is unknown. The fake-resident stress test
  (200/200 under `-race`) checked scheduling, not numerics, which is consistent with this surviving it.

## Gates 2–4 (recorded, NOT graded: identity failed)

For the record only, candidate ÷ batch, paired median of 3: 1 client copy 2.113× and chat 1.259×; 4 clients copy 0.817× and chat
0.692×; staggered 0.963× aggregate with p99 2.540×; 7B copy 2.081× alone and 0.637× at 4 clients. The 2026-09-29 exploratory smoke
had projected the copy miss (about 0.78×); that projection was for the speed gates, which this identity failure supersedes.

## What happens next

- Do not recommend, default, or document `-spec-adaptive` as safe. The task doc's MC4 candidate entry carries this result.
- **To find the bug:** rerun the 4-client chat workload with the candidate and with batch, keep reply TEXT (not hashes), and find the
  first divergent token of one conversation; then bisect on whether the divergence starts at a switch boundary. A likely place to
  look is a numeric difference between the CUDA spec verify path and the MC3 batched step (the lossless argument assumes they are
  bit-identical for the same rows), or a position/KV off-by-one at the switch. This is a hypothesis, not a finding.
- The Metal run of the same gates (the Mac's night queue) should not be queued until this is understood.

## Text-level follow-up (2026-10-02 morning, exploratory diagnosis, not a graded measurement)

Reproduced with reply TEXT kept (`text-repro/`, `text_repro.py` and `text_repro2.py`), on the same pinned `serve-cuda` at `efaae8a6`, 1.5B int4,
greedy, fresh server per arm, 4 concurrent clients unless stated.

- **The candidate's output is garbage from the first 1–3 tokens, not a late near-tie flip.** The first token is right and the next ones
  are not: `"```000.4.0.0.0…"`, `"TokenTokenToken…"`, `"If0.5*10000000000…"`, `"Sure0\nHello, how can I assist you"` (chat), and the
  same on copy (candidate 0/8 identical to batch). Completion lengths are wrong too (1, 11, 27 and 157 tokens where batch gives 55–256).
- **Controls:** batch run twice is identical (copy 8/8, chat 4/4); `-spec ngram` alone is identical on chat (4/4) and on copy 7/8 (one reply
  diverges late, at char 540 of 693, consistent with the small baseline in the graded run).
- **Bisection:**
  - 2 clients is already garbage (0/2 identical), so it needs only two concurrent generations;
  - `GOINFER_SPEC_ADAPTIVE_NEVER_YIELD=1` is garbage too, so it is **not** the yield policy (consistent with the 2026-09-29 note that the
    cost is concurrent admission itself);
  - `-spec ngram -spec-adaptive -max-concurrent=1` is identical to batch (4/4 and 2/2), so it is the adaptive path running with other
    generations in flight.
- **One hypothesis killed:** the CUDA resident's batched verify (`ForwardN`) on a non-zero KV slot after another slot ran a different
  prompt is bit-identical to a single-slot reference (cosine 1.000000000 both directions, llama-tiny, int4; scratch test, not committed).
  So the slot-write itself is fine in isolation.
- **Not found yet.** What remains: the interleaving between generations (a generation's prefill, MC3 step or commit running against
  the shared resident between another generation's exclusive sections: a state the resident keeps at resident level rather than per
  slot, such as a position or token record), or the seed/first-round handoff. The next step is an in-process CUDA test with two
  concurrent adaptive generations on a tiny fixture (the existing stress test uses the fake resident, which has no numerics), then bisect there.

## ROOT CAUSE AND FIX (2026-10-02, commit `2f685d7e`)

**Cause.** An adaptive round claimed the resident through `claimExclusive`, which sets only `resBusy`, and then ran its verify
(`residentBind(slot)` then `ForwardN`) directly. Every other resident-touching section of a generation, the slot pick
(`residentAcquireSlot`), the prefill and the commit, runs under the batcher's **`busy`** flag (`exclusive` / `prefillExclusive`) and never
reads `resBusy`. So with two or more generations in flight, one generation's bind-then-prefill interleaved with another's
bind-then-verify, and each wrote the other's KV slot: the first token already wrong, then garbage. Alone, or with `-max-concurrent=1`,
nothing interleaves, which is why every 1-client cell was identical. The original design (this doc's registration of the candidate)
said "each round runs in `exclusive`"; the implementation used a bare CAS instead, and `6e9fcb99`'s own commit notes record a related
rebind race found the same way, fixed for the prefill but not for the verify.

**Why nothing caught it.** The 200/200 `-race` stress test (`decoder/spec_adaptive_switch_test.go`) runs on `mc3Fake`, which checks that
resident access is *guarded* but has no numerics and no real KV slots, so a slot written by the wrong generation is invisible to it.

**Fix.** The round's verify runs inside `batcher.exclusive` (decoder/spec_ngram.go), so it holds `busy` like every other resident touch.

**Evidence.**
- In-process on a real CUDA resident (`cuda/spec_adaptive_concurrent_test.go`: 4 concurrent adaptive generations against their lone
  references, llama-tiny int4, 6 repeats): **23 of 24 differ without the fix, 0 of 24 with it**, and 5 further runs of the scratch version
  were 0 of 24 each.
- Served on the real 1.5B (text-level repro, `text-repro/`, same arms as above, 4 clients): candidate **0/8 → 8/8** identical to batch on
  copy and **0/4 → 4/4** on chat; batch twice and spec-exclusive also identical.
- Bisection that located it: 2 clients already fail; `GOINFER_SPEC_ADAPTIVE_NEVER_YIELD=1` still fails; `-max-concurrent=1` is identical;
  the batched verify on a non-zero slot is bit-identical to a single-slot reference (so it was not the slot write itself).
- Decoder adaptive/MC3 tests clean under `-race`; the 64 parity goldens pass with 0 `deps_hash` lines moved; tagged CUDA suite 190 pass.

## RERUN, registered 2026-10-02 before it ran

The first run's identity failure was a bug, so its speed numbers are not read (they stay recorded above). The fix changes the build, so the
graded run is repeated: **same gates, same decision rule, same 72 cells**, with one change, the binary: `serve-cuda` at `2f685d7e`
(`~/goinfer-bench/mc4-candidate-cuda-2026-10-01/serve-cuda-2f685d7e`, built once from the committed fix). Harness `run-rerun.sh`
(`run.sh` with the binary and the output directory changed, `raw-rerun/`); grade with
`python3 docs/measurements/mc4-candidate-cuda-2026-10-01/gates.py docs/measurements/mc4-candidate-cuda-2026-10-01/raw-rerun`.

- **Identity is gate 1 again and is hard.** A differing reply is a bug, not a result, exactly as registered.
- **No speed projection is carried forward.** The "copy at 4 clients will miss, ~0.78×" projection above came from the 2026-09-29
  exploratory smoke, which ran on the buggy build (its replies were corrupted and short, so its throughput meant nothing). The projection
  is withdrawn. This rerun is the first valid speed measurement of the candidate under load.
- The decision rule is unchanged, including the copy-only-miss-goes-to-the-owner clause.
- Estimate 30 min (the first run took 21). Queued on nobara's night queue; the owner starts the queue.

## RERUN RESULT (graded 2026-10-02 afternoon; run 11:55-12:14 PDT, nobara-pc, job `mc4-candidate-cuda-rerun`, 19 min against 30 estimated)

**Gate 1, identity: FAIL. By the registered rule that is a bug, not a result: the speed gates below are recorded and NOT read as a verdict.** Grader output verbatim:
[`mc4-candidate-cuda-2026-10-01/raw-rerun/gates-output.txt`](mc4-candidate-cuda-2026-10-01/raw-rerun/gates-output.txt); raw cells beside it; the runner's log is `raw-rerun/night-runner.log`.

Provenance: `serve-cuda` at `2f685d7e` (the pinned binary, built once), RTX 2070 SUPER, driver 595.91.07, the 72 registered cells (a fresh server per cell, three rounds, arm order rotated), all from `~/models`.
**The box was not quiet:** it ran by day (the queue was started at 11:54) and the CUDA idle gate (load1 <= 2.0, GPU baseline) waited 15 times and never gave up; load1 read 1.5 to 2.0 at the starts. No HTTP errors, no panics in the server logs.

### Identity, in detail (this is the finding)

| cell | replies compared | candidate differing | spec-exclusive differing | batch against batch differing |
|---|---|---|---|---|
| copy 1.5B, 1 client | 18 | 0 | 0 | 0 |
| copy 1.5B, 4 clients | 72 | **2** | 2 | **1** |
| chat 1.5B, 1 client | 54 | 0 | 0 | 0 |
| chat 1.5B, 4 clients | 216 | 0 | 0 | 0 |
| staggered copy 1.5B | 270 | **3** | 0 | 0 |
| copy 7B, 1 and 4 clients (reported) | 18 + 72 | 0 | 0 | 0 |

Two different things are in that table, and they should not be read together:
- **Copy at 4 clients: not the candidate's.** Reply 0 of the cell has two variants. Batch round 1 and 3, candidate round 3 and spec-exclusive round 3 produce one; **batch round 2 itself**, candidate rounds 1 and 2 and spec-exclusive rounds 1
  and 2 produce the other. The reference arm differs from itself across rounds, so this is the engine's own run-to-run variation under four-way concurrency, present in all three arms. The registration did not anticipate it ("b's rounds also agree with each
  other" is part of the gate), the gate counts it, and **the bar is not moved here**: a bar is moved only with a mechanism, and none is established for this variation.
- **Staggered: the candidate's, and reproducible.** The candidate differs from batch at reply index 21 of 30 in **all three rounds**; spec-exclusive and batch never differ. The same reply, three times out of three, that only the
  adaptive policy changes: a deterministic candidate-specific divergence, which is what an identity gate exists to catch. By the grader's order (client by client, rounds 12, 9, 6, 3) index 21 is the first reply of the third client,
  which joins last but one; that mapping is not verified. **Not root-caused.** What is NOT known: whether it is a defect (a mode switch between a speculative round and a batched step leaving state that differs by more than numerics)
  or a legitimate numeric difference between the two modes that the first fix exposed.

### Speed numbers (recorded, NOT graded: identity failed)

Candidate / batch, paired, median of three; spec-exclusive shown for reference.

| cell | candidate / batch | pairs | p99 candidate / batch | spec-excl / batch | candidate / spec-excl |
|---|---|---|---|---|---|
| copy 1.5B, 1 client | 2.114 | 2.114 2.114 2.118 | 0.511 | 2.110 | 1.004 |
| chat 1.5B, 1 client | 1.257 | 1.256 1.257 1.262 | 0.863 | 1.258 | 1.002 |
| copy 1.5B, 4 clients | 1.265 | 1.302 1.265 1.213 | 1.361 | 1.565 | 0.808 |
| chat 1.5B, 4 clients | **0.850** | 0.851 0.847 0.850 | 1.313 | 0.791 | 1.073 |
| staggered copy 1.5B | 1.454 | 1.425 1.454 1.459 | **1.860** | 1.720 | 0.838 |
| copy 7B, 1 client | 2.080 | 2.078 2.083 2.080 | 0.483 | 2.082 | 0.999 |
| copy 7B, 4 clients | 1.170 | 1.172 1.166 1.170 | 1.397 | 1.285 | 0.909 |

Recorded without being read as a verdict: the grader flags chat at 4 clients (0.850 against the 0.97 bar) and the staggered p99 (1.860 against 1.10). The three chat pairs agree to 0.4 points, so
that number is stable. Had identity passed, a chat hard-gate failure alone would park the candidate under the decision rule. The projection withdrawn above ("copy at 4 clients ~0.78x") is replaced by a valid but ungraded
number: candidate / spec-exclusive 0.808 on copy at 4 clients, and 1.265x batch.

### Decision

Per the registered rule, **identity fails: a bug, not a result; stop and find it.** Nothing else is read. `-spec-adaptive` stays opt-in and experimental, and **must not be described as lossless**: it is not bit-identical
to batch in the staggered workload (one reproducible reply). Next: root-cause the staggered reply-21 divergence (a text-level comparison against batch, which the raw files do not allow: they keep hashes, not replies), then
fix or explain it, then the identity gate can be re-graded; separately, the copy-at-4-clients batch-against-batch variation is an engine question that predates this candidate.

