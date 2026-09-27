# `--spec ngram` on Metal: no single-stream win (0.98× copy-heavy, 0.93× chat), and under 4-client load it gives up MC3's batching (0.61× / 0.49×); multi-turn replies differ from plain decode from the second turn (2026-09-27)

MC4's spec item in [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md). Its trigger reads "MC3
shipped and spec's single-stream win is still larger than batching's per-sequence share". A model served with
`--spec ngram` takes the Metal resident exclusively, so under load it gives up MC3's batching. This measures what a
Metal user gains or loses by turning spec on.

**Result, by the pre-registered rule: S ≤ 1.05 on both workloads, so the spec item stays parked for Metal.**
- **n-gram speculation gives a lone request nothing on Metal.**
  - Even on the copy-heavy workload it exists for, a lone request runs at 0.979× plain decode. On chat it runs at
    0.928×.
  - Its verify is production's decode kernels run layer-major, so each extra verified row costs ~0.7–0.96 of a token
    (`thetaFor("metal")` = 0.96). There is little to amortise.
- **Under 4-client load it costs a great deal.** Spec runs one generation at a time where batching serves four: 0.610×
  the batched aggregate on copy and 0.489× on chat, with p99 per request 1.64× / 1.98×.
- **A second finding, about correctness.** With `--spec ngram`, every conversation's first reply matched plain decode
  exactly, and every later turn's reply differed. The single-turn copy requests matched on all 30. The flag's help and
  `ARCHITECTURE.md` promise greedy output identical to plain decode. The cause is not established (§4).
- **A third finding, unrelated to spec.** On the 7B, MC3's batch arm had 3 of 6 cells refused with HTTP 413 by serve's
  prefill-memory guard. The guard splits its margin among the running generations, but under MC3 prefills run one at a
  time (§5).

## 1. Setup

As registered (`b11bcfc3`, committed and pushed before any timing):
- M1 Pro 16 GB, macOS 26.6.2.
- One binary, `serve-metal` at `cc5f8c2c`, two arms, a fresh server per cell:
  - *batch* runs serve's defaults: MC3 batching, and a lone request runs production's own path;
  - *spec* adds `-spec ngram`. Its banner reads "one generation at a time — … no speculation or adapter".
- Two workloads, qwen2.5-coder-1.5b int4 from `~/models`, greedy:
  - *copy*, [`scripts/bench_spec_copy.py`](../../scripts/bench_spec_copy.py): each request hands back a ~960-token
    section of `decoder/model.go` at `cc5f8c2c` verbatim, 256 tokens, 2 requests per client in turn;
  - *chat*: the W7 workload as graded for MC3 (6 turns × 128 tokens, `--fixed-nonce`).
- 4, then 1 clients per workload, batch/spec × 3 pairs in the order batch spec spec batch batch spec, idle-gated per
  cell (load1 ≤ 2.0). The runs went 09:25–10:05 PDT (16:25–17:05 UTC).
- Reported only: the copy workload on qwen2.5-7b-instruct, 3 pairs at 4 and 1 clients.
  - It is incomplete: 3 of its 6 4-client batch cells hit §5's 413, and the 1-client cells never ran.
  - From 10:05 the idle gate held (load1 2.6–10, interactive editor use), and the run was stopped by hand at 10:16
    with only those reported cells left.

Raw: [`copy-15b.json`](spec-vs-batching-metal-2026-09-27/copy-15b.json), [`chat-15b.json`](spec-vs-batching-metal-2026-09-27/chat-15b.json),
[`copy-7b.json`](spec-vs-batching-metal-2026-09-27/copy-7b.json), [`run.log`](spec-vs-batching-metal-2026-09-27/run.log),
[`servers.log`](spec-vs-batching-metal-2026-09-27/servers.log), [`run.sh`](spec-vs-batching-metal-2026-09-27/run.sh).
Metrics by [`gates.py`](spec-vs-batching-metal-2026-09-27/gates.py), output in
[`gates-output.txt`](spec-vs-batching-metal-2026-09-27/gates-output.txt).

## 2. Results (1.5B, the graded cells)

| workload | clients | batch tok/s | spec tok/s | spec ÷ batch, per pair | median | p99 request, spec ÷ batch |
|---|---:|---:|---:|---|---:|---:|
| copy | 1 | 53.0–53.2 | 51.8–52.0 | 0.973 · 0.979 · 0.979 | **S = 0.979×** | 1.029× |
| copy | 4 | 81.6–83.6 | 50.2–51.0 | 0.615 · 0.607 · 0.610 | **L = 0.610×** | 1.640× |
| chat | 1 | 73.9–74.7 | 67.6–71.3 | 0.915 · 0.954 · 0.928 | **S = 0.928×** | 1.140× |
| chat | 4 | 143.8–148.7 | 69.8–72.8 | 0.489 · 0.491 · 0.485 | **L = 0.489×** | 1.983× |

- Under load, spec's aggregate is its single-stream rate: 50–51 tok/s at 4 copy clients against 52 alone, and 70–73 at
  4 chat clients against 68–71 alone. Batching turns the same load into 1.57× (copy) and 1.99× (chat) a lone request.
- Aggregates include each request's prefill, a larger share of the copy workload (~960 prompt tokens per request).

## 3. The decision

By the registered rule: S ≤ 1.05 on both workloads, so spec has no single-stream win on Metal to keep.
- MC4's spec-in-a-batch item stays parked for Metal.
- The docs now say that `--spec ngram` on Metal gives up batching under load for no gain.
- "Speculate when alone, batch under load" is not registered: there is no alone-win for it to keep.
- On CUDA the verify is cheap (θ 0.155–0.251, 2.14× on a verbatim-copy prompt in
  [`spec-decode-lane-2026-09-21.md`](spec-decode-lane-2026-09-21.md)). CUDA has no batching yet (MC1 on CUDA is only
  now starting), so the question does not arise there.

## 4. Spec output differs from plain decode after the first turn (reported, not gated)

Every reply's hash, spec vs batch at the same client count and position:

| workload | clients | replies equal | pattern |
|---|---:|---:|---|
| copy | 1 and 4 | **30 / 30** | single-turn requests |
| chat | 1 | 3 / 18 | turn 0 equal in every cell; turns 1–5 differ |
| chat | 4 | 15 / 72 | turn 0 equal for every client in every pair; one later turn equal by coincidence |

- Each arm is repeatable: its three cells give the same replies. So both arms are deterministic, and they disagree.
- The prompts are the same length turn by turn. But under spec every later turn reuses one token less of the history
  (156, 310, 465 … against 157, 311, 466 …), so it prefills one more token.
- **Not established:** whether the divergence comes from that one-token reuse gap, or from the KV that spec's verify
  pass (`ForwardBatch`) writes and the next turn reuses.
  - The spec track's own records say Metal's verify path is not bit-identical to decode
    ([`docs/spec/08-dspark-dflash.md`](../spec/08-dspark-dflash.md) § "Metal — verify curve measured", written
    before the layer-major rework).
  - The one-token re-prefill runs a sub-64-token suffix through the sequential path, i.e. decode's own kernels.
  - A same-conversation test through `Model.Generate` (spec against plain, two turns, logits compared per step) would
    separate them.
- **What it contradicts:** `serve -spec`'s help ("output is identical (greedy bit-exact …)") and `ARCHITECTURE.md`
  ("greedy: identical to plain decode"). On Metal, multi-turn, that is not what was measured.

## 5. The 7B, and a 413 from the prefill-memory guard (reported)

| clients | pairs run | spec ÷ batch | replies equal |
|---:|---|---|---:|
| 4 | pair 2 only (batch 23.2, spec 16.1 tok/s) | 0.695× | 8 / 8 |
| 1 | none (the run was stopped) | — | — |

The 4-client batch cells of pairs 1 and 3, both at 09:39–10:02, lost a request to **HTTP 413** (`prefillMemoryError`).
The spec cells never did.
- The server banner that run read "0.7 GB left for a request's own prefill". This is live available memory, lower
  than during the morning's W7 run (1.0 GB).
- Since MC3c, `prepare` divides that margin by `min(concurrent, turns running or queued + 1)`
  (`internal/serveapp/openai.go`, `AdmitPrefillMemoryShare`). With 4 generations in flight, a ~1000-token 7B prompt
  was priced against a quarter of it.
- The division fits MC3c's CPU workers, which prefill at the same time. **Under MC3 on a Metal resident, prefills run
  one at a time** in the exclusive section. The margin is also already read from live memory, so it counts what the
  others hold.
- So the 413 refused requests that would most likely have fit. Whether they would fit was not measured.
- Pair 2's batch cell passed, so whether a request is refused depends on the machine's live memory at that moment.
- It does not touch this record's graded cells (the 1.5B never hit it), or W7 on the 7B (short suffixes).

## Update 2026-09-27: §5's 413 fixed; the end-to-end check could not be run

**The fix.** `prepare` no longer divides the prefill-memory margin for a request that prefills on a GPU resident.
- CPU workers (MC3c) keep the division, because they prefill at the same time.
- On a resident, prefill passes run one at a time, and each pass's scratch is released when it ends (`metal/prefill.go`,
  fix C5, after the command buffer completes). So nothing accumulates across them. The margin is live memory, which
  already excludes what the other generations hold.
- `TestPrepare_prefillShare` (`internal/serveapp`, `goinfer_testhooks`) drives `prepare` with 3 generations ahead at
  concurrency 4 and twice a lone request's need. It found that need by bisection through the real check.
  - A CPU model is refused: the split is kept.
  - A resident model is admitted. This case was red before the fix: 0.25 GB needed, priced against 0.13 GB.
  - A lone resident request with half its need is still refused.

**The end-to-end check was attempted and does not answer.** The 7B's 4-client copy cell was run on both builds, 10:36–10:43
([`fix413-check-run.log`](spec-vs-batching-metal-2026-09-27/fix413-check-run.log),
[`fix413-check-servers.log`](spec-vs-batching-metal-2026-09-27/fix413-check-servers.log),
[`fix413-check.json`](spec-vs-batching-metal-2026-09-27/fix413-check.json)).
- The Mac was already short of memory: 34–37% free, and 617 MB of swap in use before the first server. The morning's
  cells had had more room.
- Swap reached 3.77 GB, and serve's swap guard tripped in both servers.
  - The old build tripped during load and warm-up, before any request ran, and answered all 8 with 503.
  - The fixed build served 4 of 8 while paging (2.7 tok/s), then tripped.
- So whether §5's refused requests would have fit was not measured. It needs a rerun with the memory headroom the
  09:39 cells had.
- The swap guard refused requests rather than letting the machine page further, which is its job.
