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
- **Rerun pre-registered and queued, 2026-09-28:** [`fix413-e2e-2026-09-28.md`](fix413-e2e-2026-09-28.md).
  - Both arms are built from one commit and differ only by the fix.
  - A memory ballast holds available memory in the 1.0–1.6 GB band where the arms disagree.
  - Validity rules and a do-nothing check (the unfixed build must refuse) were written first.

## Update 2026-09-27: §4's divergence found and fixed

**The cause was the turn boundary, not speculation.**
- `TestSpecVerify_forwardNMatchesForward` (`metal/`) shows spec's verify is lossless on Metal. Every round runs
  `ForwardN`, since Metal has no argmax-only verify. After the same 300-token prefix, 8 tokens as one `ForwardN` and
  as 8 `Forward` calls give 0 differing logits and 0 differing K/V, at M = 8 and at M = 1.
- `TestSpecNgram_multiTurnMatchesPlain` (`metal/`) replays two W7 turns in-process.
  - Turn 0 is identical, and so is every K/V element both arms hold after it.
  - Turn 1 diverged at token 44, with spec reusing 150 positions against plain's 151.
- Plain decode forwards every token it emits, the last included. The n-gram loop forwards a round's trailing token
  only as the next round's first row. So a generation that stopped at `max_tokens` left the cache one token short.
  That is safe, and the loop's own comment said so.
- The next turn re-prefills that one position. Metal's fast-prefill floor counts the whole prompt (start + M), so
  even a ~27-token suffix runs the f16-MMA prefill, which is not bit-identical to decode.
- That one position's K/V differs, and every later position attends to it. So every later turn diverged.

**The fix** (`decoder/spec_ngram.go`, `finishTrailing`) forwards the trailing token at exactly that exit: the one
forward a one-row round would make.
- A stop, a cancel or a full context takes no forward, as in plain decode.
- The CPU `Session` path commits the token too, so its reconcile invariant (prompt + every token in the cache) holds.
- `TestGenNgramInto_residentCommitMatchesPlain` (`decoder/`, the fake resident, runs in CI) asserts equal next-turn
  reuse, spec against plain, at five lengths. All five are one short with the fix reverted.
- Through serve (`serve-metal` at `5a92348a` plus the fix, W7, `-spec ngram`), every turn now matches the batch arm's
  reply: 6 / 6 at 1 client, 24 / 24 at 4 clients, reuse identical (157, 311, 466 …). The rate is unchanged: 69.3 and
  72.4 tok/s, against 67.6–71.3 and 69.8–72.8 before.
- Raw: [`specfix-check-chat.json`](spec-vs-batching-metal-2026-09-27/specfix-check-chat.json),
  [`specfix-check-servers.log`](spec-vs-batching-metal-2026-09-27/specfix-check-servers.log).

**§3's decision stands.** The fix restores identity and costs one forward per generation, which plain decode pays too.
Spec still gains a lone Metal request nothing, and still forfeits batching under load.

**Not checked:** the block-drafter (`--drafter`, CUDA), grammar-fused and two-model speculative loops may have the same
trailing-token shape. They were not measured here.

## Update 2026-09-27: the other speculation loops checked for the same gap

The n-gram fix above left one question open: whether the other loops end a generation the same way. Each was read
at its exit, and each finding below is tested.

| loop | where it runs | at a `max_tokens` exit | status |
|---|---|---|---|
| **block drafter** (`--drafter`, `BlockSpec.generate`) | CUDA resident | **committed one position more than it wrote.** The seed and every round end on the target's own token, emitted but not forwarded (it is the next round's anchor). `residentCommitIDs(prompt, out)` recorded it anyway. So the next turn reused that position as it stood: a rejected draft's K/V, or nothing at `max_tokens` 1. That is a correctness bug, not only an identity one. | **fixed** |
| grammar-fused (`genGrammarInto`) | CPU sessions (greedy constrained requests with `--spec`) | the n-gram loop's old shape: one token short, and consistent | **fixed**, for parity with plain decode |
| two-model (`GenerateSpeculative`) | resident or CPU; chat's `--draft`, not in serve | forgets the resident cache first and never commits it, so the next turn cold-prefills by design | unchanged |

**The block drafter.**
- `TestBlockSpecGenerate_commitsOnlyWrittenPositions` records every position the target writes, and asserts
  that every position the commit claims was written.
  - Before the fix it fails at every exit it runs: the seed only (`max_tokens` 1) and after one and two rounds. Each
    claims one unwritten position.
  - The fix forwards the trailing token at that exit: the M = 1 step the guard's fallback takes, with its capture
    folded into the drafter's context while the seam is armed, as a round's rows are. With no room in the context,
    or on a forward error, it commits only what was written.
- Two existing tests had encoded the old behaviour.
  - `TestBlockSpecGenerate_commitsResIDsOnFullCompletion` asserted `resIDs` = prompt + out while its stub host wrote
    only the prompt. It still passes: the commit is now also true.
  - The two drafter-reuse tests had counted the host's calls. They now read each generation's seed call past its
    trailing-token forward.
- **The CUDA check, RUN 2026-09-28** (`docs/prompts/nobara-cuda-spec-trailing-token-2026-09.md`; nobara-pc, RTX
  2070 SUPER 8 GB, driver 595.91.07, idle before starting — load 0.18/0.37/0.55, no other GPU process). Logs:
  [`cuda-step1-existing-gates-2026-09-28.log`](spec-vs-batching-metal-2026-09-27/cuda-step1-existing-gates-2026-09-28.log),
  [`cuda-step2-twoturn-withfix-2026-09-28.log`](spec-vs-batching-metal-2026-09-27/cuda-step2-twoturn-withfix-2026-09-28.log),
  [`cuda-step2-twoturn-reverted-2026-09-28.log`](spec-vs-batching-metal-2026-09-27/cuda-step2-twoturn-reverted-2026-09-28.log).
  - **1. The existing CUDA gates, with the fix in:** `TestGenerateBlockSpec_production`, `TestBlockSpecStream`,
    `TestFlashDecodeBlockSpecLane`, `TestFlashDecodeSpeculativeScope`, `TestFlashDecodeTwoModelSpecLane` — **5/5
    pass**. No regression in the trailing forward (`PrefillLastNArgmax` at M=1, the one-row capture fuse).
  - **2. New `cuda/spec_twoturn_test.go`** (`TestBlockSpec_twoTurnsMatchPlain`,
    `TestNgramSpec_twoTurnsMatchPlain`), qwen3-4b int4 + the real DFlash drafter, N ∈ {1, 17, 48}, run twice:
    - **With the fix (HEAD): 14/14 pass.** Every arm — n-gram, and the block drafter continued through both plain
      `Generate` and the drafter's own `GenerateStream` again — matches plain decode exactly: turn-1 losslessness,
      turn-2 `PrefillReused` equal to plain's and to len(prompt1)+len(out1), turn-2 ids equal, at every N.
    - **Reverted** (`decoder/blockspec.go` from `0e579400^`, `decoder/spec_ngram.go` from `97615930^`): **11/14
      fail**, exactly the shape the fixes describe, not a different failure:
      - **n-gram: `PrefillReused` is exactly one short at every N** (23/24, 39/40, 70/71) — the documented
        one-token gap, reproduced precisely.
      - **Block drafter: the reused COUNT stays correct, but turn 2's actual ids diverge** at N=17 (token 6 of 17)
        and N=48 (token 28 of 48) — the bug's own description, "reused that position as it stood": the position is
        still counted as reused, but its K/V is stale rather than what the trailing forward would have written.
      - **N=1 (the seed-only exit) passes even reverted, for the block drafter only.** Not a test gap: the n-gram
        arm at N=1 still shows its usual one-short count (23 vs 24) in the same run, so the harness clearly can
        see the bug when it's there. The seed-only exit's own stale position happens not to corrupt this
        particular continuation — a real, narrower boundary than the prose "or nothing at max_tokens 1" implied,
        worth knowing if `--drafter` is ever graded specifically at very short generations.
    - One test-harness-only finding, not a production bug: `BlockSpec.GenerateStream` never sets
      `Generation.PrefillReused` at all (confirmed by grep — no occurrence in `decoder/blockspec.go`), so the
      turn2-via-`GenerateStream` variant reads 0 regardless of the fix. That variant's `PrefillReused` assertion
      is skipped; its ids-equality check (the one that actually proves the KV is correct) still ran and passed at
      HEAD, failed when reverted, same as every other arm.
  - Full tagged CUDA suite after restoring both files: 177 pass, 0 fail (same baseline as before this check).
    `gofmt`, `go vet` and the pinned staticcheck 0.8.0 clean.
  - `TestDrafterVsOff_perSuite` (the optional timing comparison) was not run — this check was about correctness,
    not speed, and the two-turn tests already answer the question the prompt asked.

**The grammar-fused loop.** `TestGrammarSpec_sessionTwoTurnsMatchPlain` (the 0.5B on CPU; it skips without
`GOINFER_PREQUANT_GGUF`) ends turn 0 at `max_tokens` 6.
- With the fix reverted, the session holds 25 tokens under spec against plain's 26. The turn-1 reply still matched.
- So on CPU the gap cost one re-prefilled token, not identity. This is consistent with CPU prefill being bit-identical
  to decode here. With the fix, both hold 26.
