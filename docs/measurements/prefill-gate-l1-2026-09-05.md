# L1 §3 fidelity gate — Metal batched prefill vs sequential decode (2026-09-05)

**Verdict: L1 does not ship.** The §3 gate (`task-prefill-gap.md` §3) fails on both models tested,
at every depth tested. Metal's default stays the sequential (bit-identical-to-decode) prefill path;
`--metal-fast-prefill` / `GOINFER_METAL_BATCHED_PREFILL=1` remains the documented, disclosed opt-in.
Phase 2 (flipping the default, building the unifying `--exact-prefill` flag) does not proceed.

## Provenance

- **Machine:** MacBook Pro, Apple M1 Pro, macOS 26.6.2 (Darwin 25.6.0 arm64).
- **goinfer:** `3e3984a` (S run) / `6022b29` (D7 run + `TestPrefillTTFT`) — both on `main`, no other
  local changes. The two commits differ only by a test-harness bugfix and a subtest refactor (see
  *What broke along the way*, below); neither touches decoder/backend production code, so the two
  runs are the same measurement.
- **Checkpoints, both from `~/models` (local NVMe/SSD, the approved bench surface — not
  `/srv/models` or `/Volumes/`):**
  - **S** = `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (1.5B)
  - **D7** = `qwen2.5-7b-instruct-q4_k_m.gguf` (7B)
- **Backend/quant:** `metal`, `int4`, both arms (exact and fast) on the same resident model instance
  per cell.
- **Sampling:** none — every position is scored by argmax over full logits (greedy-equivalent by
  construction, no temperature/top-p/top-k anywhere in the harness).
- **Thermal note:** NOT instrument-read this run — no `powermetrics`/`pmset` sample was taken
  during the ~2-hour combined run. The MacBook was on AC power throughout. This is a real
  methodology gap (`benchmarks.md`'s own convention calls for a thermal note); flagged here rather
  than silently omitted, per this repo's provenance rule.
- **Test:** `metal/prefill_gate_test.go` (`TestPrefillGate`), `GOINFER_HEAVY_TESTS=1`. Command used
  for D7 (S's command was identical, `-run` differs):
  ```
  GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run 'TestPrefillGate/D7|TestPrefillTTFT' -v -timeout 4h
  ```
- **Raw logs** (this directory): `prefill-gate-d7-2026-09-05.log` (complete, 179 lines, every
  per-prompt line for D7 plus the full `TestPrefillTTFT` output) and
  `prefill-gate-metal-2026-09-05.log` (does **not** hold the real S data — see *What broke along
  the way*).

## Method

For each model, for each K ∈ {256, 1024, 3900}, over 10 realistic prose prompts (real repository
documentation read at run time, tokenized and truncated to exactly K tokens — not
`scripts/prompts.json`'s word-repetition filler, which §0 of `task-prefill-gap.md` rules out for
anything content-dependent):

1. **EXACT** — sequential `Forward` per token over the K-token prompt (today's shipped default),
   capturing the last position's logits (the "seed"). Then 64 more greedy steps
   (`argmax` → next input), building a 64-token reference continuation and that continuation's own
   per-position logits.
2. **FAST** — one batched `PrefillLast` call over the same K embeddings
   (`GOINFER_METAL_BATCHED_PREFILL=1`), giving its own seed logits. Then the SAME 64 reference
   tokens from step 1 are teacher-forced through `Forward` one at a time (i.e. the fast path is fed
   the exact path's own tokens at each step, not its own predictions) — this is what isolates each
   position's own quality from the cascade a free-running greedy comparison would carry.
3. **Score:**
   - *Seed-logit argmax agreement* — `decoder.NearTieArgmaxForTest` between the two seeds: agrees,
     or a "near-tie" flip (gap ≤ 3% of the exact logits' own range, the same rule
     `cuda/realforward_test.go`'s `argmaxF` comparison and `gpu/kv_i8_parity_test.go` already use),
     or a hard fail (gap > 3%). **Gating.**
   - *Teacher-forced top-1 agreement* — `decoder.TeacherForcedTop1AgreementForTest` over the 64
     continuation positions: fraction where the fast path's argmax equals the reference token.
     **Reported.** Separately, each of the 64 positions is ALSO scored against the exact path's own
     logits at that position via the same near-tie rule as the seed check, and a position is a hard
     fail only if it exceeds the 3% bar. **This is a substitute gate**, not the doc's originally
     specified bar — see *A gap in the gate itself*, below.
   - *KL divergence* — `decoder.KLDivergenceForTest(exact, fast)` at the seed and (mean) over the
     64 continuation positions. **Reported, not gating**, per §3.
   - *Greedy stream divergence* — **not re-measured**. `TestMetalPrefillDivergenceRate` already
     measures this (54%, cited in `metal/backend.go`'s `PrefillLast` decline comment, §A2-Metal).
     Re-running it would duplicate a test this gate only reports the existing number from.

## A gap in the gate itself, found while building it

§3's table specifies the teacher-forced check's bar as "≥ the CUDA-decode-vs-CPU figure on the same
model." That figure does not exist anywhere in this tree — confirmed by exploration before writing
any code, and unchanged after: teacher-forced top-1 agreement is a new metric, built for this gate,
and nothing on the CUDA side has measured it yet either. Substituting a made-up number would be
worse than saying so, so this run instead scores the continuation positions with the SAME 3%
near-tie rule the seed check already uses — i.e. the only bar this repo already trusts is applied
uniformly across every position instead of just the last one. The raw agreement rate is reported
alongside every cell so this substitution is visible and the real cross-backend bar can replace it
once CUDA measures its own teacher-forced-vs-CPU number.

## Results — §3 gate

| model | K | seed hard-fails | worst seed gap | continuation hard-fails | worst cont. gap | mean teacher-forced agreement | mean seed KL | mean cont. KL | verdict |
|---|---|---|---|---|---|---|---|---|---|
| S (1.5B) | 256 | 0/10 | 0.617% | 1/640 | 4.871% | 96.4% | 0.0137 | 0.0085 | **FAILED** |
| S (1.5B) | 1024 | 0/10 | 0.618% | 2/640 | 4.818% | 96.6% | 0.0095 | 0.0074 | **FAILED** |
| S (1.5B) | 3900 | 0/10 | 0.759% | 1/640 | 3.653% | 96.6% | 0.0124 | 0.0077 | **FAILED** |
| D7 (7B) | 256 | 0/10 | 1.757% | 6/640 | 5.740% | 86.7% | 0.0720 | 0.0555 | **FAILED** |
| D7 (7B) | 1024 | 0/10 | 1.229% | 6/640 | 4.546% | 89.2% | 0.0909 | 0.0500 | **FAILED** |
| D7 (7B) | 3900 | 0/10 | 2.701% | 4/640 | 4.098% | 89.2% | 0.0748 | 0.0591 | **FAILED** |

**Seed-logit agreement passes cleanly everywhere** — 0/10 hard fails at all six cells, worst gap
2.701% (D7, K=3900), under the 3% bar. This is the one check the doc's own bar is real and
established for, and Metal's batched prefill clears it: the LAST prompt position, on its own, is
fine.

**The continuation check fails at every cell**, and the failure mode is worse at the larger model:
D7's mean teacher-forced agreement (86.7–89.2%) is markedly lower than S's (96.4–96.6%), and D7's
mean KL divergence is roughly 6–7× S's at every depth. The batched path's f16-activation KV state
diverges from sequential decode's just enough that, fed forward through even a few teacher-forced
decode steps, it produces a real (not near-tie) top-1 flip 4–6 times per 640 scored positions on
D7, 1–2 times on S. Depth (K) does not obviously worsen this within a model — S is flat across
256/1024/3900, D7 is roughly flat too (6, 6, 4 hard-fails) — the model-size effect dominates
whatever depth effect there might be.

### D7, per-prompt (full detail — see *What broke along the way* for why S has no equivalent table)

| K | prompt | seed | seed gap | cont. agree | cont. hard-fails | first divergence | seed KL | cont. KL |
|---|---|---|---|---|---|---|---|---|
| 256 | 1 | AGREE | 0.000% | 92.2% | 1/64 | 5 | 0.0698 | 0.0443 |
| 256 | 2 | FLIP | 1.757% | 84.4% | 0/64 | 0 | 0.0987 | 0.0602 |
| 256 | 3 | FLIP | 0.110% | 81.2% | 0/64 | 0 | 0.0532 | 0.0649 |
| 256 | 4 | FLIP | 1.679% | 87.5% | 1/64 | 0 | 0.1112 | 0.0468 |
| 256 | 5 | AGREE | 0.000% | 92.2% | 0/64 | 5 | 0.0120 | 0.0338 |
| 256 | 6 | FLIP | 1.747% | 89.1% | 0/64 | 0 | 0.1129 | 0.0521 |
| 256 | 7 | AGREE | 0.000% | 76.6% | 2/64 | 3 | 0.0787 | 0.0827 |
| 256 | 8 | AGREE | 0.000% | 79.7% | 1/64 | 3 | 0.0112 | 0.0510 |
| 256 | 9 | AGREE | 0.000% | 90.6% | 1/64 | 6 | 0.0351 | 0.0681 |
| 256 | 10 | AGREE | 0.000% | 93.8% | 0/64 | 17 | 0.1376 | 0.0511 |
| 1024 | 1 | AGREE | 0.000% | 87.5% | 1/64 | 4 | 0.0760 | 0.0483 |
| 1024 | 2 | FLIP | 1.229% | 81.2% | 0/64 | 0 | 0.1849 | 0.0629 |
| 1024 | 3 | FLIP | 1.204% | 81.2% | 1/64 | 0 | 0.0735 | 0.0722 |
| 1024 | 4 | AGREE | 0.000% | 90.6% | 0/64 | 3 | 0.0467 | 0.0434 |
| 1024 | 5 | AGREE | 0.000% | 93.8% | 0/64 | 1 | 0.0003 | 0.0328 |
| 1024 | 6 | AGREE | 0.000% | 92.2% | 0/64 | 12 | 0.0001 | 0.0592 |
| 1024 | 7 | AGREE | 0.000% | 89.1% | 1/64 | 4 | 0.0705 | 0.0410 |
| 1024 | 8 | AGREE | 0.000% | 96.9% | 0/64 | 1 | 0.1279 | 0.0328 |
| 1024 | 9 | AGREE | 0.000% | 95.3% | 0/64 | 1 | 0.1618 | 0.0408 |
| 1024 | 10 | AGREE | 0.000% | 84.4% | 3/64 | 6 | 0.1676 | 0.0663 |
| 3900 | 1 | AGREE | 0.000% | 98.4% | 0/64 | 8 | 0.0399 | 0.0110 |
| 3900 | 2 | FLIP | 2.701% | 90.6% | 0/64 | 0 | 0.0873 | 0.0579 |
| 3900 | 3 | FLIP | 0.727% | 79.7% | 1/64 | 0 | 0.1152 | 0.0832 |
| 3900 | 4 | AGREE | 0.000% | 92.2% | 0/64 | 6 | 0.0536 | 0.0611 |
| 3900 | 5 | AGREE | 0.000% | 90.6% | 0/64 | 18 | 0.0431 | 0.0448 |
| 3900 | 6 | AGREE | 0.000% | 82.8% | 1/64 | 3 | 0.1090 | 0.0805 |
| 3900 | 7 | AGREE | 0.000% | 85.9% | 1/64 | 3 | 0.1472 | 0.0904 |
| 3900 | 8 | AGREE | 0.000% | 96.9% | 0/64 | 30 | 0.0511 | 0.0406 |
| 3900 | 9 | AGREE | 0.000% | 89.1% | 0/64 | 4 | 0.0028 | 0.0502 |
| 3900 | 10 | AGREE | 0.000% | 85.9% | 1/64 | 1 | 0.0989 | 0.0710 |

Note the seed FLIPs that are near-ties (e.g. K=256 prompt 3, gap 0.110%) still show a real
continuation cost (81.2% agreement) — the seed check and the continuation check are measuring
different things, and a near-tied seed is not a guarantee the KV it produced is harmless downstream.

## Results — TTFT (speed side, `TestPrefillTTFT`, S only, this commit)

| P | sequential | batched | speedup |
|---|---|---|---|
| 256 | 3463.32 ms | 880.79 ms | **3.93×** |
| 1024 | 14705.25 ms | 4718.41 ms | **3.12×** |
| 3900 | 74157.50 ms | 36739.51 ms | **2.02×** |

`task-prefill-gap.md`'s pre-registered band (§4 L1) was **≥3× ships, <2× reopens as a serving
investigation**, set against a previously measured 3.9× figure treated as roughly flat across
prompt length. It is not flat: the speedup decays monotonically with depth here, and K=3900 —
exactly the depth this doc's own W4 workload cares about most — lands at 2.02×, a hair above the
reopen line rather than confirming the ship band. Speed alone would be ambiguous-to-reopening at
the depth that matters; combined with the quality gate's clean failure, there is no ambiguity in
the overall verdict.

## What broke along the way (worth recording so it isn't rediscovered)

Three infrastructure problems surfaced while running this gate, none of them about the numbers
above but all of them shaping what could be measured and how:

1. **A real bug in the test itself, found on the first run.** `metalResident.Forward`'s return is a
   reused buffer ("consume before the next call" — the doc comment says so). The first version of
   this test stored returned slices directly instead of cloning them, so by the time they were
   compared later every captured logits vector had been silently overwritten by a subsequent call.
   This produced a self-contradictory first result — a 42%-gap seed "FLIP" while the logically
   identical continuation-position-0 check quietly agreed — which is what caught it. Fixed by
   cloning every logits slice at capture time (commit `3e3984a`).
2. **A cross-session collision mid-run.** Another session renamed `docs/benchmarks-archive.md` to
   `docs/legacy-benchmarks.md` (a docs restructuring, unrelated to this gate) while this test's
   hardcoded prose-fixture list still named the old path. The rename landed between S's run
   finishing and D7's starting, so D7's prompt-loading step failed instantly on
   `open ../docs/benchmarks-archive.md: no such file or directory` — the FIRST attempt at D7 never
   produced a single measurement. Fixed by pointing the fixture list at the new path (commit
   `6022b29`), and by refactoring the test into per-model subtests (`t.Run`) so a future interrupted
   run can re-target just the failed model instead of repeating a model that already has a clean
   result.
3. **`launchctl submit`-created jobs auto-restart on exit** (undocumented, discovered the hard way):
   the completed S+D7(broken)+`TestPrefillTTFT` run's log was silently overwritten by a fresh,
   immediately-failing re-invocation of the same command within moments of the original process
   exiting, before it could be read. **This is why the real S per-prompt trace no longer exists as
   a file** — only the per-K aggregate lines (the "Results" table above) were captured live via
   direct reads *during* the run, before the overwrite happened; `prefill-gate-metal-2026-09-05.log`
   on disk holds the stale, failed re-invocation's output, not the real run. The D7 rerun avoided
   this by having the submitted job remove its own launchd registration as its last action
   (`; launchctl remove $LABEL` appended to the command), closing the race before launchd could
   restart it — that log (`prefill-gate-d7-2026-09-05.log`) is complete and is what backs the
   per-prompt table above.

None of these affect the verdict: the S aggregate numbers were captured from the real, live run
before the overwrite, and D7's are complete and file-backed.

## What this settles, and what it doesn't

- **Settled:** L1 does not ship as built. Metal's batched prefill stays opt-in
  (`--metal-fast-prefill` / `GOINFER_METAL_BATCHED_PREFILL=1`), the shipped default stays the
  sequential path, and `docs/task-prefill-gap.md`'s Phase 2 (default flip, `--exact-prefill` flag)
  does not proceed from this doc's plan.
- **Not settled:** WHY the continuation check fails — this run measures that it does, not the
  mechanism. `docs/task-prefill-gap.md` §2.2/§2.3 already names the candidate cause (f16-activation
  MMA vs int8-activation decode), but that is the pre-existing hypothesis, not something this run
  isolated further.
- **Not settled:** the real "≥ the CUDA-decode-vs-CPU figure" bar §3 specifies — this run's
  continuation gate is a documented substitute (the seed check's own 3% rule, applied per-position),
  not the doc's original bar. Building that real figure needs the same scorer run on CUDA against
  CPU, which is out of scope for a Metal-only session.
- **Not run:** W2's Mac cells (`scripts/bench_peer_prefill.py` still has no Metal backend option) —
  this doc's L1 section asked for that regardless of the gate's outcome, and it remains open.
